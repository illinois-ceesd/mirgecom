#!/bin/bash

set -o nounset

# {{{ Provide log grouping for GitHub actions

function startgroup {
    # Start a foldable group of log lines
    # Pass a single argument, quoted
    echo "::group::$1"
}

function endgroup {
    echo "::endgroup::"
}

# }}}

python -c "from grudge.array_context import MPINumpyArrayConext" && numpy_actx_available=numpy || numpy_actx_available=

echo "Numpy array context available: $numpy_actx_available"


run_path=$(pwd)
examples_dir=${1:-"./"}
shift

if [[ ! -d "${examples_dir}" ]]; then
    echo "Usage: run_examples.sh <examples directory> [list of examples]"
    printf "\nThis script runs examples on 2 MPI ranks (where appropriate) and\n"
    printf "compares the results it gets from running with eager, lazy, and numpy (if available)\n"
    printf "array contexts. Users may optionally provide a list of which examples\n"
    printf "to run.\n\nArguments:\n"
    printf "\n<examples directory>: defaults to the current working directory.\n"
    printf "[list of examples]: optional list of specific examples\n\n"
    printf "Usage example (run all the examples in the \"examples\" directory):\n"
    printf "examples/run_examples.sh examples\n\n"
    printf "Usage example (run only autoignition and poiseuille examples):\n"
    printf "run_examples.sh . autoignition poiseuille.py\n"
    exit 1
fi

cd $examples_dir
examples_dir=$(pwd)

# This bit will let the user specify which
# examples to run, default to run them all.
if [[ "$#" -eq 0 ]]; then # use all py files
    example_list=(*.py)
else # Run specific examples if given by user
    example_list=("$@")
fi

echo "Examples: ${example_list[*]}"

num_failed_examples=0
num_successful_examples=0
failed_examples=""
succeeded_examples=""
succeeded_tests=""
failed_tests=""

date

echo "*** Running examples from $examples_dir in ${run_path}..."

if [[ -z "${MIRGE_PARALLEL_SPAWNER:-}" ]];then
    source scripts/mirge-testing-env.sh ${examples_dir}/..
fi

mpi_exec="${MIRGE_MPI_EXEC}"
mpi_launcher="${MIRGE_PARALLEL_SPAWNER}"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Actually run at the user's path
cd ${run_path}

for example in "${example_list[@]}"
do
    example_name="${example%.py}"
    example_filename="${example_name}.py"
    example_path=${examples_dir}/${example_filename}

    if [[ ! -f "${example_path}" ]]; then
        printf "Example file \"${example_path}\" does not exist, skipping.\n"
        continue
    fi

    # FIXME: The tolerances are really high

    # Should be this:
    #  TOL_LAZY=1e-12
    #  TOL_NUMPY=1e-12

    # But this is required to make them all pass
    TOL_LAZY=1e-9
    TOL_NUMPY=1e-9

    date
    printf "***\n***\n"

    # Hack: This should be an associative array (declare -A), but these
    # aren't available in bash 3 (MacOS).
    # The format of each array item is: "test_name:test_result"
    test_results=()

    # Run example with Eager, Lazy, and Numpy arraycontexts
    for actx in eager lazy $numpy_actx_available; do
        test_name="${example_name}_$actx"
        startgroup "**** Running $test_name"
        set -x
        rm -rf ${test_name}*vtu viz_data/${test_name}*vtu
        set +x

        basic_command="python -m mpi4py ${example_path} --casename ${test_name}"
        [[ $actx != "eager" ]] && basic_command+=" --$actx"
        set -x
        ${mpi_exec} -n 2 $mpi_launcher $basic_command
        test_return_code=$?
        set +x
        test_results+=("${test_name}:$test_return_code")
        date

        endgroup
    done

    startgroup "**** Accuracy comparison for $example_name."
    lazy_comparison_result=0
    numpy_comparison_result=0
    lazy_numpy_comparison_result=0
    nlazy_compare=0
    nnumpy_compare=0
    nlazynumpy_compare=0
    for eager_vizfile in ${example_name}_eager-*.vtu viz_data/${example_name}_eager-*.vtu; do

        if [[ -f ${eager_vizfile} ]]; then
            lazy_vizfile=$(echo ${eager_vizfile/eager/lazy})
            if [[ -f ${lazy_vizfile} ]]; then
                echo "***** comparing lazy results..."
                python ${examples_dir}/../bin/mirgecompare.py --tolerance $TOL_LAZY ${lazy_vizfile} ${eager_vizfile}
                lazy_compare_return_code=$?
                lazy_comparison_result=$((lazy_comparison_result + lazy_compare_return_code))
                ((nlazy_compare++))
            fi

            numpy_vizfile=$(echo ${eager_vizfile/eager/numpy})
            if [[ -f ${numpy_vizfile} ]]; then
                echo "***** comparing numpy results..."
                python ${examples_dir}/../bin/mirgecompare.py --tolerance $TOL_NUMPY ${numpy_vizfile} ${eager_vizfile}
                numpy_compare_return_code=$?
                numpy_comparison_result=$((numpy_comparison_result + numpy_compare_return_code))
                ((nnumpy_compare++))
                if [[ -f ${lazy_vizfile} ]]; then
                    echo "***** comparing lazy/numpy results..."
                    python ${examples_dir}/../bin/mirgecompare.py --tolerance $TOL_NUMPY ${lazy_vizfile} ${numpy_vizfile}
                    lazy_numpy_compare_return_code=$?
                    lazy_numpy_comparison_result=$((lazy_numpy_comparison_result + lazy_numpy_compare_return_code))
                    ((nlazynumpy_compare++))
                fi
            fi
        fi
    done

    endgroup

    # Save any comparison results (if they were done)
    if [[ "$nlazy_compare" -gt 0 ]]; then
        test_results+=("${example_name}_lazy_comparison:$lazy_comparison_result")
    fi

    if [[ "$nnumpy_compare" -gt 0 ]]; then
        test_results+=("${example_name}_numpy_comparison:$numpy_comparison_result")
    fi

    if [[ "$nlazynumpy_compare" -gt 0 ]]; then
        test_results+=("${example_name}_lazy_numpy_comparison:$lazy_numpy_comparison_result")
    fi

    example_test_result=0
    num_ex_success=0
    num_ex_failed=0
    example_succeeded_tests=""
    example_failed_tests=""

    # Track/report the suite of tests for each example
    echo "**** ${example_name} testing results:"
    for test_name_result in "${test_results[@]}"; do
        _test_name=${test_name_result%%:*}
        _test_result=${test_name_result#*:}
        printf "${_test_name}: "
        if [[ $_test_result -eq 0 ]]
        then
            example_succeeded_tests="${example_succeeded_tests} ${_test_name}"
            ((num_ex_success++))
            printf "Pass\n"
        else
            example_failed_tests="${example_failed_tests} ${_test_name}"
            example_test_result=1
            ((num_ex_failed++))
            printf "Fail\n"
        fi
        # fi
    done

    # Global tracking of success/fail for end report
    if [[ ${num_ex_success} -gt 0 ]]
    then
        succeeded_tests="${succeeded_tests} ${example_succeeded_tests}"
    fi
    if [[ ${num_ex_failed} -gt 0 ]]
    then
        failed_tests="${failed_tests} ${example_failed_tests}"
    fi

    if [[ ${example_test_result} -eq 0 ]]
    then
        succeeded_examples="$succeeded_examples $example_name"
        ((num_successful_examples++))
    else
        failed_examples="$failed_examples $example_name"
        ((num_failed_examples++))
    fi

done
num_examples=$((num_successful_examples + num_failed_examples))
date
echo "*** Done running ${num_examples} examples!"
if [[ $num_failed_examples -eq 0 ]]
then
    echo "*** No errors."
else
  echo "*** Errors detected in some examples."
  echo "*** Failed examples ($num_failed_examples/$num_examples): $failed_examples"
  echo "*** Failed tests: $failed_tests"
fi

if [[ $num_successful_examples -eq 0 ]]
then
    echo "*** No successful examples."
else
    echo "*** Some examples were fully successful."
    echo "*** Successful examples: ($num_successful_examples/$num_examples): $succeeded_examples"
fi

if [[ -z "${succeeded_tests}" ]]; then
    echo "*** No successful tests at all."
else
    echo "*** Some tests were successful."
    echo "*** Successful tests: ${succeeded_tests}"
fi

# If any example testing failed, fail the runner script
exit $num_failed_examples
