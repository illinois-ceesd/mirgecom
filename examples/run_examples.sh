#!/bin/bash

# set -e
set -o nounset

rm -f *.vtu *.pvtu

origin=$(pwd)
examples_dir=${1-$origin}
declare -i numfail=0
declare -i numsuccess=0
echo "*** Running examples in $examples_dir ..."
failed_examples=""
succeeded_examples=""
for example in $examples_dir/*.py
do
    help_text=$(python ${example} --help)
    if [[ "$help_text" == *"--mpi"* ]]
    then
        run_without_mpi=1
        run_with_mpi=1
        mpi_flags="--mpi"
    elif [[ "$example" == *"-mpi"* ]]
    then
        run_without_mpi=0
        run_with_mpi=1
        mpi_flags=""
    else
        run_without_mpi=1
        run_with_mpi=0
    fi
    if [[ "$help_text" == *"--lazy"* ]]
    then
        run_without_lazy=1
        run_with_lazy=1
        lazy_flags="--lazy"
    elif [[ "$example" == *"-lazy"* ]]
    then
        run_without_lazy=0
        run_with_lazy=1
        lazy_flags=""
    else
        run_without_lazy=1
        run_with_lazy=0
    fi
    run_and_check() {
        $@
        if [[ $? -eq 0 ]]
        then
            ((numsuccess=numsuccess+1))
            echo "*** Example $example succeeded."
            succeeded_examples="$succeeded_examples $example"
        else
            ((numfail=numfail+1))
            echo "*** Example $example failed."
            failed_examples="$failed_examples $example"
        fi
        echo ""
    }
    if [[ "$run_without_mpi" -eq 1 && "$run_without_lazy" -eq 1 ]]
    then
        echo "*** Running serial example: $example"
        run_and_check python ${example}
    fi
    if [[ "$run_without_mpi" -eq 1 && "$run_with_lazy" -eq 1 ]]
    then
        echo "*** Running serial lazy example: $example"
        run_and_check python ${example} ${lazy_flags}
    fi
    if [[ "$run_with_mpi" -eq 1 && "$run_without_lazy" -eq 1 ]]
    then
        echo "*** Running parallel example (2 ranks): $example"
        run_and_check mpiexec -n 2 python -m mpi4py ${example} ${mpi_flags}
    fi
    if [[ "$run_with_mpi" -eq 1 && "$run_with_lazy" -eq 1 ]]
    then
        echo "*** Running parallel lazy example (1 rank): $example"
        run_and_check mpiexec -n 1 python -m mpi4py ${example} ${mpi_flags} ${lazy_flags}
    fi
    rm -rf *vtu *sqlite *pkl *-journal restart_data
done
((numtests=numsuccess+numfail))
echo "*** Done running examples!"
if [[ $numfail -eq 0 ]]
then
    echo "*** No errors."
else
    echo "*** Errors detected." 
    echo "*** Failed tests: ($numfail/$numtests): $failed_examples"
fi
echo "*** Successful tests: ($numsuccess/$numtests): $succeeded_examples"
exit $numfail
#rm -f examples/*.vtu
