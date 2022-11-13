#!/bin/bash

#
# Usage: run-integrated-tests.sh <options> [path_to_mirgecom]
#
# This script is designed to run mirgecom tests and examples.
# Options:
#    -e|--examples: Run the examples (default = No)
#    -p|--production: Run the production tests (default = No)
#    -l|--lazy-accuracy: Run lazy accuracy tests (default = No)
#    -b|--batch: Run tests through a batch system (default = No)
#
# Each driver to test is expected to have a smoke test defined in:
# /driver_name_root_<driver_name>/scripts/smoke_test.sh
#
# See https://github.com/illinois-ceesd/drivers_y2-prediction/scripts/smoke_test.sh
# for an example `smoke_test.sh`.
#
origin=$(pwd)
MIRGE_HOME=${1:-"."}
cd ${MIRGE_HOME}
MIRGE_HOME=$(pwd)
cd -

testing_env="${MIRGE_HOME}/scripts/mirge-testing-env.sh"

declare -i numfail=0
declare -i numsuccess=0

do_examples=false
do_lazy_accuracy=false
do_production_tests=false
do_batch_job=false

NONOPT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--examples)
            do_examples=true
            shift
            ;;
        -l|--lazy-accuracy)
            do_lazy_accuracy=true
            shift
            ;;
        -p|--production)
            do_production_tests=true
            shift
            ;;
        -b|--batch)
            do_batch_job=true
            shift
            ;;
        -*|--*)
            echo "run_integrated_tests: Unknown option $1"
            exit 1
            ;;
        *)
            NONOPT_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${NONOPT_ARGS[@]}"

date

echo "Running tests in ${MIRGE_HOME} ..."

failed_tests=""
succeeded_tests=""

if [[ "${do_examples}" = "true" ]]; then

    date
    printf "\- Running Examples.\n"
    . ${MIRGE_HOME}/examples/run_examples.sh ${MIRGE_HOME}/examples
    test_result=$?
    date
    if [[ $test_result -eq 0 ]]; then
        ((numsuccess=numsuccess+1))
        printf "\-\- Example tests passed."
        succeeded_tests="${succeeded_drivers} Examples"
    else
        ((numfail=numfail+1))
        printf "\-\- Example tests failed."
        failed_tests="${failed_drivers} Examples"
    fi
fi

if [[ "${do_lazy_accuracy}" = "true" ]]; then

    date
    printf "\- Testing Lazy Accuracy.\n"
    . ${MIRGE_HOME}/examples/test_lazy_accuracy.sh
    test_result=$?
    date
    if [[ $test_result -eq 0 ]]; then
        ((numsuccess=numsuccess+1))
        printf "\-\- Lazy accuracy tests passed."
        succeeded_tests="${succeeded_drivers} LazyAccuracy"
    else
        ((numfail=numfail+1))
        printf "\-\- Lazy accuracy tests failed."
        failed_tests="${failed_drivers} LazyAccuracy"
    fi
fi

if [[ "${do_production_tests}" = "true" ]]; then

    date
    printf "\- Production testing (soon).\n"
    # . ${MIRGE_HOME}/scripts/run-production-tests.sh ${MIRGE_HOME}
    # test_result=$?
    date
    # if [[ $test_result -eq 0 ]]; then
    #     ((numsuccess=numsuccess+1))
    #     printf "\-\- Lazy accuracy tests passed."
    #     succeeded_tests="${succeeded_drivers} LazyAccuracy"
    # else
    #    ((numfail=numfail+1))
    #    printf "\-\- Lazy accuracy tests failed."
    #    failed_drivers="${failed_drivers} LazyAccuracy"
    #fi
fi

if [[ $numfail -eq 0 ]]
then
    echo "No failures."
else
    echo "Failed tests(${numfail}): ${failed_tests}"
fi

echo "Successful tests(${numsuccess}): ${succeeded_tests}"

return $numfail

