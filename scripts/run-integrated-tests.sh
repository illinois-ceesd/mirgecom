#!/bin/bash

#
# Usage: run-integrated-tests.sh <options> [path_to_mirgecom]
#
# This script is designed to run mirgecom tests and examples.
# Options:
#    -e|--examples: Run the examples (default = No)
#    -p|--production: Run the production tests (default = No)
#    -b|--batch: Run tests through a batch system (default = No)
#
# Each driver to test is expected to have a smoke test defined in:
# /driver_name_root_<driver_name>/scripts/smoke_test.sh
#
# See https://github.com/illinois-ceesd/drivers_y2-prediction/scripts/smoke_test.sh
# for an example `smoke_test.sh`.
#
do_examples=false
do_production_tests=false
do_batch_job=false

all_args="$@"

NONOPT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            do_examples=true
            do_production_tests=true
            shift
            ;;
        -e|--examples)
            do_examples=true
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

origin=$(pwd)
MIRGE_HOME=${1:-"${MIRGE_HOME:-}"}
if [[ -z "${MIRGE_HOME}" ]]; then
    . scripts/mirge-testing-env.sh
fi

printf "Running integrated tests.  MIRGE_HOME=${MIRGE_HOME}\n"

testing_env="${MIRGE_HOME}/scripts/mirge-testing-env.sh"
if [[ -z "${MIRGE_PARALLEL_SPAWNER:-}" ]]; then
    printf "Loading MIRGE testing env: ${testing_env}\n"
    . ${testing_env}
fi

declare -i numfail=0
declare -i numsuccess=0

date

echo "Running tests in ${MIRGE_HOME} ..."

failed_tests=""
succeeded_tests=""

if [[ "${do_examples}" = "true" ]]; then

    date
    printf "\- Running Examples.\n"
    ${MIRGE_HOME}/examples/run_examples.sh ${MIRGE_HOME}/examples
    test_result=$?
    date
    if [[ $test_result -eq 0 ]]; then
        ((numsuccess=numsuccess+1))
        printf "\-\- Example tests passed.\n"
        succeeded_tests="${succeeded_tests} Examples"
    else
        ((numfail=numfail+1))
        printf "\-\- Example tests failed.\n"
        failed_tests="${failed_tests} Examples"
    fi
fi

if [[ "${do_production_tests}" = "true" ]]; then

    date
    printf "\- Testing production drivers.\n"
    ${MIRGE_HOME}/scripts/run-production-tests.sh ${MIRGE_HOME}
    test_result=$?
    date
    if [[ $test_result -eq 0 ]]; then
        ((numsuccess=numsuccess+1))
        printf "\-\- Production tests passed.\n"
        succeeded_tests="${succeeded_tests} Production"
    else
        ((numfail=numfail+1))
        printf "\-\- Production tests failed.\n"
        failed_tests="${failed_tests} Production"
    fi
fi

if [[ $numfail -eq 0 ]]
then
    echo "No failures."
else
    echo "Failed tests(${numfail}): ${failed_tests}"
fi

echo "Successful tests(${numsuccess}): ${succeeded_tests}"

exit $numfail

