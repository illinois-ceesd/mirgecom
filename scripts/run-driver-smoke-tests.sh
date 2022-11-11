#!/bin/bash

#set -x
#
# Usage: run-driver-smoke-tests.sh [path_to_mirgecom] [driver_name_root]
#
# This script is designed to run the smoke tests for a collection of
# drivers. The drivers are each expected to be in the path:
# /path_to_mirgecom/driver_name_root_<driver_name>
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

DRIVER_ROOT=${2:-"production_driver"}

testing_env="${MIRGE_HOME}/scripts/mirge-testing-env.sh"

declare -i numfail=0
declare -i numsuccess=0

date

echo "Running drivers in ${MIRGE_HOME} matching ${DRIVER_ROOT} ..."

failed_drivers=""
succeeded_drivers=""

for driver in $(ls | grep "${DRIVER_ROOT}_");
do
    
    driver_path="${origin}/${driver}"
    date
    printf "\- Running smoke tests in ${driver_path}.\n"
    . ${driver}/scripts/smoke_test.sh "${testing_env}" "${driver_path}"
    test_result=$?

    if [[ $test_result -eq 0 ]]; then
        ((numsuccess=numsuccess+1))
        printf "\-\- ${driver} smoke tests passed."
        succeeded_drivers="${succeeded_drivers} ${driver}"
    else
        ((numfail=numfail+1))
        printf "\-\- ${driver} smoke tests failed."
        failed_drivers="${failed_drivers} ${driver}"
    fi
    cd ${origin}

done
# set +x
date

if [[ $numfail -eq 0 ]]
then
    echo "No failures."
else
    echo "Failed drivers(${numfail}): ${failed_drivers}"
fi

echo "Successful drivers(${numsuccess}): ${succeeded_drivers}"

return $numfail

