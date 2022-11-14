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
MIRGE_HOME=${1:-"${MIRGE_HOME}"}
if [[ -z "${MIRGE_HOME}" ]]; then
    . scripts/mirge-testing-env.sh
fi

if [[ -z "${MIRGE_PARALLEL_SPAWNER}" ]]; then
    source ${MIRGE_HOME}/scripts/mirge-testing-env.sh
fi

cd ${MIRGE_HOME}

date

printf "Running production tests in ${MIRGE_HOME} ...\n"

if [[ -z "${MIRGE_PRODUCTION_INSTALL}" ]]; then
    
    printf "... Installing production branch ...\n"
    . scripts/merge-install-production-branch.sh
    date
fi
echo "rpt reset 2: MIRGE_HOME=${MIRGE_HOME}"

printf "... Installing production drivers ...\n" 
. scripts/install-production-drivers.sh
date

printf "... Running production driver smoke tests ...\n"
. scripts/run-driver-smoke-tests.sh . production_driver

retval=$?

printf "Production tests done.\n"
date

cd ${origin}

exit $retval

