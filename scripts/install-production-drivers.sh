#!/bin/bash
#
# This script is designed to install the CEESD production tests used to
# check that changes to main don't tear up the production capability.
# The script takes one argument, the path to the mirgecom source,
# To control what production tests are installed, set the following:
#
# PRODUCTION_DRIVERS = ':' delimited list "fork/repo@branch"
# (See the example default value below)
#
MIRGE_HOME=${1:-"${MIRGE_HOME}"}
if [[ -z "${MIRGE_HOME}" ]]; then
    . scripts/mirge-testing-env.sh
fi

if [[ -z "${PRODUCTION_DRIVERS}" ]]; then
    source ${MIRGE_HOME}/scripts/production-testing-env.sh
fi

# Loop over the production drivers, clone them, and prepare for execution
# set -x
OIFS="$IFS"
IFS=':'; for production_driver_string in $PRODUCTION_DRIVERS;
do
    printf "Installing from: ${production_driver_string}\n"             
    . ${MIRGE_HOME}/scripts/install-driver.sh "$production_driver_string" "production_driver"        
done
IFS="$OIFS"
# set +x
