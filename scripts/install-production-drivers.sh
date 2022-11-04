#!/bin/bash
# This script is designed to install the CEESD production test used to
# check that changes to main don't tear up the production capability.
# The script takes one argument, the production environment setup file,
# which is typically `.ci-support/production-env-setup.sh`. To specify
# what production test is installed, the env setup script should set
# the following:
#
# PRODUCTION_DRIVERS = ':' delimited list "fork/repo@branch"
# (See the example default value below)
#
# The default values result in an install of the Y1 nozzle driver and
# Wyatt Hagen's isolator driver that work with current MIRGE-Com 
# production branch: mirgecom@y1-production.
MIRGE_HOME=${1:-"."}
cd ${MIRGE_HOME}
MIRGE_HOME=$(pwd)
cd -

PRODUCTION_DRIVERS=${PRODUCTION_DRIVERS:-"illinois-ceesd/drivers_y2-prediction@overhaul-testing"}
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
