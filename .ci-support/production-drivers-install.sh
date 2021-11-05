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
PRODUCTION_DRIVERS=${PRODUCTION_DRIVERS:-"illinois-ceesd/drivers_y1-nozzle@parallel-lazy:illinois-ceesd/drivers_y2-isolator@main:illinois-ceesd/drivers_flame1d@main"}
# Loop over the production drivers, clone them, and prepare for execution
set -x
OIFS="$IFS"
IFS=':'; for production_driver_string in $PRODUCTION_DRIVERS;
do
    PRODUCTION_DRIVER_BRANCH=$(printf "$production_driver_string" | cut -d "@" -f 2)
    PRODUCTION_DRIVER_REPO=$(printf "$production_driver_string" | cut -d "@" -f 1)
    PRODUCTION_DRIVER_NAME=$(printf "$PRODUCTION_DRIVER_REPO" | cut -d "/" -f 2)
    PRODUCTION_DRIVER_DIR="production_driver_$PRODUCTION_DRIVER_NAME"
    git clone -b "$PRODUCTION_DRIVER_BRANCH" https\://github.com/"$PRODUCTION_DRIVER_REPO" "$PRODUCTION_DRIVER_DIR"
    cd "$PRODUCTION_DRIVER_DIR"/smoke_test
    ln -s *.py driver.py  #  name the driver generically
    cd ../..
done
IFS="$OIFS"
set +x
