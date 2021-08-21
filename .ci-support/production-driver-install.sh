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
#
set -x

DEVELOPMENT_BRANCH="$GITHUB_HEAD_REF"  # this will be empty for main
PRODUCTION_DRIVERS=""
if [ -n "$DEVELOPMENT_BRANCH" ]; then
    PRODUCTION_ENV_FILE="$1"
    if [ -e "$PRODUCTION_ENV_FILE" ]; then
        . $PRODUCTION_ENV_FILE
    fi
fi
# Set to default if testing main, or user left it empty
PRODUCTION_DRIVERS=${PRODUCTION_DRIVERS:-"illinois-ceesd/drivers_y1-nozzle@main:w-hagen/isolator@master:illinois-ceesd/drivers_flame1d@main"}
OIFS="$IFS"
IFS=':'; for production_driver_string in $PRODUCTION_DRIVERS;
do
    PRODUCTION_DRIVER_BRANCH=$(printf "$production_driver_string" | cut -d "@" -f 2)
    PRODUCTION_DRIVER_REPO=$(printf "$production_driver_string" | cut -d "@" -f 1)
    PRODUCTION_DRIVER_NAME=$(printf "$PRODUCTION_DRIVER_REPO" | cut -d "/" -f 2)
    PRODUCTION_DRIVER_DIR="production_driver_$PRODUCTION_DRIVER_NAME"
    git clone -b "$PRODUCTION_DRIVER_BRANCH" https\://github.com/"$PRODUCTION_DRIVER_REPO" "$PRODUCTION_DRIVER_DIR"
    cd "$PRODUCTION_DRIVER_DIR"/smoke_test
    ln -s *.py driver.py
    cd ../..
done
IFS="$OIFS"
set +x
