#!/bin/bash
# This script is designed to install the CEESD production test used to
# check that changes to main don't tear up the production capability.
# The script takes one argument, the production environment setup file,
# which is typically `.ci-support/production-env-setup.sh`. To specify
# what production test is installed, the env setup script should set
# the following:
#
# PRODUCTION_DRIVER_FORK = the fork/home of the driver
# PRODUCTION_DRIVER_NAME = the repo name of the driver
# PRODUCTION_DRIVER_BRANCH = the branch for the driver
#
# The default values result in an install of the Y1 nozzle driver that
# works with current mirgecom@y1-production.
#
set -x

DEVELOPMENT_BRANCH="$GITHUB_HEAD_REF"  # this will be empty for main
PRODUCTION_DRIVER_FORK="illinois-ceesd"
PRODUCTION_DRIVER_NAME="drivers_y1-nozzle"
PRODUCTION_DRIVER_BRANCH="update-y1-callbacks"
if [ -n "$DEVELOPMENT_BRANCH" ]; then
    PRODUCTION_ENV_FILE="$1"
    if [ -e "$PRODUCTION_ENV_FILE" ]; then
        . $PRODUCTION_ENV_FILE
    fi
fi
git clone -b $PRODUCTION_DRIVER_BRANCH https://github.com/${PRODUCTION_DRIVER_FORK}/${PRODUCTION_DRIVER_NAME} production-driver
