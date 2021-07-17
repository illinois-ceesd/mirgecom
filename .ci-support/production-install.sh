#!/bin/bash
#
# This script is designed to patch the CEESD production capability against
# a proposed change to illinois-ceesd/mirgecom@main. The script reads the
# environment config file `.ci-support/production-testing-env.sh`, that
# should set up the expected control variables, which are as follows:
#
# The proposed changes to test may be in a fork, or a local branch. For
# forks, the environment config files should set:
#
# DEVELOPMENT_FORK = The development fork (default=illinois-ceesd)
#
# The production capability to test against may be specified outright, or
# patched by the incoming development. The following vars control the
# production environment:
#
# PRODUCTION_BRANCH = The production branch (default=y1-production)
# PRODUCTION_FORK = The production fork (default=illinois-ceesd)
#
# If the environment file does not exist, the current development is
# tested against `mirgecom@y1-production`.
#
set -x

if [ -n "$DEVELOPMENT_BRANCH" ]; then
    PRODUCTION_ENV_FILE="$1"
    if [ -e "$PRODUCTION_ENV_FILE" ]; then
        echo "Reading production configuration for ${DEVELOPMENT_BRANCH}."
        . $PRODUCTION_ENV_FILE
    else
        echo "Using default production configuration for ${DEVELOPMENT_BRANCH}."
        echo "To customize, set up .ci-support/production-testing-env.sh."
    fi
fi
DEVELOPMENT_BRANCH=${DEVELOPMENT_BRANCH:-"main"}
DEVELOPMENT_FORK=${DEVELOPMENT_FORK:-"illinois-ceesd"}
PRODUCTION_BRANCH=${PRODUCTION_BRANCH:-"y1-production"}
PRODUCTION_FORK=${PRODUCTION_FORK:-"illinois-ceesd"}

echo "Production environment settings:"
if [ -n "${PRODUCTION_ENV_FILE}" ]; then
    echo "PRODUCTION_ENV_FILE=$PRODUCTION_ENV_FILE"
    cat ${PRODUCTION_ENV_FILE}
fi  
echo "DEVELOPMENT_FORK=$DEVELOPMENT_FORK"
echo "DEVELOPMENT_BRANCH=$DEVELOPMENT_BRANCH"
echo "PRODUCTION_FORK=$PRODUCTION_FORK"
echo "PRODUCTION_BRANCH=$PRODUCTION_BRANCH"

# Install the production branch with emirge
./install.sh --fork=${DEVELOPMENT_FORK} --branch=${DEVELOPMENT_BRANCH}
source config/activate_env.sh
cd mirgecom

# This junk is needed to be able to execute git commands properly
git config user.email "ci-runner@ci.machine.com"
git config user.name "CI Runner"

# Merge in the production environment
git remote add production https://github.com/${PRODUCTION_FORK}/mirgecom
git fetch production
git merge production/${PRODUCTION_BRANCH} --no-edit
pip install -r requirements.txt
