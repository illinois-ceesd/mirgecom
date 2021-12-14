#!/bin/bash
set -x
#
# This script is designed to patch the CEESD production capability into
# a proposed change to illinois-ceesd/mirgecom@main. The script reads the
# environment config file `.ci-support/production-testing-env.sh`, that
# should set up the expected control variables.
#
# The production capability to test against may be specified outright, or
# patched by the incoming development. The following vars control the
# production environment:
#
# PRODUCTION_BRANCH = The production branch (default=production)
# PRODUCTION_FORK = The production fork (default=illinois-ceesd)
#
MIRGE_HOME=${1:-"."}
PRODUCTION_BRANCH=${PRODUCTION_BRANCH:-"production"}
PRODUCTION_FORK=${PRODUCTION_FORK:-"illinois-ceesd"}

echo "MIRGE_HOME=${MIRGE_HOME}"
echo "PRODUCTION_FORK=$PRODUCTION_FORK"
echo "PRODUCTION_BRANCH=$PRODUCTION_BRANCH"

cd ${MIRGE_HOME}
git status

# This junk is needed to be able to execute git commands properly
git config user.email "ci-runner@ci.machine.com"
git config user.name "CI Runner"

# Making a dedicated production remote adds production forks
git remote add production https://github.com/${PRODUCTION_FORK}/mirgecom
git fetch production

# Merge the production branch for testing the production drivers
git merge production/${PRODUCTION_BRANCH} --no-edit
# Pick up any requirements.txt
pip install -r requirements.txt
cd -
