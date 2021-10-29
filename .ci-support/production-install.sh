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

EMIRGE_INSTALL_PATH=${1:-"."}
PRODUCTION_BRANCH=${PRODUCTION_BRANCH:-"y1-production"}
PRODUCTION_FORK=${PRODUCTION_FORK:-"illinois-ceesd"}

echo "EMIRGE_INSTALL_PATH=${EMIRGE_INSTALL_PATH}"
echo "PRODUCTION_FORK=$PRODUCTION_FORK"
echo "PRODUCTION_BRANCH=$PRODUCTION_BRANCH"

# Install the version of mirgecom we wish to test from source
./install.sh --skip-clone --install-prefix=${EMIRGE_INSTALL_PATH}/ --conda-env=${EMIRGE_INSTALL_PATH}/mirgecom/conda-env.yml --pip-pkgs=${EMIRGE_INSTALL_PATH}/mirgecom/requirements.txt

# Activate the environment (needed?)
source ${EMIRGE_INSTALL_PATH}/config/activate_env.sh
cd ${EMIRGE_INSTALL_PATH}/mirgecom

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
