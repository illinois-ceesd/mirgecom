#!/bin/bash
# set -x
#
# This script is designed to patch the CEESD production capability into
# a proposed change to illinois-ceesd/mirgecom@main.  It takes 1 input
# argument which is the file path to the mirgecom installation directory.
#
# The following vars control the production environment:
#
# PRODUCTION_BRANCH = The production branch (default=production)
# PRODUCTION_FORK = The production fork (default=illinois-ceesd)
#
MIRGE_HOME=${1:-"${MIRGE_HOME}"}
if [[ -z "${MIRGE_HOME}" ]]; then
    . scripts/mirge-testing-env.sh
fi

if [[ -z "${PRODUCTION_BRANCH}" ]]; then
    . ${MIRGE_HOME}/scripts/production-testing-env.sh
fi

echo "MIRGE_HOME=${MIRGE_HOME}"
echo "PRODUCTION_FORK=$PRODUCTION_FORK"
echo "PRODUCTION_BRANCH=$PRODUCTION_BRANCH"

cd ${MIRGE_HOME}

git status
set -x
# This junk is needed to be able to execute git commands properly
GIT_USER="$(git config user.name)"
if [[ -z "${GIT_USER}" ]]; then
    git config user.email "ci-runner@ci.machine.com"
    git config user.name "CI Runner"
fi

# Making a dedicated production remote adds production forks
if git config remote.production.url > /dev/null; then 
    git remote remove production
fi

git remote add production https://github.com/${PRODUCTION_FORK}/mirgecom
git fetch production

# Merge the production branch for testing the production drivers
git merge production/${PRODUCTION_BRANCH} --no-edit

# Pick up any requirements.txt
pip install -r requirements.txt

set +x

export MIRGE_PRODUCTION_INSTALL="${PRODUCTION_FORK}/mirgecom@${PRODUCTION_BRANCH}"

cd -
