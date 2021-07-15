#!/bin/bash
#
# This script is designed to patch the CEESD production capability against
# a proposed change to illinois-ceesd/mirgecom@main. The script reads the
# environment config file `.ci-support/production-testing-env.sh`, that
# should set up the expected control variables, which are as follows:
#
# The proposed changes to test may be in a fork, or a local branch. For
# forks, the environment config files should set:

# DEVELOPMENT_FORK = The fork from which the changes are coming (if any)
#
# The production capability to test against may be specified outright, or
# patched by the incoming development. The following vars control the
# production environment:
#
# PRODUCTION_BRANCH = The base production branch to be installed by emirge
# PRODUCTION_CHANGE_FORK = The fork/home of production changes (if any)
# PRODUCTION_CHANGE_BRANCH = Branch from which to pull prod changes (if any)
#
# If the environment file does not exist, the current development is
# tested against `mirgecom@y1-production`. 
set -x

# defaults and automatics
DEVELOPMENT_BRANCH="$GITHUB_HEAD_REF"  # this will be empty for main
DEVELOPMENT_FORK="illinois-ceesd"
PRODUCTION_BRANCH="y1-production"
PRODUCTION_CHANGE_FORK=""
PRODUCTION_CHANGE_BRANCH=""
PRODUCTION_ENV_FILE="$1"
if [ -n "$DEVELOPMENT_BRANCH" ]; then
    if [ -e "$PRODUCTION_ENV_FILE" ]; then
        echo "Reading production configuration for ${DEVELOPMENT_BRANCH}."
        . $PRODUCTION_ENV_FILE
    else
        echo "Using default production configuration for ${DEVELOPMENT_BRANCH}."
        echo "To customize, set up .ci-support/production-testing-env.sh."
    fi
fi
echo "Production environment settings:"
echo "PRODUCTION_ENV_FILE=$PRODUCTION_ENV_FILE"
echo "DEVELOPMENT_FORK=$DEVELOPMENT_FORK"
echo "DEVELOPMENT_BRANCH=$DEVELOPMENT_BRANCH"
echo "PRODUCTION_BRANCH=$PRODUCTION_BRANCH"
echo "PRODUCTION_CHANGE_FORK=$PRODUCTION_CHANGE_FORK"
echo "PRODUCTION_CHANGE_BRANCH=$PRODUCTION_CHANGE_BRANCH"

# Install the production branch with emirge
./install.sh --branch=${PRODUCTION_BRANCH}
cd mirgecom

# This junk is needed to be able to execute git commands properly
git config user.email "ci-runner@ci.machine.com"
git config user.name "CI Runner"

# Make any requested changes to production
if [ -n "${PRODUCTION_CHANGE_BRANCH}" ]; then
    if [ -z "${PRODUCTION_CHANGE_FORK}"]; then
        PRODUCTION_CHANGE_FORK="illinois-ceesd"
    fi
    git remote add production_change https://github.com/${PRODUCTION_CHANGE_FORK}/mirgecom
    git fetch production_change
    git checkout production_change/${PRODUCTION_CHANGE_BRANCH}
    git checkout ${PRODUCTION_BRANCH}
    git merge production_change/${PRODUCTION_CHANGE_BRANCH} --no-edit
else
    echo "No updates to production branch (${PRODUCTION_BRANCH})"
fi

# Merge in the current developement if !main
if [ -n "$DEVELOPMENT_BRANCH" ]; then
    if [ -z "$DEVELOPMENT_FORK"]; then
        DEVELOPMENT_FORK="illinois-ceesd"
    fi
    git remote add changes https://github.com/${DEVELOPMENT_FORK}/mirgecom
    git fetch changes
    git checkout changes/${DEVELOPMENT_BRANCH}
    git checkout ${PRODUCTION_BRANCH}
    git merge changes/${DEVELOPMENT_BRANCH} --no-edit
fi
