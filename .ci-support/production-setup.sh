#!/bin/bash

set -x
PRODUCTION_CHANGE_OWNER=""
PRODUCTION_CHANGE_BRANCH=""
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git config user.email "stupid@dumb.com"
git config user.name "CI Runner"
if [ -n "${PRODUCTION_CHANGE_BRANCH}" ]; then
    if [ -z "${PRODUCTION_CHANGE_OWNER}"]; then
        PRODUCTION_CHANGE_OWNER="illinois-ceesd"
    fi
    git remote add production_change https://github.com/${PRODUCTION_CHANGE_OWNER}/mirgecom
    git fetch production_change
    git checkout production_change/${PRODUCTION_CHANGE_BRANCH}
    git checkout ${CURRENT_BRANCH}
    git merge production_change/${PRODUCTION_CHANGE_BRANCH} --no-edit
else
    echo "No updates to production branch (${CURRENT_BRANCH})"
fi
CURRENT_FORK_OWNER=""
if [ -n "${GITHUB_HEAD_REF}" ]; then
    if [ -z "${CURRENT_FORK_OWNER}"]; then
        CURRENT_FORK_OWNER="illinois-ceesd"
    fi
    git remote add changes https://github.com/${CURRENT_FORK_OWNER}/mirgecom
    git fetch changes
    cp -r .ci-support y1-ci-support
    git checkout changes/${GITHUB_HEAD_REF}
    cp ./y1-ci-support/production-setup.sh .ci-setup/
    git add .ci-setup/production-setup.sh
    git commit -am 'Retain y1 setup script.'
    git checkout ${CURRENT_BRANCH}
    git merge changes/${GITHUB_HEAD_REF} --no-edit
fi
