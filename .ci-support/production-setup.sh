#!/bin/bash

set -x
PRODUCTION_CHANGE_OWNER="illinois-ceesd"
PRODUCTION_CHANGE_BRANCH=""
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git config user.email "stupid@dumb.com"
git config user.name "CI Runner"
if [ -n "${PRODUCTION_CHANGE_BRANCH}" ]; then
    git remote add production_change https://github.com/${PRODUCTION_CHANGE_OWNER}/mirgecom
    git fetch production_change
    git checkout production_change/${PRODUCTION_CHANGE_BRANCH}
    git checkout ${CURRENT_BRANCH}
    git merge production_change/${PRODUCTION_CHANGE_BRANCH} --no-edit
else
    echo "No updates to production branch (${CURRENT_BRANCH})"
fi
CURRENT_FORK_OWNER="illinois-ceesd"
if [ -n "${GITHUB_HEAD_REF}" ]; then
    git remote add changes https://github.com/${CURRENT_FORK_OWNER}/mirgecom
    git fetch changes
    git checkout changes/${GITHUB_HEAD_REF}
    git checkout ${CURRENT_BRANCH}
    git merge changes/${GITHUB_HEAD_REF} --no-edit
fi
