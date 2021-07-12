#!/bin/bash

set -x

PRODUCTION_DRIVER_NAME="drivers_y1-nozzle"
PRODUCTION_DRIVER_CHANGE_OWNER="illinois-ceesd"
PRODUCTION_DRIVER_CHANGE_BRANCH="update-y1-callbacks"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git config user.email "stupid@dumb.com"
git config user.name "CI Runner"
if [ -n "${PRODUCTION_DRIVER_CHANGE_BRANCH}" ]; then
    git remote add driver_change https://github.com/${PRODUCTION_DRIVER_CHANGE_OWNER}/${PRODUCTION_DRIVER_NAME}
    git fetch driver_change
    git checkout driver_change/${PRODUCTION_DRIVER_CHANGE_BRANCH}
    git checkout ${CURRENT_BRANCH}
    git merge driver_change/${PRODUCTION_DRIVER_CHANGE_BRANCH} --no-edit
else
    echo "No updates to production driver branch (${CURRENT_BRANCH})"
fi
