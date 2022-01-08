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

MIRGE_HOME=${1:-"."}
echo "MIRGE_HOME=${MIRGE_HOME}"

if [[ "$GITHUB_HEAD_REF" == "$PRODUCTION_PR_MAIN_BRANCH" ]]; then
  # GITHUB_HEAD_REF is only set for PR builds.
  # https://docs.github.com/en/actions/learn-github-actions/environment-variables
  PRODUCTION_BRANCH="production-merge/${PRODUCTION_PR_MAIN_BRANCH}"
  PRODUCTION_FORK=${PRODUCTION_PR_FORK:-"illinois-ceesd"}
else
  PRODUCTION_BRANCH="${PRODUCTION_BRANCH:-"production"}"
  PRODUCTION_FORK=${PRODUCTION_FORK:-"illinois-ceesd"}
fi


echo "PRODUCTION_FORK=$PRODUCTION_FORK"
echo "PRODUCTION_BRANCH=$PRODUCTION_BRANCH"

cd "${MIRGE_HOME}" || exit 1
git status

# $CI is set by Github Actions
# https://docs.github.com/en/actions/learn-github-actions/environment-variables
if [[ "$CI" == "true" ]]; then
  # This is needed in order for git to create a merge commit
  git config user.email "ci-runner@ci.machine.com"
  git config user.name "CI Runner"
fi

git fetch -f \
        "https://github.com/${PRODUCTION_FORK}/mirgecom.git" \
        "$PRODUCTION_BRANCH":ci-prod-test

git fetch -f \
        "https://github.com/illinois-ceesd/mirgecom.git" \
        production:ci-prod-upstream

if ! git merge ci-prod-test --no-edit; then
  echo "*** The branch of this CI run failed to merge with"
  echo "*** the current production-targeted branch:"
  echo "*** ${PRODUCTION_BRANCH}"
  if [[ "$PRODUCTION_BRANCH" == "production" && -n "$GITHUB_HEAD_REF" ]]; then
    echo "*** You must create a branch off of '$GITHUB_HEAD_REF'"
    echo "*** that is up-to-date with respect to the 'production' branch in the main"
    echo "*** mirgecom repo and make it known to the CI in"
    echo "*** .ci-support/production-testing-env.sh. See the comments there."
  else
    echo "*** You must update this branch to be up-to-date with"
    echo "*** respect to the branch of this test run."
  fi
  exit 1
fi

if ! git merge ci-prod-upstream --no-edit; then
  echo "*** The current mirgecom branch failed to merge with"
  echo "*** mirgecom's current production branch."
  echo "*** You must update this branch to be up-to-date with"
  echo "*** respect to 'production' in the main mirgecom repository."
  exit 1
fi

# Pick up any requirements.txt
pip install -r requirements.txt
cd - || exit 1
