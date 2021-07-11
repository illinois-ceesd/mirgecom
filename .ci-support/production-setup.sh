set -x
PRODUCTION_CHANGE_OWNER="illinois-ceesd"
PRODUCTION_CHANGE_BRANCH=""
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ ${PRODUCTION_CHANGE_BRANCH} != "" ]; then
    git remote add production_change https://github.com/${PRODUCTION_CHANGE_OWNER}/mirgecom
    git fetch production_change
    git checkout production_change/${PRODUCTION_CHANGE_BRANCH}
    git checkout ${CURRENT_BRANCH}
    git merge production_change/${PRODUCTION_CHANGE_BRANCH} --no-edit
else
    echo "No updates to production branch (${CURRENT_BRANCH})"
fi
