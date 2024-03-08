#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Tioga.

POCL_CACHE_ROOT=${POCL_CACHE_ROOT:-"/usr/workspace/wsa/$USER/pocl-scratch"}
XDG_CACHE_ROOT=${XDG_CACHE_HOME:-"/usr/workspace/wsa/$USER/xdg-scratch"}
POCL_CACHE_DIR=${POCL_CACHE_DIR:-"${POCL_CACHE_ROOT}/rank$FLUX_TASK_RANK"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$FLUX_TASK_RANK"}
export POCL_CACHE_DIR
export XDG_CACHE_HOME

export ROCR_VISIBLE_DEVICES=$FLUX_TASK_LOCAL_ID

"$@"
