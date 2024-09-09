#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Tioga.

MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
XDG_CACHE_ROOT=${XDG_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/xdg-cache"}

XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$FLUX_TASK_RANK"}

export XDG_CACHE_HOME
echo "Before env customization (local,global) rank ($FLUX_TASK_LOCAL_ID,$FLUX_TASK_RANK)"
env | grep ROCR_VISIBLE_DEVICES
# export ROCR_VISIBLE_DEVICES=$FLUX_TASK_LOCAL_ID
echo "After env customization (local,global) rank ($FLUX_TASK_LOCAL_ID,$FLUX_TASK_RANK)"
env | grep ROCR_VISIBLE_DEVICES

"$@"
