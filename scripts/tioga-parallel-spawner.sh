#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Tioga.

MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
XDG_CACHE_ROOT=${XDG_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/xdg-cache"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$FLUX_TASK_RANK"}
export XDG_CACHE_HOME
# Check if MIRGE_CACHE_DISABLE is set to 1
if [ "${MIRGE_CACHE_DISABLE:-0}" == "1" ]; then
    export LOOPY_NO_CACHE=1
    export PYOPENCL_NO_CACHE=1
    export POCL_KERNEL_CACHE=0
    export CUDA_CACHE_DISABLE=1
fi
export ROCR_VISIBLE_DEVICES=$FLUX_TASK_LOCAL_ID

"$@"
