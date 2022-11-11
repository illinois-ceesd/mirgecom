#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Lassen
# unset CUDA_CACHE_DISABLE
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"/tmp/$USER/xdg-scratch"}
POCL_CACHE_ROOT=${POCL_CACHE_ROOT:-"/tmp/$USER/pocl-scratch"}

export POCL_CACHE_DIR="${POCL_CACHE_ROOT}/$$"

"$@"
