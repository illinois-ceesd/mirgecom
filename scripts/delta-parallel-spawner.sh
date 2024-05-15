#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Delta.
# unset CUDA_CACHE_DISABLE
export CUDA_CACHE_DISABLE=0

MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache"}
XDG_CACHE_ROOT=${XDG_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/xdg-cache"}
CUDA_CACHE_ROOT=${CUDA_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/cuda-cache"}

XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$SLURM_PROCID"}
CUDA_CACHE_PATH=${CUDA_CACHE_DIR:-"${CUDA_CACHE_DIR}/rank$SLURM_PROCID"}

export XDG_CACHE_HOME
export CUDA_CACHE_PATH

"$@"
