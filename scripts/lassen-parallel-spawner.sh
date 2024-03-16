#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Lassen

# Reenable CUDA cache
# unset CUDA_CACHE_DISABLE
export CUDA_CACHE_DISABLE=0

# MIRGE env vars used to setup cache locations
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)"}
POCL_CACHE_ROOT=${POCL_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/pocl-cache"}
XDG_CACHE_ROOT=${XDG_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/xdg-cache"}
CUDA_CACHE_ROOT=${CUDA_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/cuda-cache"}

# These vars are used by pocl, pyopencl, loopy, and cuda for cache location
POCL_CACHE_DIR=${POCL_CACHE_DIR:-"${POCL_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}
CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"${CUDA_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}

export POCL_CACHE_DIR
export XDG_CACHE_HOME
export CUDA_CACHE_PATH

"$@"
