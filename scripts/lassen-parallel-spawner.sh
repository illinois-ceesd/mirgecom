#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Lassen
# unset CUDA_CACHE_DISABLE
POCL_CACHE_ROOT=${POCL_CACHE_ROOT:-"$(pwd)/pocl-scratch"}
XDG_CACHE_ROOT=${XDG_CACHE_HOME:-"$(pwd)/xdg-scratch"}
POCL_CACHE_DIR=${POCL_CACHE_DIR:-"${POCL_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}

# Reenable CUDA cache
export CUDA_CACHE_DISABLE=0
CUDA_CACHE_PATH_ROOT="$(pwd)/cuda-scratch"
CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"${CUDA_CACHE_PATH_ROOT}/rank$OMPI_COMM_WORLD_RANK"}

export POCL_CACHE_DIR
export XDG_CACHE_HOME
export CUDA_CACHE_PATH

"$@"
