#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Lassen

# Reenable CUDA cache
export CUDA_CACHE_DISABLE=0

# MIRGE env vars used to setup cache locations
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${MIRGE_CACHE_ROOT}/xdg-cache"}
CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"${MIRGE_CACHE_ROOT}/cuda-cache"}

export XDG_CACHE_HOME
export CUDA_CACHE_PATH

"$@"
