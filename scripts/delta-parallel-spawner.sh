#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Delta.
# unset CUDA_CACHE_DISABLE
POCL_CACHE_ROOT=${POCL_CACHE_ROOT:-"/scratch/bbkf/$USER/pocl-scratch"}
XDG_CACHE_ROOT=${XDG_CACHE_HOME:-"/scratch/bbkf/$USER/xdg-scratch"}
POCL_CACHE_DIR=${POCL_CACHE_DIR:-"${POCL_CACHE_ROOT}/rank$SLURM_PROCID"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$SLURM_PROCID"}
export POCL_CACHE_DIR
export XDG_CACHE_HOME

"$@"
