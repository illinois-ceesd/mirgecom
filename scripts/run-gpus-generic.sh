#!/bin/bash
#
# Generic script to run on multiple GPU ranks.
# It works by "hiding" GPUs from CUDA that do not
# correspond to the local rank ID on the nodes, such that
# only a single GPU is visible to each process.
#
# This is useful on systems such as 'porter' that have multiple
# GPUs on a node but don't have an MPI process launcher that
# handles GPU distribution.
#
# Run it like this:
#   mpiexec -n 2 bash run-gpus-generic.sh python -m mpi4py pulse-mpi.py --lazy
# unset CUDA_CACHE_DISABLE
POCL_CACHE_ROOT=${POCL_CACHE_ROOT:-"/tmp/$USER/pocl-scratch"}
XDG_CACHE_ROOT=${XDG_CACHE_HOME:-"/tmp/$USER/xdg-scratch"}

if [[ -n "$OMPI_COMM_WORLD_NODE_RANK" ]]; then
    # Open MPI
    export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
    RANK_ID="rank${OMPI_COMM_WORLD_RANK}"
elif [[ -n "$MPI_LOCALRANKID" ]]; then
    # mpich/mvapich
    export CUDA_VISIBLE_DEVICES=$MPI_LOCALRANKID
    RANK_ID="rank${MPI_LOCALRANKID}"
fi

POCL_CACHE_DIR=${POCL_CACHE_DIR:-"${POCL_CACHE_ROOT}/${RANK_ID}"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/${RANK_ID}"}

export POCL_CACHE_DIR
export XDG_CACHE_HOME

"$@"
