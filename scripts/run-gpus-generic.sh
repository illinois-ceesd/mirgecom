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
#   mpiexec -n 2 bash run-gpus-generic.sh python -m mpi4py pulse.py --lazy

export CUDA_CACHE_DISABLE=0
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${MIRGE_CACHE_ROOT}/xdg-cache"}
CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"${MIRGE_CACHE_ROOT}/cuda-cache"}

if [[ -n "$OMPI_COMM_WORLD_NODE_RANK" ]]; then
    # Open MPI
    export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
elif [[ -n "$MPI_LOCALRANKID" ]]; then
    # mpich/mvapich
    export CUDA_VISIBLE_DEVICES=$MPI_LOCALRANKID
fi

export XDG_CACHE_HOME
export CUDA_CACHE_PATH

"$@"
