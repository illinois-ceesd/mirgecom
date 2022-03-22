#!/bin/bash

# Generic script to run on multiple GPU ranks.
# It works by "hiding" GPUs from CUDA that do not
# correspond to the local rank ID on the nodes, such that
# only a single GPU is visible to each process.
# This is useful on systems such as 'porter' that have multiple
# GPUs on a node but don't have an MPI process launcher that
# handles GPU distribution.
#
# Run it like this:
#   mpirun -n 2 bash run_gpus_generic.sh

set -ex

if [[ -n "$OMPI_COMM_WORLD_NODE_RANK" ]]; then
    # Open MPI
    export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
elif [[ -n "$MPI_LOCALRANKID" ]]; then
    # mpich/mvapich
    export CUDA_VISIBLE_DEVICES=$MPI_LOCALRANKID
fi

# Assumes POCL
export PYOPENCL_TEST="port:nv"

python -m mpi4py pulse-mpi.py --lazy
