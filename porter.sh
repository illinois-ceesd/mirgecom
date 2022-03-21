#!/bin/bash

# Script to run on multiple GPU ranks on porter (with Open MPI).
# Run it like this:
#   mpirun -n 2 bash porter.sh

set -ex

echo $OMPI_COMM_WORLD_LOCAL_RANK

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

export PYOPENCL_CTX=0:1

python -m mpi4py pulse-mpi.py --lazy
