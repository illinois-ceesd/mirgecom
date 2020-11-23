#!/bin/bash
#BSUB -nnodes 4                   # number of nodes
#BSUB -W 30                       # walltime in minutes
#BSUB -q pbatch                   # queue to use

# Run this script with 'bsub lassen.bsub.sh'

# Put any environment activation here, e.g.:
# source ../../config/activate_env.sh

# OpenCL device selection here:
# export PYOPENCL_CTX="port:tesla"    # Run on Nvidia GPU with pocl
# export PYOPENCL_CTX="port:pthread"  # Run on CPU with pocl

nnodes=$(echo $LSB_MCPU_HOSTS | wc -w)
nnodes=$((nnodes/2-1))
nproc=$((4*nnodes)) # 4 ranks per node, 1 per GPU

# See
# https://mirgecom.readthedocs.io/en/latest/running.html#avoiding-overheads-due-to-caching-of-kernels
# on why this is important
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"

lrun -g 1 -n $nproc python -m mpi4py ./vortex-mpi.py
