#!/bin/bash
#BSUB -nnodes 4                   # number of nodes
#BSUB -W 30                       # walltime in minutes
#BSUB -q pbatch                   # queue to use

# Run this script with 'bsub lassen.bsub.sh'

# Put any environment activation here, e.g.:
# source ../../config/activate_env.sh

# OpenCL device selection:
export PYOPENCL_CTX="port:tesla"      # Run on Nvidia GPU with pocl
# export PYOPENCL_CTX="port:pthread"  # Run on CPU with pocl

nnodes=$(echo $LSB_MCPU_HOSTS | wc -w)
nnodes=$((nnodes/2-1))
nproc=$((4*nnodes)) # 4 ranks per node, 1 per GPU

echo nnodes=$nnodes nproc=$nproc

# -a 1: 1 task per resource set
# -g 1: 1 GPU per resource set
# -n $nproc: $nproc resource sets
jsrun_cmd="jsrun -g 1 -a 1 -n $nproc"

# See
# https://mirgecom.readthedocs.io/en/latest/running.html#avoiding-overheads-due-to-caching-of-kernels
# on why this is important
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
export XDG_CACHE_HOME_ROOT="${MIRGE_CACHE_ROOT}/xdg-cache/rank"

# Reenable CUDA cache
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH_ROOT="${MIRGE_CACHE_ROOT}/cuda-cache/rank"

# Print task allocation
$jsrun_cmd js_task_info

echo "----------------------------"

# Run application
$jsrun_cmd bash -c 'CUDA_CACHE_PATH=$CUDA_CACHE_PATH_ROOT$OMPI_COMM_WORLD_RANK XDG_CACHE_HOME=$XDG_CACHE_HOME_ROOT$OMPI_COMM_WORLD_RANK python -m mpi4py ../examples/pulse.py --lazy'
