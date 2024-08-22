#!/bin/bash
#SBATCH -N 2                        # number of nodes
#SBATCH -t 00:30:00                 # walltime (hh:mm:ss)
#SBATCH -p pbatch                   # queue to use

# Run this script with 'sbatch quartz.sbatch.sh'

# Put any environment activation here, e.g.:
# source ../../config/activate_env.sh

# OpenCL device selection:
# export PYOPENCL_CTX="port:pthread"  # Run on CPU with pocl

nnodes=$SLURM_JOB_NUM_NODES
nproc=$nnodes # 1 rank per node

echo nnodes=$nnodes nproc=$nproc

# See
# https://mirgecom.readthedocs.io/en/latest/running.html#avoiding-overheads-due-to-caching-of-kernels
# on why this is important
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
export XDG_CACHE_HOME="${MIRGE_CACHE_ROOT}/xdg-cache"

# Run application
# -O: switch on optimizations
srun -n $nproc python -O -m mpi4py ./vortex.py
