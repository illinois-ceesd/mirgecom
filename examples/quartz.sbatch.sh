#!/bin/bash
#SBATCH -N 2                        # number of nodes
#SBATCH -t 00:30:00                 # walltime
#SBATCH -p pbatch                   # queue to use

# Run this script with 'sbatch quartz.sbatch.sh'

# Put any environment activation here, e.g.:
# source ../../config/activate_env.sh

# Put any device selection here, e.g.:
# export PYOPENCL_CTX=":"

nnodes=$SLURM_JOB_NUM_NODES
nproc=$nnodes # 1 rank per node

# See
# https://mirgecom.readthedocs.io/en/latest/running.html#avoiding-overheads-due-to-caching-of-kernels
# on why this is important
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"

srun -n $nproc python -m mpi4py ./vortex-mpi.py
