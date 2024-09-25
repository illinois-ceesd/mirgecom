#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH -t 00:30:00              # walltime (hh:mm:ss)
#SBATCH --partition=gpuA40x4
#SBATCH --ntasks-per-node=4      # change this if running on a partition with other than 4 GPUs per node
#SBATCH --gpus-per-node=4        # change this if running on a partition with other than 4 GPUs per node
#SBATCH --gpu-bind=single:1
#SBATCH --account=bbkf-delta-gpu
#SBATCH --exclusive              # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --gpus-per-task=1

# Run this script with 'sbatch delta.sbatch.sh'

# Delta user guide:
# - https://wiki.ncsa.illinois.edu/display/DSC/Delta+User+Guide
# - https://ncsa-delta-doc.readthedocs-hosted.com/en/latest/

# Put any environment activation here, e.g.:
# source ../../config/activate_env.sh

# OpenCL device selection:
export PYOPENCL_CTX="port:nvidia"     # Run on Nvidia GPU with pocl
# export PYOPENCL_CTX="port:pthread"  # Run on CPU with pocl

nnodes=$SLURM_JOB_NUM_NODES
nproc=$SLURM_NTASKS

echo nnodes=$nnodes nproc=$nproc

srun_cmd="srun -N $nnodes -n $nproc"

# See
# https://mirgecom.readthedocs.io/en/latest/running.html#avoiding-overheads-due-to-caching-of-kernels
# on why this is important
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
export XDG_CACHE_HOME_ROOT="${MIRGE_CACHE_ROOT}/xdg-cache/rank"

# Run application
$srun_cmd bash -c 'XDG_CACHE_HOME=$XDG_CACHE_HOME_ROOT$SLURM_PROCID python -u -O -m mpi4py ./pulse.py'
