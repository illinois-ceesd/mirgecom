#! /bin/bash

#flux: --nodes=2
#flux: --time=30
#flux: --output=runOutput.txt

# Run this script with 'flux batch tioga.flux.sh'

module load rocm/5.7.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/pe/cce/16.0.1/cce/x86_64/lib/

set -x

nnodes=($(flux resource info))
nproc=$((8*nnodes)) # 8 ranks per node, 1 per GPU

echo nnodes=$nnodes nproc=$nproc

export PYOPENCL_CTX="AMD:0"

run_cmd="flux run -N $nnodes -n $nproc --exclusive"

export XDG_CACHE_ROOT=${XDG_CACHE_HOME:-"$(pwd)/xdg-scratch"}
export POCL_CACHE_ROOT=${POCL_CACHE_ROOT:-"$(pwd)/pocl-scratch"}

$run_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_ROOT/$FLUX_TASK_RANK XDG_CACHE_HOME=$XDG_CACHE_ROOT/$FLUX_TASK_RANK ROCR_VISIBLE_DEVICES=$FLUX_TASK_LOCAL_ID python -m mpi4py examples/pulse.py --lazy '
