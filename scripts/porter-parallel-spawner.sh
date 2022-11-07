#!/bin/bash
#
# Adapted from the original `run_gpus_generic.sh`, designed
# to wrap the spawning of parallel mirgecom drivers on Porter.

if [[ -n "$OMPI_COMM_WORLD_NODE_RANK" ]]; then
    # Open MPI
    export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
elif [[ -n "$MPI_LOCALRANKID" ]]; then
    # mpich/mvapich
    export CUDA_VISIBLE_DEVICES=$MPI_LOCALRANKID
fi

"$@"
