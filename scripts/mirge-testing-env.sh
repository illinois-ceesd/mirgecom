#!/bin/bash
#
# Applications may source this file for a set of environment
# variables to make it more convenient to exercise parallel
# mirgecom applications on various platforms.

MIRGE_HOME=${1:-"."}
cd ${MIRGE_HOME}
MIRGE_HOME=$(pwd)
cd -

MIRGE_PARALLEL_SPAWNER=""
MIRGE_MPI_EXEC="mpiexec"
XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
PYOPENCL_TEST=""
PYOPENCL_CTX=""

if [[ $(hostname) == "porter" ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/run-gpus-generic.sh"
    PYOPENCL_TEST="port:nv"
    PYOPENCL_CTX="port:nv"

elif [[ $(hostname) == "lassen"* ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/lassen-parallel-spawner.sh"
    PYOPENCL_TEST="port:tesla"
    PYOPENCL_CTX="port:tesla"
    MIRGE_MPI_EXEC="jsrun -g 1 -a 1"    
fi

export MIRGE_HOME
export MIRGE_PARALLEL_SPAWNER
export MIRGE_MPI_EXEC
export XDG_CACHE_HOME
export PYOPENCL_TEST
export PYOPENCL_CTX

