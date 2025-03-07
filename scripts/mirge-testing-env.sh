#!/bin/bash
#
# Applications may source this file for a set of environment
# variables to make it more convenient to exercise parallel
# mirgecom applications on various platforms.

# set -x

MIRGE_HOME=${1:-"${MIRGE_HOME:-}"}
if [[ -z "${MIRGE_HOME}" ]]; then
    THIS_LOC=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    MIRGE_HOME="${THIS_LOC}/../"
fi

cd ${MIRGE_HOME}
MIRGE_HOME="$(pwd)"
cd -

MIRGE_PARALLEL_SPAWNER=""
MIRGE_MPI_EXEC="mpiexec"
PYOPENCL_TEST=""
PYOPENCL_CTX=""

# Add new hosts here, and <hostname>-parallel-spawner.sh
if [[ $(hostname) == "porter" ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/porter-parallel-spawner.sh"
    PYOPENCL_TEST="port:nv"
    PYOPENCL_CTX="port:nv"

elif [[ $(hostname) == "lassen"* ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/lassen-parallel-spawner.sh"
    PYOPENCL_TEST="port:tesla"
    PYOPENCL_CTX="port:tesla"
    MIRGE_MPI_EXEC="jsrun -g 1 -a 1"

elif [[ $(hostname) == "delta"* ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/delta-parallel-spawner.sh"
    PYOPENCL_CTX="port:nvidia"
    PYOPENCL_TEST="port:nvidia"
    MIRGE_MPI_EXEC="srun"
elif [[ $(hostname) == "tioga"* ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/tioga-parallel-spawner.sh"
    PYOPENCL_CTX="AMD:0"  # ROCR_VISIBLE_DEVICES handles device visibility
    PYOPENCL_TEST="AMD:0"
    MIRGE_MPI_EXEC="flux run --exclusive"
elif [[ $(hostname) == "tuolumne"* ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/tioga-parallel-spawner.sh"
    PYOPENCL_CTX="AMD:0"  # ROCR_VISIBLE_DEVICES handles device visibility
    PYOPENCL_TEST="AMD:0"
    MIRGE_MPI_EXEC="flux run --exclusive"
fi

export MIRGE_HOME
export MIRGE_PARALLEL_SPAWNER
export MIRGE_MPI_EXEC
export PYOPENCL_TEST
export PYOPENCL_CTX

