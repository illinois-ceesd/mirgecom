#!/bin/bash
set -x
# This script is designed to run the CEESD "production" drivers after
# they have been prepared by an external helper script called
# production-drivers-install.sh. The drivers are each expected to be
# in a directory called "production_driver_*" and are expected to have
# a test driver in "production_driver_*/smoke_test/driver.py".
DRIVERS_HOME=${1:-"."}
cd ${DRIVERS_HOME}
DRIVERS_HOME=$(pwd)

printf "DRIVERS_HOME: ${DRIVERS_HOME}\n"

rm -rf parallel_spawner.sh mirge_testing_resource.sh

if [[ $(hostname) == "porter" ]]; then
    cat <<EOF > parallel_spawner.sh
if [[ -n "\$OMPI_COMM_WORLD_NODE_RANK" ]]; then
    # Open MPI
    export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK
elif [[ -n "\$MPI_LOCALRANKID" ]]; then
     # mpich/mvapich
     export CUDA_VISIBLE_DEVICES=\$MPI_LOCALRANKID
fi

"\$@"
EOF
    cat <<EOF > mirge_testing_resource.sh
export PYOPENCL_TEST="port:nv"
export PYOPENCL_CTX="port:nv"
export MIRGE_MPI_EXEC="mpiexec"
export MIRGE_PARALLEL_SPAWNER="bash ${DRIVERS_HOME}/parallel_spawner.sh"
EOF

elif [[ $(hostname) == "lassen"* ]]; then

    cat <<EOF > parallel_spawner.sh
export POCL_CACHE_DIR=\$POCL_CACHE_DIR_ROOT/\$\$
"\$@"
EOF
    cat <<EOF > mirge_testing_resource.sh
export PYOPENCL_CTX="port:tesla"
export PYOPENCL_TEST="port:tesla"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
export MIRGE_MPI_EXEC="jsrun -g 1 -a 1"
export MIRGE_PARALLEL_SPAWNER="bash ${DRIVERS_HOME}/parallel_spawner.sh"
EOF
else
    cat <<EOF > mirge_testing_resource.sh
export MIRGE_MPI_EXEC="mpiexec"
EOF
fi

if [[ -f "parallel_spawner.sh" ]]; then
    chmod +x parallel_spawner.sh
fi

testing_resource="${DRIVERS_HOME}/mirge_testing_resource.sh"
printf "Testing Resources:\n"
if [[ -f "mirge_testing_resource.sh" ]]; then
    cat $testing_resource
else
    printf "None\n"
fi

for production_driver in $(ls | grep "production_driver_");
do
    driver_path="${DRIVERS_HOME}/${production_driver}"
    . ${production_driver}/scripts/smoke_test.sh "${testing_resource}" "${driver_path}"
    test_result=$?

    if [[ $test_result -eq 0 ]]; then
        printf "${production_driver} tests passed."
    else
        printf "${production_driver} tests failed."
    fi

done

cd -
set +x
