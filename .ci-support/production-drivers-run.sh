#!/bin/bash
set -x
# This script is designed to run the CEESD "production" drivers after
# they have been prepared by an external helper script called
# production-drivers-install.sh. The drivers are each expected to be
# in a directory called "production_driver_*" and are expected to have
# a test driver in "production_driver_*/smoke_test/driver.py".
DRIVERS_HOME=${1:-"."}
cd ${DRIVERS_HOME}
for production_driver in $(ls | grep "production_driver_");
do
    cd "$production_driver"/smoke_test
    python -m mpi4py ./driver.py -i run_params.yaml
    cd ../../
done
cd -
