#!/bin/bash

set -x
set -e
set -o pipefail

test_list="vortex-mpi.py pulse-mpi.py"
for file in ${test_list}
do
    . ./compare_lazy_solution.sh ${file} 1e-13
done
