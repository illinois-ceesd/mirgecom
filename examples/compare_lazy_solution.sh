#!/bin/bash

set -x
set -e

SIM="$1"
TOL="$2"
BINDIR="$3"
if [ -z "$BINDIR" ]; then
    BINDIR="../bin"
fi
if [ -z "$SIM" ]; then
    echo "Compare an eager run to a lazy run."
    echo "Usage: compare_lazy_solution.sh <simuation driver file>"
    exit 1
fi
if [ -z "$TOL" ]; then
    TOL=" --tolerance 1e-12 "
else
    TOL=" --tolerance $TOL "
fi
MPIARGS=
if [[ "$SIM" == *"-mpi.py" ]]; then
    MPIRUN="mpirun -n 2"
    MPIARGS=" -m mpi4py "
fi
casename_base=$(echo ${SIM/%.py})
$MPIRUN python ${MPIARGS} ${SIM} --casename ${casename_base}-eager
$MPIRUN python ${MPIARGS} ${SIM} --casename ${casename_base}-lazy --lazy
for vizfile in $(ls ${casename_base}-eager-*.vtu)
do
    lazy_vizfile=$(echo ${vizfile/eager/lazy})
    python ${BINDIR}/mirgecompare.py ${TOL} ${vizfile} ${lazy_vizfile}
done
