#!/bin/bash

set -ex
set -o nounset

rm -f *.vtu *.pvtu

origin=$(pwd)
examples_dir=${1-$origin}

date
echo "*** Running examples in $examples_dir ..."

if [[ -z "${MIRGE_PARALLEL_SPAWNER:-}" ]];then
    . ${examples_dir}/scripts/mirge-testing-env.sh ${examples_dir}/..
fi

mpi_exec="${MIRGE_MPI_EXEC}"
mpi_launcher="${MIRGE_PARALLEL_SPAWNER}"

cd $examples_dir

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

for example in *.py
do
    if [[ $example == "wave-nompi.py" ]]; then
        echo "Skipping $example"
        continue
    fi

    TOL_LAZY=1e-04
    TOL_NUMPY=1e-05

    if [[ $example == "thermally-coupled.py" ]]; then
        echo "Setting tolerance=1 for $example"
        TOL_LAZY=1
        TOL_NUMPY=1
    fi


    date
    printf "***\n***\n"

    ${mpi_exec} -n 2 $mpi_launcher python -m mpi4py $example --casename ${example}-eager
    ${mpi_exec} -n 2 $mpi_launcher python -m mpi4py $example --casename ${example}-lazy --lazy
    ${mpi_exec} -n 2 $mpi_launcher python -m mpi4py $example --casename ${example}-numpy --numpy

    # Note: not all examples produce vtu files, for these the accuracy test won't be
    # run.
    for vizfile in $(ls ${example}-eager-*.vtu); do
        lazy_vizfile=$(echo ${vizfile/eager/lazy})
        python ${examples_dir}/../bin/mirgecompare.py --tolerance $TOL_LAZY ${vizfile} ${lazy_vizfile}
        numpy_vizfile=$(echo ${vizfile/eager/numpy})
        python ${examples_dir}/../bin/mirgecompare.py --tolerance $TOL_NUMPY ${vizfile} ${numpy_vizfile}
    done

    date
    printf "***\n"
    # FIXME: This could delete data from other runs
    # rm -rf *vtu *sqlite *pkl *-journal restart_data
done

cd ${origin}
echo "*** Done running examples!"
