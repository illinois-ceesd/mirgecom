#!/bin/bash

# set -e
set -o nounset

rm -f *.vtu *.pvtu

origin=$(pwd)
examples_dir=${1-$origin}
declare -i numfail=0
declare -i numsuccess=0
date
echo "*** Running examples in $examples_dir ..."
failed_examples=""
succeeded_examples=""

mpi_exec="mpiexec"
mpi_launcher=""
if [[ $(hostname) == "porter" ]]; then
    mpi_launcher="bash scripts/run_gpus_generic.sh"
elif [[ $(hostname) == "lassen"* ]]; then
    export PYOPENCL_CTX="port:tesla"
    export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
    mpi_exec="jsrun -g 1 -a 1"
fi

examples=""
for example in $examples_dir/*.py
do
    example_file=$(basename $example)
    examples="$examples $example_file"
done

cd $examples_dir

for example in $examples
do
    date
    printf "***\n***\n"
    if [[ "$example" == *"-mpi-lazy.py" ]]
    then
        echo "*** Running parallel lazy example (2 ranks): $example"
        set -x
        ${mpi_exec} -n 2 python -u -O -m mpi4py ${example} --lazy
        example_return_code=$?
        set +x
    elif [[ "$example" == *"-mpi.py" ]]; then
        echo "*** Running parallel example (2 ranks): $example"
        set -x
        ${mpi_exec} -n 2 $mpi_launcher python -u -O -m mpi4py ${example}
        example_return_code=$?
        set +x
    elif [[ "$example" == *"-lazy.py" ]]; then
        echo "*** Running serial lazy example: $example"
        set -x
        python -u -O ${example} --lazy
        example_return_code=$?
        set +x
    else
        echo "*** Running serial example: $example"
        set -x
        python -u -O ${example}
        example_return_code=$?
        set +x
    fi
    date
    printf "***\n"
    if [[ $example_return_code -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo "*** Example $example succeeded."
        succeeded_examples="$succeeded_examples $example"
    else
        ((numfail=numfail+1))
        echo "*** Example $example failed."
        failed_examples="$failed_examples $example"
    fi
    # FIXME: This could delete data from other runs
    # rm -rf *vtu *sqlite *pkl *-journal restart_data
done
((numtests=numsuccess+numfail))
echo "*** Done running examples!"
if [[ $numfail -eq 0 ]]
then
    echo "*** No errors."
else
    echo "*** Errors detected."
    echo "*** Failed tests: ($numfail/$numtests): $failed_examples"
fi
echo "*** Successful tests: ($numsuccess/$numtests): $succeeded_examples"
exit $numfail
#rm -f examples/*.vtu
