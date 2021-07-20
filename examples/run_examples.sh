#!/bin/bash

# set -e
set -o nounset

rm -f *.vtu *.pvtu

origin=$(pwd)
examples_dir=${1-$origin}
declare -i exitcode=0
echo "*** Running examples in $examples_dir ..."
failed_examples=""
for example in $examples_dir/*.py
do
    if [[ "$example" == *"-mpi-lazy.py" ]]
    then
        echo "*** Running parallel lazy example (1 rank): $example"
        mpiexec -n 1 python -m mpi4py ${example} --lazy
        rm -rf *vtu *sqlite *pkl *-journal restart_data
    elif [[ "$example" == *"-mpi.py" ]]; then
        echo "*** Running parallel example (2 ranks): $example"
        mpiexec -n 2 python -m mpi4py ${example}
        rm -rf *vtu *sqlite *pkl *-journal restart_data
    elif [[ "$example" == *"-lazy.py" ]]; then
        echo "*** Running serial lazy example: $example"
        python ${example} --lazy
        rm -rf *vtu *sqlite *pkl *-journal restart_data
    else
        echo "*** Running serial example: $example"
        python ${example}
        rm -rf *vtu *sqlite *pkl *-journal restart_data
    fi
    if [[ $? -eq 0 ]]
    then
        echo "*** Example $example succeeded."
    else
        ((exitcode=exitcode+1))
        echo "*** Example $example failed."
        failed_examples="$failed_examples $example"
    fi
done
echo "*** Done running examples!"
if [[ $exitcode -eq 0 ]]
then
    echo "*** No errors."
else
    echo "*** Errors detected ($exitcode):($failed_examples )"
    exit $exitcode
fi
#rm -f examples/*.vtu
