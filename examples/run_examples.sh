#!/bin/bash

# set -e
set -o nounset

rm -f *.vtu *.pvtu

origin=$(pwd)
examples_dir=${1-$origin}
declare -i numfail=0
declare -i numsuccess=0
echo "*** Running examples in $examples_dir ..."
failed_examples=""
succeeded_examples=""
for example in $examples_dir/*.py
do
    if [[ "$example" == *"-mpi-lazy.py" ]]
    then
        echo "*** Running parallel lazy example (1 rank): $example"
        mpiexec -n 1 python -m mpi4py ${example} --lazy
    elif [[ "$example" == *"-mpi.py" ]]; then
        echo "*** Running parallel example (2 ranks): $example"
        mpiexec -n 2 python -m mpi4py ${example}
    elif [[ "$example" == *"-lazy.py" ]]; then
        echo "*** Running serial lazy example: $example"
        python ${example} --lazy
    else
        echo "*** Running serial example: $example"
        python ${example}
    fi
    if [[ $? -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo "*** Example $example succeeded."
        succeeded_examples="$succeeded_examples $example"
    else
        ((numfail=numfail+1))
        echo "*** Example $example failed."
        failed_examples="$failed_examples $example"
    fi
    rm -rf *vtu *sqlite *pkl *-journal restart_data
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
