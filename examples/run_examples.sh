#!/bin/bash

# set -e
set -o nounset

origin=$(pwd)
examples_dir=${1-$origin}
declare -i exitcode=0
echo "Running examples in $examples_dir ..."
for example in $examples_dir/*.py
do
    if [[ "$example" == *"-mpi.py" ]]
    then
        echo "Running parallel example: $example"        
        srun -n 2 python ${example}
    else
        echo "Running serial example: $example"        
        python ${example}
    fi
    if [[ $? -eq 0 ]]
    then
        echo "Example $example succeeded."
    else
        ((exitcode=exitcode+1))
        echo "Example $example failed."
    fi
done
echo "Done running examples!"
if [[ $exitcode -eq 0 ]]
then
    echo "No errors."
else
    echo "Errors detected ($exitcode)."
    exit $exitcode
fi
#rm -f examples/*.vtu
