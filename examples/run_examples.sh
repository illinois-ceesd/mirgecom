#!/bin/bash

# set -e

examples_dir=${1}
if [ -z ${examples_dir} ]
then
    examples_dir="."
fi
declare -i exitcode=0
printf "Running examples in ${examples_dir}...\n"
for example in $(ls ${examples_dir}/*.py)
do
    printf "Running example: ${example}\n"
    if [ "mpi" == *"${example}"* ]
    then
        mpiexec -n 2 python ${example}
    else
        python ${example}
    fi
    printf "Example ${example} "
    if [ $? -eq 0 ]
    then
        printf "succeeded.\n"
    else
        ((exitcode=exitcode+1))
        printf "failed.\n"
    fi
done
printf "Done running examples!\n"
if [ $exitcode -eq 0 ]
then
    printf "No errors.\n"
else
    printf "Errors detected (${exitcode}).\n"
    exit $exitcode
fi
#rm -f examples/*.vtu
