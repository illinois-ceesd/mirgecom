#!/bin/bash

# set -e

origin=$(pwd)
examples_dir=${1-$origin}
declare -i exitcode=0
printf "Running examples in ${examples_dir}...\n"
for example in ${examples_dir}/*.py
do
    if [[ "${example}" == *"mpi"* ]]
    then
        printf "Running parallel example: ${example}\n"        
        mpiexec -n 2 python ${example}
    else
        printf "Running serial example: ${example}\n"        
        python ${example}
    fi
    if [ $? -eq 0 ]
    then
        printf "Example ${example} succeeded.\n"
    else
        ((exitcode=exitcode+1))
        printf "Example ${example} failed.\n"
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
