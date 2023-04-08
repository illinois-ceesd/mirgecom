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

function check_return_code()
{
    local return_code=$1
    local example=$2
    if [[ $return_code -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo "*** Example $example succeeded."
        succeeded_examples="$succeeded_examples $example"
    else
        ((numfail=numfail+1))
        echo "*** Example $example failed."
        failed_examples="$failed_examples $example"
    fi
}

if [[ -z "${MIRGE_PARALLEL_SPAWNER:-}" ]];then
    . ${examples_dir}/scripts/mirge-testing-env.sh ${examples_dir}/..
fi

mpi_exec="${MIRGE_MPI_EXEC}"
mpi_launcher="${MIRGE_PARALLEL_SPAWNER}"

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
    if [[ "$example" == *"-mpi.py" ]]; then
        echo "*** Running distributed eager example (2 ranks): $example"
        set -x
        ${mpi_exec} -n 2 $mpi_launcher python -u -O -m mpi4py ${example}
        check_return_code $? ${example}_eager
        set +x
        echo "*** Running distributed lazy example (2 ranks): $example"
        set -x
        ${mpi_exec} -n 2 $mpi_launcher python -u -O -m mpi4py ${example} --lazy
        check_return_code $? ${example}_lazy
        set +x
    else
        echo "*** Running serial eager example: $example"
        set -x
        python -u -O ${example}
        check_return_code $? ${example}_eager
        set +x
        echo "*** Running serial lazy example: $example"
        set -x
        python -u -O ${example} --lazy
        check_return_code $? ${example}_lazy
        set +x
    fi
    date
    printf "***\n"
    rm -rf *vtu *sqlite *pkl *-journal restart_data
done
((numtests=numsuccess+numfail))

cd ${origin}
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
