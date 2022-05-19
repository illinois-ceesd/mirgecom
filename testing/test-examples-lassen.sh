#!/bin/bash

EMIRGE_HOME=$1
origin=$(pwd)
EXAMPLES_HOME=$2
# examples_dir=${1-$origin}
BATCH_SCRIPT_NAME="examples-lassen-batch.sh"
examples_dir="${EXAMPLES_HOME}"

cat <<EOF > ${BATCH_SCRIPT_NAME}
#!/bin/bash

#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 120
#BSUB -q pdebug

printf "Running with EMIRGE_HOME=${EMIRGE_HOME}\n"

source "${EMIRGE_HOME}/config/activate_env.sh"
export PYOPENCL_CTX="port:tesla"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
rm -rf \$XDG_CACHE_HOME
rm -f timing-run-done
which python
conda env list
env
env | grep LSB_MCPU_HOSTS

serial_spawner_cmd="jsrun -g 1 -a 1 -n 1"
parallel_spawner_cmd="jsrun -g 1 -a 1 -n 2"

set -o nounset

rm -f *.vtu *.pvtu

declare -i numfail=0
declare -i numsuccess=0
echo "*** Running examples in $examples_dir ..."
failed_examples=""
succeeded_examples=""

for example in $examples_dir/*.py
do
    if [[ "\$example" == *"-mpi-lazy.py" ]]
    then
        echo "*** Running parallel lazy example (1 rank): \$example"
        \$serial_spawner_cmd python -O -m mpi4py \${example} --lazy
    elif [[ "\$example" == *"-mpi.py" ]]; then
        echo "*** Running parallel example (2 ranks): \$example"
        \$parallel_spawner_cmd python -O -m mpi4py \${example}
    elif [[ "\$example" == *"-lazy.py" ]]; then
        echo "*** Running serial lazy example: \$example"
        python -O \${example} --lazy
    else
        echo "*** Running serial example: \$example"
        python -O \${example}
    fi
    if [[ \$? -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo "*** Example \$example succeeded."
        succeeded_examples="\$succeeded_examples \$example"
    else
        ((numfail=numfail+1))
        echo "*** Example \$example failed."
        failed_examples="\$failed_examples \$example"
    fi
    rm -rf *vtu *sqlite *pkl *-journal restart_data
done
((numtests=numsuccess+numfail))
echo "*** Done running examples!"
if [[ \$numfail -eq 0 ]]
then
    echo "*** No errors."
else
    echo "*** Errors detected."
    echo "*** Failed tests: (\$numfail/\$numtests): \$failed_examples"
fi
echo "*** Successful tests: (\$numsuccess/\$numtests): \$succeeded_examples"

printf "\$numfail\n" > example-testing-results
touch example-testing-done
exit \$numfail

EOF

rm -f example-testing-done
chmod +x ${BATCH_SCRIPT_NAME}
# ---- Submit the batch script and wait for the job to finish
bsub ${BATCH_SCRIPT_NAME}
# ---- Wait 25 minutes right off the bat
sleep 1500
iwait=0
while [ ! -f ./example-testing-done ]; do 
    iwait=$((iwait+1))
    if [ "$iwait" -gt 89 ]; then # give up after almost 2 hours
        printf "Timed out waiting on batch job.\n"
        exit 1 # skip the rest of the script
    fi
    sleep 60
done
sleep 30  # give the batch system time to spew its junk into the log
cat *.out > example-testing-output
date >> example-testing-output
rm *.out
date
