#!/bin/bash

EMIRGE_HOME=$1
TESTING_RESULTS_FILE=$2
TESTING_LOG_FILE=$3

printf "Testing examples.\n"
./test-examples-quartz.sh ${EMIRGE_HOME} ../examples
examples_script_result=$?
printf "Examples script result: ${examples_result}"
cat example-testing-output >> ${TESTING_LOG_FILE}
examples_testing_result=$(cat example-testing-results)
printf "mirgecom-examples: ${examples_testing_result}\n" >> ${TESTING_RESULTS_FILE}
