declare -i numfail=0
declare -i numsuccess=0

echo "*** Running tests for mirgecom."

failed_tests=""
succeeded_tests=""

printf "*** Testing documents ..."
return_value=0
python -m pytest --cov=mirgecom --durations=0 --tb=native --junitxml=pytest_doc.xml --doctest-modules -rxsw ../doc/*.rst ../doc/*/*.rst
if [[ $? -eq 0 ]]
then
    ((numsuccess=numsuccess+1))
    echo "*** Doc tests succeeded."
    succeeded_tests="Docs"
else
    ((numfail=numfail+1))
    echo "*** Doc tests failed."
    failed_tests="Docs"
fi

printf "*** Running pytest on testing files  ...\n"
for test_file in test_*.py
do
    printf " ** Test file: ${test_file}.\n"
    test_name=$(echo $test_file | cut -d "_" -f 2 | cut -d "." -f 1)
    printf " ** Test name: ${test_name}.\n"
    python -m pytest --cov=mirgecom --cov-append --durations=0 --tb=native --junitxml=pytest_${test_name}.xml -rxsw $test_file
    if [[ $? -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo " ** All tests in ${test_file} succeeded."
        succeeded_tests="${succeeded_tests} ${test_name}"
    else
        ((numfail=numfail+1))
        echo " ** Some tests in ${test_file} failed."
        failed_tests="${failed_tests} ${test_name}"
    fi
done
((numtests=numsuccess+numfail))
echo "*** Done running tests!"
if [[ $numfail -eq 0 ]]
then
    echo "*** No failures."
else
    echo "*** Failures detected."
    echo "*** Failed tests: ($numfail/$numtests): $failed_tests"
fi
echo "*** Successful tests: ($numsuccess/$numtests): $succeeded_tests"
exit $numfail
