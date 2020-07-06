#!/bin/bash

rm -f examples/*.vtu
printf "Running examples...\n"
for example in $(ls examples/*.py)
do
    python ${example}
done
printf "done!\n" 
#rm -f examples/*.vtu
