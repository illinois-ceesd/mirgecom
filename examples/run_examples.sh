#!/bin/bash

rm -f examples/*.vtu
printf "Running examples...\n"
python examples/euler-flow.py
printf "done!\n" 
rm -f examples/*.vtu
