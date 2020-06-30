#!/bin/sh

rm -f examples/*.vtu
printf "Running examples...\n"
python examples/eulerflow.py
printf "done!\n" 
rm -f examples/*.vtu
