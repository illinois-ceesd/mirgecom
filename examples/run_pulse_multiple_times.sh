#!/bin/bash

for i in $(seq 1 25);
do
	echo "Round ${i} out of 25"
	PYOPENCL_CTX="0:1" python -m mpi4py nick_pulse.py --lazy --elm=64 --order=1
done
