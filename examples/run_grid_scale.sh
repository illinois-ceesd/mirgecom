#!/bin/bash

srun -n 1 ./mcrun pulse-scale-mpi.py 8
srun -n 1 ./mcrun pulse-scale-mpi.py 16
srun -n 1 ./mcrun pulse-scale-mpi.py 32
srun -n 1 ./mcrun pulse-scale-mpi.py 64
srun -n 1 ./mcrun pulse-scale-mpi.py 128
srun -n 1 ./mcrun pulse-scale-mpi.py 256
#srun -n 1 ./mcrun pulse-scale-mpi.py 512
#srun -n 1 ./mcrun pulse-scale-mpi.py 1024
