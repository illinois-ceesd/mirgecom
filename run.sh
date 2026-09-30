#! /bin/bash

# Uncomment on beers for CPU runs
# export PYOPENCL_TEST=Port:

# Uncomment on beers for GPU runs
# export PYOPENCL_TEST=NVIDIA:

# Uncomment on Tuolumne
# module load rocm/7.2.0
# module load cray-mpich-abi
# export PYOPENCL_TEST=AMD:gfx

export PYTHONHASHSEED=0
export LOOPY_NO_CACHE=1

# smaller problem for timestep time
# (cd examples && python -O -m mpi4py gas-in-box.py \
#   --lazy --dimension=3 --tpe --nsteps=20 --weak-scale=1 \
#   --navierstokes --boundaries --polynomial-order=1 \
#   ) 2>&1 | tee out.txt

# larger problem for timestep time
# (cd examples && python -O -m mpi4py gas-in-box.py \
#   --lazy --dimension=3 --tpe --nsteps=20 --weak-scale=16 \
#   --navierstokes --artificial-viscosity=3 --boundaries --polynomial-order=3 \
#   --mixture --flame --iters=2 --limiter \
#   ) 2>&1 | tee out.txt

# problem for array contraction memory use reduction
# (cd examples && python -O -m mpi4py gas-in-box.py \
#   --lazy --dimension=3 --tpe --nsteps=20 --weak-scale=4 \
#   --navierstokes --artificial-viscosity=3 --boundaries --polynomial-order=3 \
#   --mixture --flame --iters=2 --limiter \
#   ) 2>&1 | tee out.txt
