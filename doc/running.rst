Running MirgeCOM
================

Running with large numbers of ranks and nodes
---------------------------------------------

Several packages used in MirgeCOM cache generated files on the hard
disk in order to speed up multiple executions of the same kernel. This can lead
to slowdowns on startup when executing on many ranks and/or nodes due to concurrent
hard disk accesses. An indicator of concurrency issues are warnings like these::

   .conda/envs/dgfem/lib/python3.8/site-packages/pyopencl/cache.py:101: UserWarning:
   could not obtain cache lock--delete '.cache/pyopencl/pyopencl-compiler-cache-v2-py3.8.3.final.0/lock' if necessary


In order to avoid the slowdowns and warnings, users can direct the packages to create
cache files in directories that are private to each rank by using the ``XDG_CACHE_HOME``
environment variable, such as in the following example::

   $ srun -n 512 bash -c 'XDG_CACHE_HOME=/p/lscratchh/diener3/xdg-scratch$SLURM_PROCID python examples/wave-eager-mpi.py'
