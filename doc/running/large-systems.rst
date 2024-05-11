Running with large numbers of ranks and nodes
=============================================

Running MirgeCOM on large systems can be challenging due to the startup overheads of
Python and the MirgeCOM-related packages, as well as due to caching effects of kernels.
As a general rule, make sure to execute MirgeCOM on a parallel file system, not on
NFS-based file systems. On Quartz and Lassen, for example, this would mean running on the
`/p/lscratchh/` and `/p/gpfs1/` file systems, respectively. See the
`Livermore documentation <https://computing.llnl.gov/tutorials/lc_resources/>`__
for more information.


Avoiding the startup overhead of Python
---------------------------------------

On large systems, the file system can become a bottleneck for loading Python
packages, especially when not running on a parallel file system. To avoid this
overhead, it is possible to create a zip file with the Python modules
to speed up the startup process. `Emirge
<https://github.com/illinois-ceesd/emirge/>`__ contains a helper script to
create such a zip file. This can be used by specifying the ``--modules``
parameter to ``install.sh`` when installing emirge, or by running
``makezip.sh`` after installation.


.. _caching-errors:

Avoiding errors and overheads due to caching of kernels
-------------------------------------------------------

Several packages used in MirgeCOM cache generated files on the hard
disk in order to speed up multiple executions of the same kernel. This can lead
to errors and slowdowns when executing on multiple ranks due to concurrent
hard disk accesses. Indicators of file system concurrency issues include::

   .conda/envs/dgfem/lib/python3.8/site-packages/pyopencl/cache.py:101: UserWarning:
   could not obtain cache lock--delete '.cache/pyopencl/pyopencl-compiler-cache-v2-py3.8.3.final.0/lock' if necessary

and::

   pocl-cuda: failed to generate PTX
   CUDA_ERROR_FILE_NOT_FOUND: file not found

In order to avoid these issues, users should direct the packages to create
cache files in directories that are private to each rank by using the ``XDG_CACHE_HOME`` and ``POCL_CACHE_DIR``
environment variables, such as in the following example::

   $ export XDG_CACHE_ROOT="/tmp/$USER/xdg-cache"
   $ export POCL_CACHE_ROOT="/tmp/$USER/pocl-cache"
   $ srun -n 512 bash -c 'POCL_CACHE_DIR=$POCL_CACHE_ROOT/$$ XDG_CACHE_HOME=$XDG_CACHE_ROOT/$$ python -m mpi4py examples/wave.py'


There is also on-disk caching of compiled kernels done by CUDA itself.
As of 01/2023, we have not observed issues specific to this caching.
The CUDA caching behavior can also be controlled via
`environment variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cuda_cache_disable#cuda-environment-variables>`__.
