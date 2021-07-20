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


Avoiding overheads due to caching of kernels
--------------------------------------------

Several packages used in MirgeCOM cache generated files on the hard
disk in order to speed up multiple executions of the same kernel. This can lead
to slowdowns on startup when executing on many ranks and/or nodes due to concurrent
hard disk accesses. An indicator of concurrency issues are warnings like these::

   .conda/envs/dgfem/lib/python3.8/site-packages/pyopencl/cache.py:101: UserWarning:
   could not obtain cache lock--delete '.cache/pyopencl/pyopencl-compiler-cache-v2-py3.8.3.final.0/lock' if necessary


In order to avoid the slowdowns and warnings, users can direct the packages to create
cache files in directories that are private to each node by using the ``XDG_CACHE_HOME``
environment variable, such as in the following example::

   $ export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
   $ srun -n 512 python -m mpi4py examples/wave-mpi.py'
