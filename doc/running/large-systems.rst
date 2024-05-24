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
