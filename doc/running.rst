Running MirgeCOM
================

Running with large numbers of ranks and nodes
---------------------------------------------

Running MirgeCOM on large systems can be challenging due to the startup overheads of
Python and the MirgeCOM-related packages, as well as due to caching effects of kernels.
As a general rule, make sure to execute MirgeCOM on a parallel file system, not on
NFS-based file systems. On Quartz and Lassen, for example, this would mean running on the
`/p/lscratchh/` and `/p/gpfs1/` file systems, respectively. See the
`Livermore documentation <https://computing.llnl.gov/tutorials/lc_resources/>`__
for more information.


Avoiding the startup overhead of Python
***************************************

On large systems, the file system can become a bottleneck for loading Python
packages, especially when not running on a parallel file system. To avoid this
overhead, it is possible to create a zip file with the Python modules
to speed up the startup process. `Emirge
<https://github.com/illinois-ceesd/emirge/>`__ contains a helper script to
create such a zip file. This can be used by specifying the ``--modules``
parameter to ``install.sh`` when installing emirge, or by running
``makezip.sh`` after installation.

Avoiding overheads due to caching of kernels
********************************************


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
   $ srun -n 512 python -m mpi4py examples/wave-eager-mpi.py'


OpenCL device selection
-----------------------

The :mod:`pyopencl` package supports selecting the device on which to run OpenCL code.
There are multiple ways in which this selection can happen:

- **Interactively.** When running mirgecom on the command line (i.e., not in a
  batch script), it will present a (potentially multiple-choice) dialog with
  the available OpenCL drivers (such as pocl) and the device (such as CPU or
  GPU). After a device has been selected, pyopencl will print the chosen
  device in the form of a value for the ``PYOPENCL_TEST`` environment
  variable that can be used for subsequent executions.

- **Through the PYOPENCL_TEST or PYOPENCL_CTX environment variables.** 
  Pyopencl will use the value of the ``PYOPENCL_TEST`` or ``PYOPENCL_CTX``
  environment variables when set for device selection. The value of either variable
  can either be list indices (such as ``0:1`` for first driver, second device),
  or named abbreviations (such as ``port:tesla`` for pocl (=Portable OpenCL)
  and the first Nvidia Tesla GPU).

- **Default device.** When not using one of the options above, pyopencl will
  run on the first available device (which might not be the device you want)
  by default.

.. note::

   The device selection functionality described here is provided by the
   :func:`pyopencl.create_some_context`,
   :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl`, and
   :func:`meshmode.array_context.pytest_generate_tests_for_pyopencl_array_context`
   functions used in the default simulation drivers and tests. It is also
   possible to write your own device selection code with
   :func:`pyopencl.get_platforms`, :meth:`pyopencl.Platform.get_devices`, and
   :class:`pyopencl.Context`.

.. note::

   Each MPI rank (=Python process) can only use one device during the
   execution, even if there are multiple devices available. Although it is
   possible to run multiple MPI ranks on the same device, we do not recommend
   doing this, as it will lead to contention.

.. note::

   You can also use the ``clinfo`` command (automatically installed when installing
   emirge) to list all available OpenCL devices and their parameters.

Profiling kernel execution
--------------------------

You can use :class:`mirgecom.profiling.PyOpenCLProfilingArrayContext` instead of
:class:`~meshmode.array_context.PyOpenCLArrayContext` to profile kernel executions.
In addition to using this array context, you also need to enable profiling in the underlying
:class:`pyopencl.CommandQueue`, like this::

   queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

Note that profiling has a performance impact (~20% at the time of this writing).

.. automodule:: mirgecom.profiling

Scripting launches
------------------

In certain cases it may be useful to launch MIRGE-Com (or other MPI tasks) from inside a
Python script. The `squirm <https://github.com/majosm/squirm>`__ package provides
functionality to help with this.

Running on specific systems
---------------------------

This section discusses how to run mirgecom on various clusters.
There are also several example run scripts in mirgecom's ``examples/``
`folder <https://github.com/illinois-ceesd/mirgecom/tree/master/examples>`__.

General
*******

In general, we recommend running mirgecom with 1 MPI rank (= python process) per cluster node.
For GPU execution, we recommend running with 1 MPI rank per GPU.
Kernel execution will be parallelized automatically through pocl
(either on CPU or GPU, depending on the options you selected and what is available
on the system).


Quartz
******

On the Quartz machine, running mirgecom should be straightforward.
An example batch script for the slurm batch system is given below:

.. literalinclude:: ../examples/quartz.sbatch.sh
    :language: bash

Run this with ``sbatch <script.sh>``.

More information about Quartz can be found here:

- `Machine overview <https://hpc.llnl.gov/hardware/platforms/Quartz>`__
- `Detailed information <https://computing.llnl.gov/tutorials/lc_resources/>`__
- `Programming environment <https://computing.llnl.gov/tutorials/linux_clusters/>`__

Lassen
******

On Lassen, we recommend running 1 MPI rank per GPU on each node. Care must be
taken to restrict each rank to a separate GPU to avoid competing for access to
the GPU. The easiest way to do this is by specifying the ``-g 1`` argument to
``lrun``. An example batch script for the LSF batch system is given below:

.. literalinclude:: ../examples/lassen.bsub.sh
    :language: bash


Run this with ``bsub <script.sh>``.

More information about Lassen can be found here:

- `Machine overview <https://hpc.llnl.gov/hardware/platforms/Lassen>`__
- `Detailed information and programming environment <https://hpc.llnl.gov/training/tutorials/using-lcs-sierra-system>`__
