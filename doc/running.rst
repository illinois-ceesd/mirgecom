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
cache files in directories that are private to each rank by using the ``XDG_CACHE_HOME``
environment variable, such as in the following example::

   $ srun -n 512 bash -c 'XDG_CACHE_HOME=/p/lscratchh/diener3/xdg-scratch$SLURM_PROCID python examples/wave-eager-mpi.py'


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


Running on specific systems
---------------------------

General
*******

In general, we recommend running mirgecom with 1 MPI rank (= python process) per node.
Kernel execution will be parallelized automatically through pocl
(either on CPU or GPU, depending on the options you selected and what is available
on the system).


Quartz
******

On the Quartz machine, running mirgecom should be straightforward. An example batch script for the slurm batch system is given below:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -t 00:30:00 # walltime
   #SBATCH -p pbatch

   nnodes=$SLURM_JOB_NUM_NODES
   nproc=$nnodes # 1 rank per node
   export XDG_CACHE_HOME=/tmp/$USER/xdg-scratch # See above on why this is important
   srun -n $nproc python examples/wave-eager-mpi.py

Run this with ``sbatch <script.sh>``.


Lassen
******

On Lassen, we recommend running 1 MPI rank per GPU on each node. Care must be
taken to restrict each rank to a separate GPU to avoid competing for access to
the GPU. The easiest way to do this is by specifying the ``-g 1`` argument to
``lrun``. An example batch script for the LSF batch system is given below:

.. code-block:: bash

   #BSUB -nnodes 4                   # number of nodes
   #BSUB -W 30                       # walltime in minutes
   #BSUB -q pbatch                   # queue to use

   nnodes=$(echo $LSB_MCPU_HOSTS | wc -w)
   nnodes=$((nnodes/2-1))
   nproc=$((4*nnodes)) # 4 ranks per node, 1 per GPU

   export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch" # See above on why this is important
   lrun -g 1 -n $nproc python examples/wave-eager-mpi.py


Run this with ``bsub <script.sh>``.
