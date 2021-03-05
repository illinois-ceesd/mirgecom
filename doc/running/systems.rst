Running on specific systems
===========================

This section discusses how to run mirgecom on various clusters.
There are also several example run scripts in mirgecom's ``examples/``
`folder <https://github.com/illinois-ceesd/mirgecom/tree/main/examples>`__.

General
-------

In general, we recommend running mirgecom with 1 MPI rank (= python process) per cluster node.
For GPU execution, we recommend running with 1 MPI rank per GPU.
Kernel execution will be parallelized automatically through pocl
(either on CPU or GPU, depending on the options you selected and what is available
on the system).


Quartz
------

On the Quartz machine, running mirgecom should be straightforward.
An example batch script for the slurm batch system is given below:

.. literalinclude:: ../../examples/quartz.sbatch.sh
    :language: bash

Run this with ``sbatch <script.sh>``.

More information about Quartz can be found here:

- `Machine overview <https://hpc.llnl.gov/hardware/platforms/Quartz>`__
- `Detailed information <https://computing.llnl.gov/tutorials/lc_resources/>`__
- `Programming environment <https://computing.llnl.gov/tutorials/linux_clusters/>`__

Lassen
------

On Lassen, we recommend running 1 MPI rank per GPU on each node. Care must be
taken to restrict each rank to a separate GPU to avoid competing for access to
the GPU. The easiest way to do this is by specifying the ``-g 1`` argument to
``lrun``. An example batch script for the LSF batch system is given below:

.. literalinclude:: ../../examples/lassen.bsub.sh
    :language: bash


Run this with ``bsub <script.sh>``.

More information about Lassen can be found here:

- `Machine overview <https://hpc.llnl.gov/hardware/platforms/Lassen>`__
- `Detailed information and programming environment <https://hpc.llnl.gov/training/tutorials/using-lcs-sierra-system>`__
