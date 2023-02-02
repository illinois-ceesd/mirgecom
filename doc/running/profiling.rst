Profiling and logging
=====================


Profiling memory consumption with :mod:`memray`
-----------------------------------------------

|Mirgecom| automatically tracks overall memory consumption on host and
devices via :mod:`logpyle`, but :mod:`memray` can be used to gain a finer-grained
understanding of how much memory is allocated in which parts of the code.

|Mirgecom| allocates two types of memory during execution:

#. Python host memory for :mod:`numpy` data, Python lists and dicts, etc. This memory
   is always heap-allocated via ``malloc()`` calls.
#. OpenCL device memory for the mesh, etc. At the time of this writing, this memory
   is *by default* allocated via OpenCL's Shared Virtual Memory (SVM) mechanism and
   uses a :mod:`pyopencl`
   `memory pool <https://documen.tician.de/pyopencl/tools.html#memory-pools>`__.
   When running with ``pocl`` on the CPU, the SVM memory is allocated via
   ``malloc()`` calls. When running with ``pocl`` on Nvidia GPUs, the SVM memory is
   allocated using CUDA's managed (unified) memory, via
   `cuMemAllocManaged() <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32>`__.


After installing :mod:`memray` (via e.g. ``$ conda install memray``), memory
consumption can be profiled on Linux or MacOS in the following way::

   # Collect the trace:
   $ python -m memray run --native -m mpi4py examples/wave.py --lazy
   [...]
   # Create a flamegraph HTML
   $ python -m memray flamegraph memray-wave.py.44955.bin
   [...]
   # Open the HTML file
   $ open memray-flamegraph-wave.44955.html

.. note::

   The flamegraph analysis (as well as other analysis tools) needs to be run on the
   same system where the trace was collected, as it needs access to the symbols from
   the machine's binaries. The resulting HTML files can be opened on any system.

.. note::

   Although tracing the allocations has a low performance overhead, the resulting
   trace files and flamegraphs can reach sizes of hundreds of MBytes.
   :mod:`memray` releases after 1.6.0 will include an option (``--aggregate``) to
   reduce the sizes of these files.

.. warning::

   For the reasons outlined in the next subsection, we highly recommend running the
   analysis when running on CPUs, not GPUs.

Common issues
^^^^^^^^^^^^^

#. **Incorrectly low memory consumption when running with pocl-cuda on GPUs**

   When running with pocl-cuda on Nvidia GPUs, the memory consumption will appear to
   be much lower than when running the same analysis on the CPU. The reason for this
   is that we use unified memory on Nvidia GPUs, in which case the SVM memory
   allocations will not be counted against the running application, but against the
   CUDA driver and runtime, thus hiding the memory consumption from tools such as
   ``ps`` or :mod:`memray`. The overall consumption can still be estimated by
   looking at the system memory via e.g. ``free``.


#. **High virtual memory consumption with an installed pocl-cuda**

   When pocl-cuda initializes, it consumes a large amount of virtual memory (~100
   GByte) just due to the initialization. To make the output of memray easier to
   understand (e.g., memray sizes the flamegraph according to virtual memory
   consumed), we recommend disabling or uninstalling pocl-cuda for profiling memory
   consumption, via e.g. ``$ conda uninstall pocl-cuda``.


Profiling kernel execution
--------------------------

You can use :class:`mirgecom.profiling.PyOpenCLProfilingArrayContext` instead of
:class:`~arraycontext.PyOpenCLArrayContext` to profile kernel executions.
In addition to using this array context, you also need to enable profiling in the
underlying :class:`pyopencl.CommandQueue`, like this::

   queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

Note that profiling has a performance impact (~20% at the time of this writing).

.. automodule:: mirgecom.profiling


Time series logging
-------------------

Mirgecom supports logging of simulation and profiling quantities with the help
of :mod:`logpyle`. Logpyle requires
classes to describe how quantities for logging are calculated. For |Mirgecom|, these
classes are described below.

.. automodule:: mirgecom.logging_quantities

An overview of how to use logpyle is given in the :any:`Logpyle documentation <logpyle>`.
