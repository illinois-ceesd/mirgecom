Caching in mirgecom
===================


OpenCL Kernels
--------------

OpenCL kernels are cached on hard disks on multiple levels in a mirgecom
execution. This has the advantage of reducing the compilation time of kernels
when running the same driver multiple times, but can lead to inconsistent
performance results, especially when comparing multiple executions against
each other, as subsequent executions can appear faster than earlier ones due
to caching effects.


The following list discusses mirgecom-related packages that use caching.

Loopy
+++++

Loopy stores generated PyOpenCL kernels in ``$XDG_CACHE_HOME`` by default. You can
export ``LOOPY_NO_CACHE=1`` to disable caching. See
`here <https://github.com/inducer/loopy/blob/e21e8f85d289abbca27ac6abfd71874155fa49da/loopy/__init__.py#L402-L406>`__
for details.

PyOpenCL
++++++++

PyOpenCL stores generated OpenCL kernels in ``$XDG_CACHE_HOME`` by default. You can
export ``PYOPENCL_NO_CACHE=1`` to disable caching. See
`here <https://documen.tician.de/pyopencl/runtime_program.html#envvar-PYOPENCL_NO_CACHE>`__
for details.

POCL
++++

POCL stores compilation results in ``$POCL_CACHE_DIR`` or ``$XDG_CACHE_HOME``
by default. You can export ``POCL_KERNEL_CACHE=0`` to disable caching. See
`here <http://portablecl.org/docs/html/env_variables.html>`__ for details.

CUDA
++++

CUDA stores binary kernels in ``~/.nv/ComputeCache`` by default. You can
export ``CUDA_CACHE_DISABLE=1`` to disable caching. See
`here <https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/>`__
for details.


.. warning::

   The CUDA JIT cache is disabled by default on Lassen.
