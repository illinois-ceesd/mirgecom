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

:mod:`loopy` stores the source of generated PyOpenCL kernels and their
invokers in ``$XDG_CACHE_HOME/pytools/pdict-*`` by default. You can export
``LOOPY_NO_CACHE=1`` to disable caching. See `here
<https://github.com/inducer/loopy/blob/e21e8f85d289abbca27ac6abfd71874155fa49da/loopy/__init__.py#L402-L406>`__
for details.

.. note::

   :mod:`loopy` uses :class:`pytools.persistent_dict.PersistentDict`
   for caching. :class:`~pytools.persistent_dict.PersistentDict` also keeps an in-memory
   cache.

.. note::

   ``$XDG_CACHE_HOME`` by default refers to ``~/.cache`` on Linux and
   ``~/Library/Caches/`` on MacOS.


PyOpenCL
++++++++

:mod:`pyopencl` stores generated OpenCL kernels (their C source code as well
as compiled binary code) in ``$XDG_CACHE_HOME/.pyopencl`` by default. You
can export ``PYOPENCL_NO_CACHE=1`` to disable caching. See `here
<https://documen.tician.de/pyopencl/runtime_program.html#envvar-PYOPENCL_NO_CACHE>`__
for details.

POCL
++++

POCL stores compilation results (LLVM bitcode and shared libraries) in
``$POCL_CACHE_DIR`` or ``$XDG_CACHE_HOME/pocl`` by default. You can export
``POCL_KERNEL_CACHE=0`` to disable caching. See `here
<http://portablecl.org/docs/html/env_variables.html>`__ for details.

.. note::

   For POCL, ``$XDG_CACHE_HOME`` refers to ``~/.cache`` even on MacOS.

CUDA
++++

CUDA stores binary kernels in ``~/.nv/ComputeCache`` by default. You can
export ``CUDA_CACHE_DISABLE=1`` to disable caching. See `here
<https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/>`__
for details.


.. warning::

   The CUDA JIT cache is disabled by default on Lassen.
