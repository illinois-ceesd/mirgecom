OpenCL kernel caching
=====================

OpenCL kernels are cached on hard disk on multiple levels during a |mirgecom|
execution. This has the advantage of reducing the compilation time of kernels
when running the same driver multiple times.

The following sections discuss |mirgecom|-related packages that use caching.


Loopy
+++++

:mod:`loopy` stores the source of generated PyOpenCL kernels and their
invokers in ``$XDG_CACHE_HOME/pytools/pdict-*-loopy`` by default. You can export
``LOOPY_NO_CACHE=1`` to disable caching. See `here
<https://github.com/inducer/loopy/blob/e21e8f85d289abbca27ac6abfd71874155fa49da/loopy/__init__.py#L402-L406>`__
for details.

.. note::

   :mod:`loopy` uses :class:`pytools.persistent_dict.PersistentDict`
   for caching. :class:`~pytools.persistent_dict.PersistentDict` also keeps an
   in-memory cache.

.. note::

   When ``$XDG_CACHE_HOME`` is not set, the cache dir defaults to
   ``~/.cache`` on Linux and ``~/Library/Caches/`` on MacOS.


PyOpenCL
++++++++

:mod:`pyopencl` caches in ``$XDG_CACHE_HOME/pyopencl`` (kernel source
code and binaries returned by the OpenCL runtime) and
``$XDG_CACHE_HOME/pytools/pdict-*-pyopencl`` (invokers, generated source code)
by default. You can export ``PYOPENCL_NO_CACHE=1`` to disable caching. See `here
<https://documen.tician.de/pyopencl/runtime_program.html#envvar-PYOPENCL_NO_CACHE>`__
for details.

.. note::

   PyOpenCL does not cache kernel binaries in memory by default. To keep the
   compiled version of a kernel in memory, simply retain the
   :class:`pyopencl.Program` or :class:`pyopencl.Kernel` objects. Loopy's
   :class:`loopy.LoopKernel` already holds handles to compiled
   :class:`pyopencl.Kernel` objects.


PoCL
++++

PoCL stores compilation results (LLVM bitcode and shared libraries) in
``$POCL_CACHE_DIR`` or ``$XDG_CACHE_HOME/pocl`` by default. You can export
``POCL_KERNEL_CACHE=0`` to disable caching. See `here
<http://portablecl.org/docs/html/using.html#tuning-pocl-behavior-with-env-variables>`__ for details.

.. note::

   When ``$POCL_CACHE_DIR`` and ``$XDG_CACHE_HOME`` are not set, PoCL's cache
   dir defaults to ``~/.cache/pocl`` on Linux and MacOS.


CUDA
++++

CUDA stores binary kernels in ``~/.nv/ComputeCache`` by default. You can
export ``CUDA_CACHE_DISABLE=1`` to disable caching. See `here
<https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/>`__
for details.


.. warning::

   The CUDA JIT cache is disabled by default on Lassen.
