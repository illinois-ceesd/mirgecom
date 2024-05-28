OpenCL kernel caching
=====================

OpenCL kernels are cached in memory and on hard disk on multiple levels during
a |mirgecom| execution. This has the advantage of reducing the compilation time
of kernels when running the same driver multiple times.

The following sections discuss |mirgecom|-related packages that use caching,
with a focus on configuring the disk-based caching.

.. note::

   The following bash code can be used to remove all disk caches used by |mirgecom| on Linux and MacOS:

   .. Note that the following code is not in a code block so that it
      renders with line breaks.

   ``$ rm -rf $XDG_CACHE_HOME/pytools/pdict* ~/.cache/pytools/pdict*
   ~/Library/Caches/pytools/pdict*  $XDG_CACHE_HOME/pyopencl
   ~/.cache/pyopencl  ~/Library/Caches/pyopencl $POCL_CACHE_DIR
   $XDG_CACHE_HOME/pocl ~/.cache/pocl ~/.nv/ComputeCache $CUDA_CACHE_PATH``

.. note::

   The following bash code can be used to disable all disk caches::

      $ export LOOPY_NO_CACHE=1
      $ export PYOPENCL_NO_CACHE=1
      $ export POCL_KERNEL_CACHE=0
      $ export CUDA_CACHE_DISABLE=1

.. note::

   Disabling disk caching for a specific package only affects
   that particular package. For example, disabling disk caching for :mod:`loopy`
   does not affect the caching behavior of :mod:`pyopencl` or *PoCL*.


Loopy
-----

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
--------

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

.. note::

   PyOpenCL uses ``clCreateProgramWithSource`` on the first compilation and
   caches the OpenCL binary it retrieves. The second time the same source
   is compiled, it uses ``clCreateProgramWithBinary`` to hand the binary
   to the CL runtime (such as PoCL). This can lead to different caching
   behaviors on the first three compilations depending on how the CL runtime
   itself performs caching.


PoCL
----

*PoCL* stores compilation results (LLVM bitcode and shared
libraries) in ``$POCL_CACHE_DIR`` or ``$XDG_CACHE_HOME/pocl`` by default. You
can export ``POCL_KERNEL_CACHE=0`` to disable caching. See `here
<http://portablecl.org/docs/html/using.html#tuning-pocl-behavior-with-env-variables>`__ for details.

.. note::

   When ``$POCL_CACHE_DIR`` and ``$XDG_CACHE_HOME`` are not set, *PoCL*'s cache
   dir defaults to ``~/.cache/pocl`` on Linux and MacOS.


CUDA
----

CUDA stores binary kernels in ``~/.nv/ComputeCache`` (on Linux only, we do
not support CUDA devices on MacOS) by default. You can
export ``CUDA_CACHE_DISABLE=1`` to disable caching, and select a different
cache directory with ``CUDA_CACHE_PATH``. See `here
<https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/>`__
for details.


.. warning::

   The CUDA JIT cache is disabled by default on Lassen, i.e.,
   ``CUDA_CACHE_DISABLE=1`` is set by default. Source: email by
   J. Gyllenhaal on 03/12/2020.
