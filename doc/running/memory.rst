Managing memory consumption
===========================

Some memory allocators appear to deal poorly with the allocation
patterns occurring as part of running :mod:`mirgecom`, be it the
tests or actual simulation runs. Unfortunately, the memory allocator
in the GNU C library that is widely used in Linux appears to be
one of those. The root cause for this appears to be fragmentation of the
allocated space. This can lead to out-of-memory situations on machines with
limited memory (e.g. in Github CI or on personal laptops). See `this issue
<https://github.com/illinois-ceesd/mirgecom/issues/212>`__ for some
investigation.

For simulation runs, it is generally best to specify
:class:`pyopencl.tools.MemoryPool` (with the
:class:`~pyopencl.tools.ImmediateAllocator`) as the allocator when creating the
:class:`meshmode.array_context.PyOpenCLArrayContext`.  This avoids the potentially
high cost for the high rate of memory allocation performed by :mod:`mirgecom`,
and it additionally appears to avoid the fragmentation described above.

When that is not possible (e.g. in pytest), the use of an alternate allocator
(such as `jemalloc <https://github.com/jemalloc/jemalloc>`__) has been observed
to help. To use it, install it from conda forge::

    conda install jemalloc

and then start your runs using (Linux)::

    LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so.2 python myscript.py

or (macOS):

    (TBD)
