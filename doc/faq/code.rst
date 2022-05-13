Why on earth does the code...?
==============================

Why does everything go through :mod:`loopy`?
--------------------------------------------

- We don't quite know what execution hardware we'll target, and even if we did, the
  answer to that might change over time. As of this writing, it's likely that at
  least GPUs by Nvidia, Intel, and AMD are in the mix. But large-vector ARM
  CPUs are not out of the question.

  Loopy lets us generate code for all these :class:`~loopy.TargetBase`\ s (e.g.
  CUDA, OpenCL) with ease.

- Likely, the loop structure for efficient code will not look the same on each
  machine. (E.g. because of vectorization, or GPU threads/blocks, or...) So we need a
  tool that lets us write reasonably readable code while being able to
  rewrite/transform that code to suit the loop structure that the machine likes.
  That tool is :mod:`loopy`.

- If we allowed some code to sidestep :mod:`loopy`, then all of those functions would
  have to be performed separately, and likely by hand, for that piece of code, which
  is likely not feasible.


What's the point of array contexts?
-----------------------------------

Arrays are the central large-scale data structure that :mod:`mirgecom` operates on.
:class:`arraycontext.ArrayContext` allows us flexibility in how those are
created (and which ones we use) just by changing which array context we run with.
Some array types you might choose include :class:`pyopencl.array.Array`\ s on the GPU
through OpenCL, or lazily-evaluated :class:`pytato.Array`\ s.

