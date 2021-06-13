Arrays and Array Containers in :mod:`mirgecom`
==============================================

:mod:`mirgecom` is quite flexible in terms of what arrays it can use; in fact,
no aspect of the code prescribes the use of a specific type of array.
Instead, we use a :class:`arraycontext.ArrayContext` to determine the
types of arrays on which the computation should take place. At its simplest,
this can be a :class:`arraycontext.PyOpenCLArrayContext`, which uses
:class:`pyopencl.array.Array`\ s that are eagerly evaluated. "Eager evaluation"
means that when you type ``a + b`` with two arrays ``a`` and ``b``, then the
result is computed as soon as the expression is evaluated.

Lazy/deferred evaluation
------------------------
For efficiency, it is sometimes better to defer the evaluation of the
expression, to have an opportunity to apply some optimizations. We call this
*lazy* or *deferred evaluation*; it is a common trick in efficiency-minded
software (e.g. Google's Tensorflow, or expression templates in C++).
This is realized in :mod:`mirgecom` via an :class:`~arraycontext.ArrayContext`
based on :mod:`pytato` (not merged to ``main``) at the time of this writing.

Frozen and thawed arrays
------------------------

All array contexts have a notion of *thawed* and *frozen* arrays that is important
to understand, see :ref:`arraycontext:freeze-thaw` for more details.

Array Containers
----------------

Arrays in :mod:`mirgecom` live in (somewhat) deeply nested data structures
that are :class:`~arraycontext.ArrayContainer`\ s. Array containers typically
support arithmetic and can be passed to most methods in
:class:`~arraycontext.ArrayContext` that take arrays: they simply get applied to
all arrays in the container. The same goes for many of the
discretization-focused functions in :mod:`grudge` and :mod:`meshmode`.

For example, think of the solver state for Euler's equations of gas dynamics:

* At the outermost level, there is :class:`mirgecom.fluid.ConservedVars`,
  which contains...
* :attr:`~mirgecom.fluid.ConservedVars.momentum`, which is a :class:`numpy.ndarray`
  of :class:`~numpy.dtype` "object" (an "object array" for short), which contains...
* :class:`meshmode.dof_array.DOFArray`\ s, i.e. arrays representing a scalar
  solution field on a :class:`meshmode.discretization.Discretization`. These
  contain...
* the actual arrays managed by the array context, typically
  two-dimensional arrays of shape ``(num_elements, num_dofs_per_element)``.
