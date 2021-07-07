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

You might be wondering whether a given function or method will return frozen or
thawed data. In general, the documentation should state that, however since
a fair bit of functionality predates these concepts, we are still catching up
in terms of updating the documentation. (Help welcome!)

In the meantime, these rules of thumb should cover most cases:

* If you did not provide an array context to the function
  (explicitly or implicitly via an input array), you will receive frozen data.
* Any data that is cached/long-lived/"at rest" is going to be frozen.
* If the data is the result of a `memoized` function, then it will likely
  be frozen.

To demonstrate the effect of this, first we need some setup:

.. doctest::

   >>> import pyopencl as cl
   >>> from arraycontext import PyOpenCLArrayContext, thaw
   >>> ctx = cl.create_some_context()
   >>> queue = cl.CommandQueue(ctx)
   >>> actx = PyOpenCLArrayContext(queue)
   >>> from meshmode.mesh.generation import generate_regular_rect_mesh
   >>> mesh = generate_regular_rect_mesh(a=(0, 0), b=(1, 1), nelements_per_axis=(10, 10))
   >>> from grudge import DiscretizationCollection
   >>> dcoll = DiscretizationCollection(actx, mesh, order=5)

Most quantities that are maintained by the discretization will be frozen. For example,
if one wanted to grab the nodes of the mesh or normals of a named surface in the mesh:

.. doctest::

   >>> from grudge.dof_desc import DOFDesc
   >>> nodes = thaw(dcoll.nodes(), actx)
   >>> dd = DOFDesc("all_faces")
   >>> nhat = thaw(dcoll.normal(dd), actx)

What can go wrong?  Attempts to operate on frozen data will yield errors similar to
the following:

.. doctest::

   >>> dcoll.nodes() * 5
   Traceback (most recent call last):
    ...
   ValueError: array containers with frozen arrays cannot be operated upon

Fortunately, recovering from this is straightforward:

.. doctest::

   >>> nodes = thaw(dcoll.nodes(), actx)
   >>> result = nodes * 5

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
