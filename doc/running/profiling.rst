Profiling and logging
=====================


Profiling kernel execution
--------------------------

You can use :class:`mirgecom.profiling.PyOpenCLProfilingArrayContext` instead of
:class:`~meshmode.array_context.PyOpenCLArrayContext` to profile kernel executions.
In addition to using this array context, you also need to enable profiling in the underlying
:class:`pyopencl.CommandQueue`, like this::

   queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

Note that profiling has a performance impact (~20% at the time of this writing).

.. automodule:: mirgecom.profiling


Time series logging
-------------------

Mirgecom supports logging of simulation and profiling quantities with the help
of :mod:`logpyle`. Logpyle requires
classes to describe how quantities for logging are calculated. For mirgecom, these classes are described below.

.. automodule:: mirgecom.logging_quantities

An overview of how to use logpyle is given in the :any:`Logpyle documentation <logpyle>`.

.. note::

   Note that activating logging currently has an overhead of roughly 0.2
   seconds per timestep at the time of this writing, in addition to any
   overhead from the kernel profiling.
