Profiling kernel execution
==========================

You can use :class:`mirgecom.profiling.PyOpenCLProfilingArrayContext` instead of
:class:`~meshmode.array_context.PyOpenCLArrayContext` to profile kernel executions.
In addition to using this array context, you also need to enable profiling in the underlying
:class:`pyopencl.CommandQueue`, like this::

   queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

Note that profiling has a performance impact (~20% at the time of this writing).

.. automodule:: mirgecom.profiling
