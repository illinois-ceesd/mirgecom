"""Provide some utilities for handling ArrayContexts.

.. autofunction:: get_reasonable_array_context_class
.. autofunction:: actx_class_is_lazy
.. autofunction:: actx_class_is_eager
.. autofunction:: actx_class_is_profiling
.. autofunction:: actx_class_is_numpy
.. autofunction:: initialize_actx
"""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Type, Dict, Any

import pyopencl as cl
from arraycontext import ArrayContext


def get_reasonable_array_context_class(*, lazy: bool, distributed: bool,
                        profiling: bool, numpy: bool = False) -> Type[ArrayContext]:
    """Return a :class:`~arraycontext.ArrayContext` with the given constraints."""
    if lazy and profiling:
        raise ValueError("Can't specify both lazy and profiling")

    if numpy:
        if profiling:
            raise ValueError("Can't specify both numpy and profiling")
        if lazy:
            raise ValueError("Can't specify both numpy and lazy")

        from warnings import warn
        warn("The NumpyArrayContext is still under development")

        if distributed:
            from grudge.array_context import MPINumpyArrayContext
            return MPINumpyArrayContext
        else:
            from grudge.array_context import NumpyArrayContext
            return NumpyArrayContext

    if profiling:
        from mirgecom.profiling import PyOpenCLProfilingArrayContext
        return PyOpenCLProfilingArrayContext

    from grudge.array_context import \
        get_reasonable_array_context_class as grudge_get_reasonable_actx_class

    return grudge_get_reasonable_actx_class(lazy=lazy, distributed=distributed)


def actx_class_is_lazy(actx_class: Type[ArrayContext]) -> bool:
    """Return True if *actx_class* is lazy."""
    from arraycontext import PytatoPyOpenCLArrayContext
    return issubclass(actx_class, PytatoPyOpenCLArrayContext)


def actx_class_is_eager(actx_class: Type[ArrayContext]) -> bool:
    """Return True if *actx_class* is eager."""
    from arraycontext import PyOpenCLArrayContext
    return issubclass(actx_class, PyOpenCLArrayContext)


def actx_class_is_profiling(actx_class: Type[ArrayContext]) -> bool:
    """Return True if *actx_class* has profiling enabled."""
    from mirgecom.profiling import PyOpenCLProfilingArrayContext
    return issubclass(actx_class, PyOpenCLProfilingArrayContext)


def actx_class_is_pyopencl(actx_class: Type[ArrayContext]) -> bool:
    """Return True if *actx_class* is PyOpenCL-based."""
    from arraycontext import PyOpenCLArrayContext
    return issubclass(actx_class, PyOpenCLArrayContext)


def actx_class_is_numpy(actx_class: Type[ArrayContext]) -> bool:
    """Return True if *actx_class* is numpy-based."""
    try:
        from grudge.array_context import NumpyArrayContext
        if issubclass(actx_class, NumpyArrayContext):
            return True
        else:
            return False
    except ImportError:
        return False


def initialize_actx(
        actx_class: Type[ArrayContext],
        comm=None, *,
        use_axis_tag_inference_fallback: bool = False,
        use_einsum_inference_fallback: bool = False) -> ArrayContext:
    """Initialize a new :class:`~arraycontext.ArrayContext` based on *actx_class*."""
    from arraycontext import PyOpenCLArrayContext, PytatoPyOpenCLArrayContext
    from grudge.array_context import (MPIPyOpenCLArrayContext,
                                      MPIPytatoArrayContext,
                                      MPINumpyArrayContext)

    actx_kwargs: Dict[str, Any] = {}

    if comm:
        actx_kwargs["mpi_communicator"] = comm

    if actx_class_is_numpy(actx_class):
        if comm:
            assert issubclass(actx_class, MPINumpyArrayContext)
        else:
            assert not issubclass(actx_class, MPINumpyArrayContext)
    else:
        cl_ctx = cl.create_some_context()
        if actx_class_is_profiling(actx_class):
            queue = cl.CommandQueue(cl_ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            queue = cl.CommandQueue(cl_ctx)
        actx_kwargs["queue"] = queue

        from mirgecom.simutil import get_reasonable_memory_pool
        alloc = get_reasonable_memory_pool(cl_ctx, queue)
        actx_kwargs["allocator"] = alloc

        if actx_class_is_lazy(actx_class):
            assert issubclass(actx_class, PytatoPyOpenCLArrayContext)
            actx_kwargs["use_axis_tag_inference_fallback"] = \
                use_axis_tag_inference_fallback
            actx_kwargs["use_einsum_inference_fallback"] = \
                use_einsum_inference_fallback
            if comm:
                assert issubclass(actx_class, MPIPytatoArrayContext)
                actx_kwargs["mpi_base_tag"] = 12000
            else:
                assert not issubclass(actx_class, MPIPytatoArrayContext)
        else:
            assert issubclass(actx_class, PyOpenCLArrayContext)
            actx_kwargs["force_device_scalars"] = True
            if comm:
                assert issubclass(actx_class, MPIPyOpenCLArrayContext)
            else:
                assert not issubclass(actx_class, MPIPyOpenCLArrayContext)

    return actx_class(**actx_kwargs)
