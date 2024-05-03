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
import os
import logging

import pyopencl as cl
from arraycontext import ArrayContext, PyOpenCLArrayContext

logger = logging.getLogger(__name__)


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


def _check_cache_dirs_node() -> None:
    """Check whether multiple ranks share cache directories on the same node."""
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()

    if size <= 1:
        return

    from mirgecom.mpi import shared_split_comm_world

    with shared_split_comm_world() as node_comm:
        node_rank = node_comm.Get_rank()

        def _check_var(var: str) -> None:
            from warnings import warn

            try:
                my_path = os.environ[var]
            except KeyError:
                warn(f"Please set the '{var}' variable in your job script to "
                    "avoid file system overheads when running on large numbers of "
                    "ranks. See https://mirgecom.readthedocs.io/en/latest/running/large-systems.html "  # noqa: E501
                    "for more information.")
                # Create a fake path so there will not be a second warning below.
                my_path = f"no/such/path/rank{node_rank}"

            all_paths = node_comm.gather(my_path, root=0)

            if node_rank == 0:
                assert all_paths
                if len(all_paths) != len(set(all_paths)):
                    hostname = MPI.Get_processor_name()
                    dup = [path for path in set(all_paths)
                                if all_paths.count(path) > 1]

                    from warnings import warn
                    warn(f"Multiple ranks are sharing '{var}' on node '{hostname}'. "
                        f"Duplicate '{var}'s: {dup}.")

        _check_var("XDG_CACHE_HOME")

        if os.environ.get("XDG_CACHE_HOME") is None:
            # When XDG_CACHE_HOME is set but POCL_CACHE_DIR is not, pocl
            # will use XDG_CACHE_HOME as the cache directory.
            _check_var("POCL_CACHE_DIR")

        # We haven't observed an issue yet that 'CUDA_CACHE_PATH' fixes,
        # so disable this check for now.
        # _check_var("CUDA_CACHE_PATH")


def _check_gpu_oversubscription(actx: PyOpenCLArrayContext) -> None:
    """
    Check whether multiple ranks are running on the same GPU on each node.

    Only works with CUDA devices currently due to the use of the
    PCI_DOMAIN_ID_NV extension.
    """
    from mpi4py import MPI
    import pyopencl as cl

    size = MPI.COMM_WORLD.Get_size()

    if size <= 1:
        return

    dev = actx.queue.device

    # This check only works with Nvidia GPUs
    from pyopencl.characterize import nv_compute_capability
    if nv_compute_capability(dev) is None:
        return

    from mirgecom.mpi import shared_split_comm_world

    with shared_split_comm_world() as node_comm:
        try:
            domain_id = hex(dev.pci_domain_id_nv)
        except (cl._cl.LogicError, AttributeError):
            from warnings import warn
            warn("Cannot detect whether multiple ranks are running on the"
                 " same GPU because it requires Nvidia GPUs running with"
                 " pyopencl>2021.1.1 and (Nvidia CL or pocl>1.6).")
            return

        node_rank = node_comm.Get_rank()

        bus_id = hex(dev.pci_bus_id_nv)
        slot_id = hex(dev.pci_slot_id_nv)

        dev_id = (domain_id, bus_id, slot_id)

        dev_ids = node_comm.gather(dev_id, root=0)

        if node_rank == 0:
            assert dev_ids
            if len(dev_ids) != len(set(dev_ids)):
                hostname = MPI.Get_processor_name()
                dup = [item for item in dev_ids if dev_ids.count(item) > 1]

                from warnings import warn
                warn(f"Multiple ranks are sharing GPUs on node '{hostname}'. "
                     f"Duplicate PCIe IDs: {dup}.")


def log_disk_cache_config(actx: PyOpenCLArrayContext) -> None:
    """Log the disk cache configuration."""
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    res = f"Rank {rank} disk cache config: "

    from pyopencl.characterize import nv_compute_capability, get_pocl_version
    dev = actx.queue.device

    # Variables set any to any value => cache is disabled
    loopy_cache_enabled = bool(os.getenv("LOOPY_NO_CACHE", True))
    pyopencl_cache_enabled = bool(os.getenv("PYOPENCL_NO_CACHE", True))

    loopy_cache_dir = ("(" + os.getenv("XDG_CACHE_HOME", "default dir") + ")"
                       if loopy_cache_enabled else "")
    pyopencl_cache_dir = ("(" + os.getenv("XDG_CACHE_HOME", "default dir") + ")"
                          if pyopencl_cache_enabled else "")

    res += f"loopy: {loopy_cache_enabled} {loopy_cache_dir}; "
    res += f"pyopencl: {pyopencl_cache_enabled} {pyopencl_cache_dir}; "

    if get_pocl_version(dev.platform) is not None:
        # Variable set to '0' => cache is disabled
        pocl_cache_enabled = os.getenv("POCL_KERNEL_CACHE", "1") != "0"
        pocl_cache_dir = ("(" + os.getenv("POCL_CACHE_DIR", "default dir") + ")"
                        if pocl_cache_enabled else "")

        res += f"pocl: {pocl_cache_enabled} {pocl_cache_dir}; "

    if nv_compute_capability(dev) is not None:
        # Variable set to '1' => cache is disabled
        cuda_cache_enabled = os.getenv("CUDA_CACHE_DISABLE", "0") != "1"
        cuda_cache_dir = ("(" + os.getenv("CUDA_CACHE_PATH", "default dir") + ")"
                          if cuda_cache_enabled else "")
        res += f"cuda: {cuda_cache_enabled} {cuda_cache_dir};"

    res += "\n"
    logger.info(res)


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

    actx = actx_class(**actx_kwargs)

    # Check cache directories and log disk cache configuration for
    # PyOpenCL-based actx (Non-PyOpenCL actx classes don't use loopy, pyopencl,
    # or pocl, and therefore we don't need to examine their caching).
    if actx_class_is_pyopencl(actx_class):
        assert isinstance(actx, PyOpenCLArrayContext)
        _check_gpu_oversubscription(actx)
        _check_cache_dirs_node()
        log_disk_cache_config(actx)

    return actx
