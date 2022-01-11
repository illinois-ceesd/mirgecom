"""MPI helper functionality.

.. autofunction:: mpi_entry_point
.. autofunction:: pudb_remote_debug_on_single_rank
.. autofunction:: enable_rank_labeled_print
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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

from functools import wraps
import os
import sys

from contextlib import contextmanager
from typing import Callable, Any


@contextmanager
def shared_split_comm_world():
    """Create a context manager for a MPI.COMM_TYPE_SHARED comm."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

    try:
        yield comm
    finally:
        comm.Free()


def _check_gpu_oversubscription():
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

    cl_ctx = cl.create_some_context()
    dev = cl_ctx.devices

    # No support for multi-device contexts
    if len(dev) > 1:
        raise NotImplementedError("multi-device contexts not yet supported")

    dev = dev[0]

    # Allow running multiple ranks on non-GPU devices
    if not (dev.type & cl.device_type.GPU):
        return

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
            if len(dev_ids) != len(set(dev_ids)):
                hostname = MPI.Get_processor_name()
                dup = [item for item in dev_ids if dev_ids.count(item) > 1]

                raise RuntimeError(
                      f"Multiple ranks are sharing GPUs on node '{hostname}'."
                      f" Duplicate PCIe IDs: {dup}.")


def mpi_entry_point(func):
    """
    Return a decorator that designates a function as the "main" function for MPI.

    Declares that all MPI code that will be executed on the current process is
    contained within *func*. Calls `MPI_Init()`/`MPI_Init_thread()` and sets up a
    hook to call `MPI_Finalize()` on exit.
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if "mpi4py.run" not in sys.modules:
            raise RuntimeError("Must run MPI scripts via mpi4py (i.e., 'python -m "
                        "mpi4py <args>').")

        if "mpi4py.MPI" in sys.modules:
            raise RuntimeError("mpi4py.MPI imported before designated MPI entry "
                        "point. Check for prior imports.")

        # Avoid hwloc version conflicts by forcing pocl to load before mpi4py
        # (don't ask). See https://github.com/illinois-ceesd/mirgecom/pull/169
        # for details.
        import pyopencl as cl
        cl.get_platforms()

        # Avoid https://github.com/illinois-ceesd/mirgecom/issues/132 on
        # some MPI runtimes.
        import mpi4py
        mpi4py.rc.recv_mprobe = False

        # Runs MPI_Init()/MPI_Init_thread() and sets up a hook for MPI_Finalize() on
        # exit
        from mpi4py import MPI

        # This code warns the user of potentially slow startups due to file system
        # locking when running with large numbers of ranks. See
        # https://mirgecom.readthedocs.io/en/latest/running.html#running-with-large-numbers-of-ranks-and-nodes
        # for more details
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        if size > 1 and rank == 0 and "XDG_CACHE_HOME" not in os.environ:
            from warnings import warn
            warn("Please set the XDG_CACHE_HOME variable in your job script to "
                 "avoid file system overheads when running on large numbers of "
                 "ranks. See https://mirgecom.readthedocs.io/en/latest/running.html#running-with-large-numbers-of-ranks-and-nodes"  # noqa: E501
                 " for more information.")

        _check_gpu_oversubscription()

        func(*args, **kwargs)

    return wrapped_func


def pudb_remote_debug_on_single_rank(func: Callable):
    """
    Designate a function *func* to be debugged with ``pudb`` on rank 0.

    To use it, add this decorator to the main function that you want to debug,
    after the :func:`mpi_entry_point` decorator:

    .. code-block:: python

        @mpi_entry_point
        @pudb_remote_debug_on_single_rank
        def main(...)


    Then, you can connect to pudb on rank 0 by running
    ``telnet 127.0.0.1 6899`` in a separate terminal and continue to use pudb
    as normal.
    """
    @wraps(func)
    def wrapped_func(*args: Any, **kwargs: Any):
        # pylint: disable=import-error
        from pudb.remote import debug_remote_on_single_rank
        from mpi4py import MPI
        debug_remote_on_single_rank(MPI.COMM_WORLD, 0, func, *args, **kwargs)

    return wrapped_func


def enable_rank_labeled_print() -> None:
    """Enable prepending the rank number to every message printed with print()."""
    def rank_print(*args, **kwargs):
        """Prepends the rank number to the print function."""
        if "mpi4py.MPI" in sys.modules:
            from mpi4py import MPI
            out_str = f"[{MPI.COMM_WORLD.Get_rank()}]"
        else:
            out_str = "[ ]"

        __builtins__["oldprint"](out_str, *args, **kwargs)

    if "oldprint" not in __builtins__:
        __builtins__["oldprint"] = __builtins__["print"]
    __builtins__["print"] = rank_print
