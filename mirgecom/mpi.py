"""MPI helper functionality.

.. autofunction:: mpi_entry_point

.. autoclass:: DistributedContext
.. autoclass:: MPILikeDistributedContext
.. autoclass:: NoMPIDistributedContext
.. autoclass:: MPIDistributedContext
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

from abc import ABCMeta, abstractmethod
from functools import wraps
import os
import sys
import numpy as np

from contextlib import contextmanager


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


class DistributedContext(metaclass=ABCMeta):
    """
    A generic distributed environment.

    .. autoproperty:: rank
    .. autoproperty:: size
    .. automethod:: barrier
    .. automethod:: bcast
    .. automethod:: allreduce
    """

    @property
    @abstractmethod
    def rank(self):
        """Get the index of the current process."""
        pass

    @property
    @abstractmethod
    def size(self):
        """Get the number of processes."""
        pass

    @abstractmethod
    def barrier(self):
        """Perform a global barrier."""
        pass

    @abstractmethod
    def bcast(self, local_values, root=0) -> None:
        """
        Perform a global broadcast.

        Parameters
        ----------
        local_values:
            The value or array of values on which the broadcast is to be performed.

        root:
            The process from which the data will be broadcast.
        """
        pass

    @abstractmethod
    def allreduce(self, local_values, op):
        """
        Perform a global reduction.

        Parameters
        ----------
        local_values:
            The value or array of values on which the reduction operation is to be
            performed.

        op: str
            Reduction operation to be performed. Must be one of "min", "max", "sum",
            "prod", "lor", or "land".

        Returns
        -------
        Any ( like *local_values* )
            Returns the result of the reduction operation on *local_values*
        """
        pass


class MPILikeDistributedContext(DistributedContext):
    """
    A distributed environment that might have a communicator.

    .. autoproperty:: rank
    .. autoproperty:: size
    .. autoproperty:: comm
    .. automethod:: barrier
    .. automethod:: bcast
    .. automethod:: allreduce
    """

    @property
    @abstractmethod
    def comm(self):
        """
        Get the communicator.

        :returns: An MPI communicator or None.
        """
        pass


class NoMPIDistributedContext(MPILikeDistributedContext):
    """
    A non-distributed MPI-like environment.

    .. autoproperty:: rank
    .. autoproperty:: size
    .. autoproperty:: comm
    .. automethod:: barrier
    .. automethod:: bcast
    .. automethod:: allreduce
    """

    @property
    def rank(self):  # noqa: D102
        return 0

    @property
    def size(self):  # noqa: D102
        return 1

    def barrier(self):  # noqa: D102
        pass

    @property
    def comm(self):
        """
        Get the communicator.

        :returns: None.
        """
        return None

    def bcast(self, local_values, root=0) -> None:  # noqa: D102
        if root != 0:
            raise ValueError("Invalid root.")

    def allreduce(self, local_values, op):  # noqa: D102
        if np.ndim(local_values) == 0:
            return local_values
        else:
            op_to_numpy_func = {
                "min": np.minimum,
                "max": np.maximum,
                "sum": np.add,
                "prod": np.multiply,
                "lor": np.logical_or,
                "land": np.logical_and,
            }
            from functools import reduce
            return reduce(op_to_numpy_func[op], local_values)


class MPIDistributedContext(MPILikeDistributedContext):
    """
    An MPI-based distributed environment.

    .. automethod:: __init__
    .. autoproperty:: rank
    .. autoproperty:: size
    .. autoproperty:: comm
    .. automethod:: barrier
    .. automethod:: bcast
    .. automethod:: allreduce
    """

    def __init__(self, comm):
        self._comm = comm

    @property
    def rank(self):  # noqa: D102
        return self.comm.Get_rank()

    @property
    def size(self):  # noqa: D102
        return self.comm.Get_size()

    @property
    def comm(self):
        """
        Get the communicator.

        :returns: An MPI communicator.
        """
        return self._comm

    def barrier(self):  # noqa: D102
        self.comm.barrier()

    def bcast(self, local_values, root=0) -> None:  # noqa: D102
        self.comm.bcast(local_values, root=root)

    def allreduce(self, local_values, op):  # noqa: D102
        from mpi4py import MPI
        op_to_mpi_op = {
            "min": MPI.MIN,
            "max": MPI.MAX,
            "sum": MPI.SUM,
            "prod": MPI.PROD,
            "lor": MPI.LOR,
            "land": MPI.LAND,
        }
        return self.comm.allreduce(local_values, op=op_to_mpi_op[op])


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
        # https://mirgecom.readthedocs.io/en/latest/running/large-systems.html
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

        func(*args, dist_ctx=MPIDistributedContext(MPI.COMM_WORLD), **kwargs)

    return wrapped_func
