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

                from warnings import warn
                warn(f"Multiple ranks are sharing GPUs on node '{hostname}'. "
                     f"Duplicate PCIe IDs: {dup}.")


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


################################################
#
# CPU helper functions for sending and receiving
#
################################################
def _isend_cpu(mpi_communicator, data_ary, receiver_rank, Tag, data_ary_size, profiler):
    """
    Returns :
        MPI send request
    Inputs :
         data_ary : data to be communicated -- in a format that the array context understands
    receiver_rank : MPI rank receiving data
              Tag : MPI communication tag 
    """
    if profiler:
        profiler.init_start(data_ary_size, receiver=receiver_rank, sender=None)
    Return_Request = mpi_communicator.Isend(data_ary, receiver_rank, tag=Tag)
    if profiler:
        profiler.init_stop(receiver_rank)
    
    return Return_Request 
    
def _irecv_cpu(mpi_communicator, data_ary, sender_rank, Tag, data_ary_size, profiler):
    """
    Returns mpi recv request
    """
    if profiler:
        profiler.init_start(data_ary_size, receiver=None, sender=sender_rank)
    Return_Request = mpi_communicator.Irecv(data_ary, sender_rank, tag=Tag)
    if profiler:
        profiler.init_stop(sender_rank)

    return Return_Request 

def _wait_cpu(mpi_req, profiler):
    """
    Returns nothing 
    """
    if profiler:
        profiler.finish_start()
    mpi_req.Wait()
    if profiler:
        profiler.finish_stop()

def _waitsome_cpu(mpi_reqs, profiler):
    """
    Returns indices 
    """
    from mpi4py import MPI
    if profiler:
        profiler.finish_start()
    indices = MPI.Request.Waitsome(mpi_reqs)
    if profiler:
        profiler.finish_stop()

    return indices

def _waitall_cpu(mpi_reqs, profiler):
    """
    Returns nothing 
    """
    from mpi4py import MPI
    if profiler:
        profiler.finish_start()
    MPI.Request.waitall(mpi_reqs)
    if profiler:
        profiler.finish_stop()

################################################
#
# Profiling object to contain all communication
# data
#
################################################
class CommunicationProfile:
    """
    Holds all communication profile data
    """

    def __init__(self):
        """
        Initializes communication profile object
        """
        self.ppn = 4

        # Variables to hold total times for sends and receives
        # as well as the data transfer costs encurred for communication
        self.init_t    = 0.0 
        self.finish_t  = 0.0 
        self.dev_cpy_t = 0.0

        self.init_t_start    = 0.0 
        self.finish_t_start  = 0.0 
        self.dev_cpy_t_start = 0.0

        self.init_inter_t = 0.0
        self.init_intra_t = 0.0
        self.finish_inter_t = 0.0
        self.finish_intra_t = 0.0

        self.init_inter_t_start = 0.0
        self.init_intra_t_start = 0.0
        self.finish_inter_t_start = 0.0
        self.finish_intra_t_start = 0.0

        # Variables to hold numbers of messages initialized and received
        # as well as the number of data copies to and from device
        self.init_m    = 0
        self.finish_m  = 0 
        self.dev_cpy_m = 0

        # Lists to contain ALL messages sizes for initialized and
        # received messages per rank
        self.send_msg_sizes   = []
        self.recv_msg_sizes = []

    def reset(self):
        self.init_t    = 0.0 
        self.finish_t  = 0.0 
        self.dev_cpy_t = 0.0

        self.init_m    = 0
        self.finish_m  = 0 
        self.dev_cpy_m = 0

        self.send_msg_sizes   = []
        self.recv_msg_sizes = []

    def init_start(self, msg_size=None, receiver=None, sender=None):
        from mpi4py import MPI
        # if receiver:
        #    my_rank       = MPI.COMM_WORLD.Get_rank()
        #    remainder     = my_rank % self.ppn 
        #    highest_local = self.ppn - remainder + my_rank
        #    lowest_local  = my_rank - remainder 
        #    if (receiver < lowest_local) or (receiver > highest_local):
        #        self.init_inter_t_start = MPI.Wtime()
        #    else:
        #        self.init_intra_t_start = MPI.Wtime()
        # else: 
        self.init_t_start = MPI.Wtime()

        self.init_m += 1
        if msg_size and receiver:
            #print('to ', receiver,'sz ', msg_size)
            self.send_msg_sizes.append([receiver, msg_size])
        elif msg_size and sender:
            #print('from ', sender,'sz ', msg_size)
            self.recv_msg_sizes.append([sender, msg_size])

    def init_stop(self, receiver=None):
        from mpi4py import MPI
        # if receiver:
        #    my_rank       = MPI.COMM_WORLD.Get_rank()
        #    remainder     = my_rank % self.ppn 
        #    highest_local = self.ppn - remainder + my_rank
        #    lowest_local  = my_rank - remainder 
        #    if (receiver < lowest_local) or (receiver > highest_local):
        #        self.init_inter_t += (MPI.Wtime() - self.init_inter_t_start)
        #    else:
        #        self.init_intra_t += (MPI.Wtime() - self.init_intra_t_start)
        #else: 
        self.init_t += (MPI.Wtime() - self.init_t_start)

    def finish_start(self, msg_size=None, receiver=None):
        from mpi4py import MPI
        if receiver:
            my_rank       = MPI.COMM_WORLD.Get_rank()
            remainder     = my_rank % self.ppn 
            highest_local = self.ppn - remainder + my_rank
            lowest_local  = my_rank - remainder 
            if (receiver < lowest_local) or (receiver > highest_local):
                self.finish_inter_t_start = MPI.Wtime()
            else:
                self.finish_intra_t_start = MPI.Wtime()
        else: 
            self.finish_t_start = MPI.Wtime()
        
        self.finish_m += 1
        
    def finish_stop(self, receiver=None):
        from mpi4py import MPI
        if receiver:
            my_rank       = MPI.COMM_WORLD.Get_rank()
            remainder     = my_rank % self.ppn 
            highest_local = self.ppn - remainder + my_rank
            lowest_local  = my_rank - remainder 
            if (receiver < lowest_local) or (receiver > highest_local):
                self.finish_inter_t += (MPI.Wtime() - self.finish_inter_t_start)
            else:
                self.finish_intra_t += (MPI.Wtime() - self.finish_intra_t_start)
        else: 
            self.finish_t += (MPI.Wtime() - self.finish_t_start)

    def dev_copy_start(self):
        from mpi4py import MPI
        self.dev_cpy_t_start = MPI.Wtime()
        self.dev_cpy_m += 1
    
    def dev_copy_stop(self):
        from mpi4py import MPI
        self.dev_cpy_t += (MPI.Wtime() - self.dev_cpy_t_start)

    def average(self):
        """
        Returns profiling data averages in a tuple of the form:
        ( init_avg, finish_avg, dev_avg )
        
          init_avg : initialization time average, 
        finish_avg : finishing communication time average,
           dev_avg : average amount of time spent copying data to/from device
        """

        init_avg     = self.init_t / self.init_m
        finish_avg   = self.finish_t / self.finish_m
        dev_copy_avg = self.dev_cpy_t / self.dev_cpy_m

        return (init_avg, finish_avg, dev_copy_avg)

    def finalize(self):
        """
        Finalizes profiling data
        """
        self.print_profile()
        self.print_msg_sizes()

        return

    def print_profile(self):
        """
        Formatted print of profiling data
        """
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        num_procs = MPI.COMM_WORLD.Get_size()

        for i in range(num_procs):
            if i == rank:
                print(F'------------------Process {rank:4d}------------')
                print(F'Init Total Time {self.init_intra_t+self.init_inter_t:.5f}')
                print(F'Init Intra Time {self.init_intra_t:.5f}')
                print(F'Init Inter Time {self.init_inter_t:.5f}')
                print(F'Init Messages {self.init_m:4d}')
                print(F'Finish Total Time {self.finish_t:.5f}')
                print(F'Finish Messages {self.finish_m:4d}')
                print(F'Device Copy Total Time {self.dev_cpy_t:.5f}')
                print(F'Device Copies {self.dev_cpy_m:4d}')
            MPI.COMM_WORLD.Barrier()

        return

    def print_msg_sizes(self):
        import numpy as np
        from mpi4py import MPI
        p = MPI.COMM_WORLD.Get_rank()
        np.save('initialized_send_msg_sizes_p'+str(p), np.array(self.send_msg_sizes))
        np.save('initialized_recv_msg_sizes_p'+str(p), np.array(self.recv_msg_sizes))
        return 

################################################
#
# Communicator object to hold which sending
# and receiving functions to use 
#
################################################
class ProfilingCommunicator:
    """
    Communication class
    actx : meshmode array_context
    """

    def __init__(self, comm=None, cflag=False, profile=True):
        """
        Initialization function
        """
        from mpi4py import MPI
        self.mpi_communicator = comm
        if comm is None:
            self.mpi_communicator = MPI.COMM_WORLD

        self.d_type       = MPI.DOUBLE # The MPI datatype being communicated
        self.cuda_flag    = cflag      # Whether the MPI is CUDA-Aware and running on Nvidia GPU
        self.comm_profile = None       # Communication profile is not initialized unless profile
                                       # flag is set

        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Initialize communication routines to be performed via CPU communication
        self.isend    = _isend_cpu
        self.irecv    = _irecv_cpu
        self.wait     = _wait_cpu
        self.waitall  = _waitall_cpu
        self.waitsome = _waitsome_cpu

        # Create CommunicationProfile object
        if profile:
            self.comm_profile = CommunicationProfile()

    def Isend(self, data_ary, dest=None, tag=None, size=0):
        """
             data_ary : data to be communicated -- in a format that the array context understands
        receiver_rank : MPI rank receiving data
                  Tag : MPI communication tag 
        """
        return self.isend(self.mpi_communicator, data_ary, dest, tag, size, self.comm_profile)

    def Irecv(self, buf=None, source=None, tag=None, size=0):
        """
             data_ary : aray for data to be received into -- in a format that the array context understands
          sender_rank : MPI rank sending data
                  Tag : MPI communication tag 
        """
        return self.irecv(self.mpi_communicator, buf, source, tag, size, self.comm_profile)

    def Wait(self, mpi_req):
        self.wait(mpi_req, self.comm_profile)

    def Waitsome(self, mpi_reqs):
        return self.waitsome(mpi_reqs, self.comm_profile)

    def Waitall(self, mpi_reqs):
        self.waitall(mpi_reqs, self.comm_profile)

    def allreduce(self, array, op=None):
        from mpi4py import MPI
        if op is None:
            op = MPI.MAX
        return self.mpi_communicator.allreduce(array, op=op)
    
    def reduce(self, obj, op=None, root=0):
        from mpi4py import MPI
        if op is None:
            op = MPI.MAX
        return self.mpi_communicator.reduce(obj, op=op, root=root)
    
    def barrier(self):
        return self.mpi_communicator.barrier()
    
    def Barrier(self):
        return self.mpi_communicator.Barrier()

    def bcast(self, obj, root=0):
        return self.mpi_communicator.bcast(obj, root=0)
    
    def gather(self, obj, root=0):
        return self.mpi_communicator.gather(obj, root=0)
    
    def Get_rank(self):
        return self.mpi_communicator.Get_rank()
    
    def Get_size(self):
        return self.mpi_communicator.Get_size()

    def Dup(self):
        return self.mpi_communicator.Dup()
