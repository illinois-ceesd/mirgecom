"""MPI helper functionality.

.. autofunction:: mpi_entry_point
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

        # Check whether multiple ranks are running on the same GPU on each node.
        # Only works with pocl-cuda devices currently.
        if size > 1:
            node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
            cl_ctx = cl.create_some_context()
            dev = cl_ctx.get_info(cl.context_info.DEVICES)[0]
            platform = dev.get_info(cl.device_info.PLATFORM)
            platform_name = platform.get_info(cl.platform_info.NAME)

            if platform_name == "Portable Computing Language":
                try:
                    domain_id = dev.get_info(0x4010)
                except cl._cl.LogicError:
                    from warnings import warn
                    warn("Can not detect whether multiple ranks are running on the"
                         " same GPU because you need at least pocl version 1.7.")
                else:
                    bus_id = dev.get_info(cl.device_info.PCI_BUS_ID_NV)
                    slot_id = dev.get_info(cl.device_info.PCI_SLOT_ID_NV)

                    dev_id = hash(tuple((domain_id, bus_id, slot_id)))

                    dev_ids = node_comm.gather(dev_id, root=0)

                    if rank == 0:
                        if len(dev_ids) != len(set(dev_ids)):
                            print(dev_ids)
                            raise RuntimeError(
                                  "Multiple ranks are running on the same GPU.")

            else:
                from warnings import warn
                warn("Can not detect whether multiple ranks are running on the "
                     f" same GPU on platform '{platform_name}'.")

        func(*args, **kwargs)

    return wrapped_func
