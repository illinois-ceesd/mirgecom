"""Provide some utilities for restarting simulations.

.. autofunction:: read_restart_data
.. autofunction:: write_restart_file
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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

import pickle
from meshmode.dof_array import array_context_for_pickling


def read_restart_data(actx, filename):
    """Read the raw restart data dictionary from the given pickle restart file."""
    with array_context_for_pickling(actx):
        with open(filename, "rb") as f:
            return pickle.load(f)


def write_restart_file(actx, restart_data, filename, comm=None):
    """Pickle the simulation data into a file for use in restarting."""
    rank = 0
    if comm:
        rank = comm.Get_rank()
    if rank == 0:
        import os
        rst_dir = os.path.dirname(filename)
        if rst_dir:
            os.makedirs(rst_dir, exist_ok=True)
    if comm:
        comm.barrier()
    with array_context_for_pickling(actx):
        with open(filename, "wb") as f:
            pickle.dump(restart_data, f)


def redist_restart_data(actx, src_data=None, rst_filename=None, mesh_filename=None,
                        get_mesh_data=None, partition_generator_func=None,
                        comm=None, logmgr=None):
    """Redistribute the restart data."""
    # rank = 0
    nparts = 1
    nparts_old = src_data["num_parts"]
    if comm:
        nparts = comm.Get_size()
        # rank = comm.Get_rank()
        comm.Broadcast
    if nparts == nparts_old:
        return src_data

    # temporarily
    return src_data


def perform_restart(actx, restart_filename, mesh_filename=None,
                    get_mesh_data=None, partition_generator_func=None,
                    comm=None, logmgr=None):
    """Restart solution even if decomp changes."""
    import os
    nparts = 1

    if comm:
        nparts = comm.Get_size()

    rst_data = {}
    if os.path.exists(restart_filename):
        with array_context_for_pickling(actx):
            with open(restart_filename, "rb") as f:
                rst_data = pickle.load(f)
                if rst_data["num_parts"] == nparts:
                    return rst_data

    return redist_restart_data(actx, src_data=rst_data,
                               rst_filename=restart_filename, comm=comm,
                               mesh_filename=mesh_filename,
                               get_mesh_data=get_mesh_data,
                               partition_generator_func=partition_generator_func,
                               logmgr=logmgr)
