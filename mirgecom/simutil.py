"""Provide some utilities for building simulation applications.

.. autofunction:: check_step
.. autofunction:: sim_checkpoint
.. autofunction:: create_parallel_grid
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

import logging

from mirgecom.io import make_status_message

logger = logging.getLogger(__name__)


def check_step(step, interval):
    """
    Check step number against a user-specified interval.

    Utility is used typically for visualization.

    - Negative numbers mean 'never visualize'.
    - Zero means 'always visualize'.

    Useful for checking whether the current step is an output step,
    or anyting else that occurs on fixed intervals.
    """
    if interval == 0:
        return True
    elif interval < 0:
        return False
    elif step % interval == 0:
        return True
    return False


def sim_checkpoint(state, step=0, t=0, dt=0, nstatus=-1,
        get_extra_status=None, nviz=-1, write_vis=None, comm=None):
    """Check simulation health, status, viz dumps, and restart."""
    rank = comm.Get_rank() if comm is not None else 0

    if check_step(step, nstatus):
        statusmsg = make_status_message(step=step, t=t, dt=dt,
            extra_status=get_extra_status(step=step, t=t, dt=dt, state=state) if
            get_extra_status else None)
        if rank == 0:
            logger.info(statusmsg)
    if check_step(step, nviz):
        if write_vis is not None:
            write_vis(step, t, state)


def create_parallel_grid(comm, generate_grid):
    """Create and partition a grid.

    Create a grid with the user-supplied grid generation function
    *generate_grid*, partition the grid, and distribute it to every
    rank in the provided MPI communicator *comm*.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the grid
    generate_grid:
        Callable of zero arguments returning a :class:`meshmode.mesh.Mesh`.
        Will only be called on one (undetermined) rank.

    Returns
    -------
    local_mesh : :class:`meshmode.mesh.Mesh`
        The local partition of the the mesh returned by *generate_grid*.
    global_nelements : :class:`int`
        The number of elements in the serial grid
    """
    from meshmode.distributed import (
        MPIMeshDistributor,
        get_partition_by_pymetis,
    )
    num_parts = comm.Get_size()
    mesh_dist = MPIMeshDistributor(comm)
    global_nelements = 0

    if mesh_dist.is_mananger_rank():

        mesh = generate_grid()

        global_nelements = mesh.nelements

        part_per_element = get_partition_by_pymetis(mesh, num_parts)
        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    return local_mesh, global_nelements
