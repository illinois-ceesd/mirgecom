"""Provide some utilities for building simulation applications.

General utilities
-----------------

.. autofunction:: check_step
.. autofunction:: inviscid_sim_timestep
.. autofunction:: write_visfile
.. autofunction:: allsync

Diagnostic utilities
--------------------

.. autofunction:: compare_fluid_solutions
.. autofunction:: check_naninf_local
.. autofunction:: check_range_local

Mesh utilities
--------------

.. autofunction:: generate_and_distribute_mesh
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

import numpy as np
from mirgecom.inviscid import get_inviscid_timestep  # bad smell?
import grudge.op as op

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


def inviscid_sim_timestep(discr, state, t, dt, cfl, eos,
                          t_final, constant_cfl=False):
    """Return the maximum stable dt."""
    mydt = dt
    t_remaining = max(0, t_final - t)
    if constant_cfl is True:
        mydt = get_inviscid_timestep(discr=discr, cv=state,
                                     cfl=cfl, eos=eos)
    return min(t_remaining, mydt)


def write_visfile(discr, io_fields, visualizer, vizname,
                  step=0, t=0, overwrite=False, vis_timer=None):
    """Write VTK output for the fields specified in *io_fields*.

    Parameters
    ----------
    visualizer:
        A :class:`meshmode.discretization.visualization.Visualizer`
        VTK output object.
    io_fields:
        List of tuples indicating the (name, data) for each field to write.
    """
    from contextlib import nullcontext
    from mirgecom.io import make_rank_fname, make_par_fname

    comm = discr.mpi_communicator
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)

    if vis_timer:
        ctm = vis_timer.start_sub_timer()
    else:
        ctm = nullcontext()

    with ctm:
        visualizer.write_parallel_vtk_file(
            comm, rank_fn, io_fields,
            overwrite=overwrite,
            par_manifest_filename=make_par_fname(
                basename=vizname, step=step, t=t
            )
        )


def allsync(local_values, comm=None, op=None):
    """Perform allreduce if MPI comm is provided."""
    if comm is None:
        return local_values
    if op is None:
        from mpi4py import MPI
        op = MPI.MAX
    return comm.allreduce(local_values, op=op)


def check_range_local(discr, dd, field, min_value, max_value):
    """Check for any negative values."""
    return (
        op.nodal_min_loc(discr, dd, field) < min_value
        or op.nodal_max_loc(discr, dd, field) > max_value
    )


def check_naninf_local(discr, dd, field):
    """Check for any NANs or Infs in the field."""
    s = op.nodal_sum_loc(discr, dd, field)
    return np.isnan(s) or (s == np.inf)


def compare_fluid_solutions(discr, red_state, blue_state):
    """Return inf norm of (*red_state* - *blue_state*) for each component."""
    resid = red_state - blue_state
    return [discr.norm(v, np.inf) for v in resid.join()]


def generate_and_distribute_mesh(comm, generate_mesh):
    """Generate a mesh and distribute it among all ranks in *comm*.

    Generate the mesh with the user-supplied mesh generation function
    *generate_mesh*, partition the mesh, and distribute it to every
    rank in the provided MPI communicator *comm*.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the mesh
    generate_mesh:
        Callable of zero arguments returning a :class:`meshmode.mesh.Mesh`.
        Will only be called on one (undetermined) rank.

    Returns
    -------
    local_mesh : :class:`meshmode.mesh.Mesh`
        The local partition of the the mesh returned by *generate_mesh*.
    global_nelements : :class:`int`
        The number of elements in the serial mesh
    """
    from meshmode.distributed import (
        MPIMeshDistributor,
        get_partition_by_pymetis,
    )
    num_parts = comm.Get_size()
    mesh_dist = MPIMeshDistributor(comm)
    global_nelements = 0

    if mesh_dist.is_mananger_rank():

        mesh = generate_mesh()

        global_nelements = mesh.nelements

        part_per_element = get_partition_by_pymetis(mesh, num_parts)
        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    return local_mesh, global_nelements


def create_parallel_grid(comm, generate_grid):
    """Generate and distribute mesh compatibility interface."""
    from warnings import warn
    warn("Do not call create_parallel_grid; use generate_and_distribute_mesh "
         "instead. This function will disappear August 1, 2021",
         DeprecationWarning, stacklevel=2)
    return generate_and_distribute_mesh(comm=comm, generate_mesh=generate_grid)
