"""Provide some utilities for building simulation applications.

.. autofunction:: check_step
.. autofunction:: inviscid_sim_timestep
.. autoexception:: ExactSolutionMismatch
.. autofunction:: sim_checkpoint
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
from meshmode.dof_array import thaw
from mirgecom.io import make_status_message
from mirgecom.inviscid import get_inviscid_timestep  # bad smell?

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
    t_remaining = t_final - t
    if t_remaining < dt:
        return max(0, t_remaining)
    if constant_cfl is True:
        from grudge.op import nodal_min
        mydt = cfl * nodal_min(
            discr, "vol",
            get_inviscid_timestep(discr=discr, eos=eos, cv=state)
        )
    return mydt


class ExactSolutionMismatch(Exception):
    """Exception class for solution mismatch.

    .. attribute:: step
    .. attribute:: t
    .. attribute:: state
    """

    def __init__(self, step, t, state):
        """Record the simulation state on creation."""
        self.step = step
        self.t = t
        self.state = state


def sim_checkpoint(discr, visualizer, eos, cv, vizname, exact_soln=None,
                   step=0, t=0, dt=0, cfl=1.0, nstatus=-1, nviz=-1, exittol=1e-16,
                   constant_cfl=False, comm=None, viz_fields=None, overwrite=False,
                   vis_timer=None):
    """Check simulation health, status, viz dumps, and restart."""
    do_viz = check_step(step=step, interval=nviz)
    do_status = check_step(step=step, interval=nstatus)
    if do_viz is False and do_status is False:
        return 0

    dependent_vars = eos.dependent_vars(cv)

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    maxerr = 0.0
    if exact_soln is not None:
        actx = cv.mass.array_context
        nodes = thaw(actx, discr.nodes())
        expected_state = exact_soln(x_vec=nodes, t=t, eos=eos)
        exp_resid = cv - expected_state
        err_norms = [discr.norm(v, np.inf) for v in exp_resid.join()]
        maxerr = discr.norm(exp_resid.join(), np.inf)

    if do_viz:
        io_fields = [
            ("cv", cv),
            ("dv", dependent_vars)
        ]
        if exact_soln is not None:
            exact_list = [
                ("exact_soln", expected_state),
            ]
            io_fields.extend(exact_list)
        if viz_fields is not None:
            io_fields.extend(viz_fields)

        from mirgecom.io import make_rank_fname, make_par_fname
        rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)

        from contextlib import nullcontext

        if vis_timer:
            ctm = vis_timer.start_sub_timer()
        else:
            ctm = nullcontext()

        with ctm:
            visualizer.write_parallel_vtk_file(comm, rank_fn, io_fields,
                overwrite=overwrite, par_manifest_filename=make_par_fname(
                    basename=vizname, step=step, t=t))

    if do_status is True:
        #        if constant_cfl is False:
        #            current_cfl = get_inviscid_cfl(discr=discr, q=q,
        #                                           eos=eos, dt=dt)
        statusmesg = make_status_message(discr=discr, t=t, step=step, dt=dt,
                                         cfl=cfl, dependent_vars=dependent_vars)
        if exact_soln is not None:
            statusmesg += (
                "\n------- errors="
                + ", ".join("%.3g" % en for en in err_norms))

        if rank == 0:
            logger.info(statusmesg)

    if maxerr > exittol:
        raise ExactSolutionMismatch(step, t=t, state=cv)


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
