"""Provide some utilities for building simulation applications.

.. autofunction:: check_step
.. autofunction:: inviscid_sim_timestep
.. autoexception:: ExactSolutionMismatch
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

import numpy as np
from meshmode.dof_array import thaw
from mirgecom.io import make_status_message
from mirgecom.euler import (
    get_inviscid_timestep,
)

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
    if constant_cfl is True:
        mydt = get_inviscid_timestep(discr=discr, q=state,
                                     cfl=cfl, eos=eos)
    if (t + mydt) > t_final:
        mydt = t_final - t
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


def sim_checkpoint(discr, visualizer, eos, q, vizname, exact_soln=None,
                   step=0, t=0, dt=0, cfl=1.0, nstatus=-1, nviz=-1, exittol=1e-16,
                   constant_cfl=False, comm=None, overwrite=False):
    """Check simulation health, status, viz dumps, and restart."""
    # TODO: Add restart
    do_viz = check_step(step=step, interval=nviz)
    do_status = check_step(step=step, interval=nstatus)
    if do_viz is False and do_status is False:
        return 0

    from mirgecom.euler import split_conserved
    cv = split_conserved(discr.dim, q)
    dependent_vars = eos.dependent_vars(cv)

    from mirgecom.tag_cells import smoothness_indicator
    tagedcells = smoothness_indicator(q[0], discr)

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    maxerr = 0.0
    if exact_soln is not None:
        actx = cv.mass.array_context
        nodes = thaw(actx, discr.nodes())
        expected_state = exact_soln(t=t, x_vec=nodes, eos=eos)
        exp_resid = q - expected_state
        err_norms = [discr.norm(v, np.inf) for v in exp_resid]
        maxerr = max(err_norms)

    if do_viz:
        io_fields = [
            ("cv", cv),
            ("dv", dependent_vars),
            ("tagged", tagedcells)
        ]
        if exact_soln is not None:
            exact_list = [
                ("exact_soln", expected_state),
            ]
            io_fields.extend(exact_list)

        from mirgecom.io import make_rank_fname, make_par_fname
        rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)
        visualizer.write_parallel_vtk_file(
            comm, rank_fn, io_fields, overwrite=overwrite,
            par_manifest_filename=make_par_fname(basename=vizname, step=step, t=t))

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
        raise ExactSolutionMismatch(step, t=t, state=q)


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
