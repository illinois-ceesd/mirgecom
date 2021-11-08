"""Provide some utilities for building simulation applications.

General utilities
-----------------

.. autofunction:: check_step
.. autofunction:: get_sim_timestep
.. autofunction:: write_visfile
.. autofunction:: global_reduce

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
import logging
import numpy as np
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


def get_sim_timestep(discr, state, t, dt, cfl, eos,
                     t_final, constant_cfl=False):
    """Return the maximum stable timestep for a typical fluid simulation.

    This routine returns *dt*, the users defined constant timestep, or *max_dt*, the
    maximum domain-wide stability-limited timestep for a fluid simulation.

    .. important::
        This routine calls the collective: :func:`~grudge.op.nodal_min` on the inside
        which makes it domain-wide regardless of parallel domain decomposition. Thus
        this routine must be called *collectively* (i.e. by all ranks).

    Two modes are supported:
        - Constant DT mode: returns the minimum of (t_final-t, dt)
        - Constant CFL mode: returns (cfl * max_dt)

    Parameters
    ----------
    discr
        Grudge discretization or discretization collection?
    state: :class:`~mirgecom.fluid.ConservedVars`
        The fluid state.
    t: float
        Current time
    t_final: float
        Final time
    dt: float
        The current timestep
    cfl: float
        The current CFL number
    eos: :class:`~mirgecom.eos.GasEOS`
        Gas equation-of-state optionally with a non-empty
        :class:`~mirgecom.transport.TransportModel` for viscous transport properties.
    constant_cfl: bool
        True if running constant CFL mode

    Returns
    -------
    float
        The maximum stable DT based on a viscous fluid.
    """
    t_remaining = max(0, t_final - t)
    mydt = dt
    if constant_cfl:
        from mirgecom.viscous import get_viscous_timestep
        from grudge.op import nodal_min
        mydt = state.array_context.to_numpy(
            cfl * nodal_min(
                discr, "vol",
                get_viscous_timestep(discr=discr, eos=eos, cv=state)))[()]
    return min(t_remaining, mydt)


def write_visfile(discr, io_fields, visualizer, vizname,
                  step=0, t=0, overwrite=False, vis_timer=None):
    """Write VTK output for the fields specified in *io_fields*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

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

    from mirgecom.mpi import make_mpi_context
    dist_ctx = make_mpi_context(discr.mpi_communicator)

    rank_fn = make_rank_fname(basename=vizname, rank=dist_ctx.rank, step=step, t=t)

    if dist_ctx.rank == 0:
        import os
        viz_dir = os.path.dirname(rank_fn)
        if viz_dir and not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

    dist_ctx.barrier()

    if vis_timer:
        ctm = vis_timer.start_sub_timer()
    else:
        ctm = nullcontext()

    with ctm:
        visualizer.write_parallel_vtk_file(
            dist_ctx.comm, rank_fn, io_fields,
            overwrite=overwrite,
            par_manifest_filename=make_par_fname(
                basename=vizname, step=step, t=t
            )
        )


def global_reduce(local_values, op, *, comm=None):
    """Perform a global reduction (allreduce if MPI comm is provided).

    This routine is a convenience wrapper for the MPI AllReduce operation
    that also works outside of an MPI context.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    local_values: numbers.Number or numpy.ndarray
        The (MPI-compatible) value or array of values on which the
        reduction operation is to be performed.

    op: str
        Reduction operation to be performed. Must be one of "min", "max", "sum",
        "prod", "lor", or "land".

    comm:
        Optional parameter specifying the MPI communicator on which the
        reduction operation (if any) is to be performed

    Returns
    -------
    Any ( like *local_values* )
        Returns the result of the reduction operation on *local_values*
    """
    from warnings import warn
    warn("global_reduce is deprecated and will disappear in Q2 "
         "2022. Use DistributedContext.allreduce() instead.", DeprecationWarning,
         stacklevel=2)

    from mirgecom.mpi import make_mpi_context
    dist_ctx = make_mpi_context(comm)

    return dist_ctx.allreduce(local_values, op)


def allsync(local_values, comm=None, op=None):
    """
    Perform allreduce if MPI comm is provided.

    Deprecated. Do not use in new code.
    """
    from warnings import warn
    warn("allsync is deprecated and will disappear in Q1 2022. "
         "Use DistributedContext.allreduce() instead.", DeprecationWarning,
         stacklevel=2)

    from mirgecom.mpi import make_mpi_context
    dist_ctx = make_mpi_context(comm)

    from mpi4py import MPI

    if op is None:
        op = MPI.MAX

    if op == MPI.MIN:
        op_string = "min"
    elif op == MPI.MAX:
        op_string = "max"
    elif op == MPI.SUM:
        op_string = "sum"
    elif op == MPI.PROD:
        op_string = "prod"
    elif op == MPI.LOR:
        op_string = "lor"
    elif op == MPI.LAND:
        op_string = "land"
    else:
        raise ValueError(f"Unrecognized MPI reduce op {op}.")

    return dist_ctx.allreduce(local_values, op_string)


def check_range_local(discr, dd, field, min_value, max_value):
    """Check for any negative values."""
    actx = field.array_context
    return (
        actx.to_numpy(op.nodal_min_loc(discr, dd, field)) < min_value
        or actx.to_numpy(op.nodal_max_loc(discr, dd, field)) > max_value
    )


def check_naninf_local(discr, dd, field):
    """Check for any NANs or Infs in the field."""
    actx = field.array_context
    s = actx.to_numpy(op.nodal_sum_loc(discr, dd, field))
    return not np.isfinite(s)


def compare_fluid_solutions(discr, red_state, blue_state):
    """Return inf norm of (*red_state* - *blue_state*) for each component.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    actx = red_state.array_context
    resid = red_state - blue_state
    return [actx.to_numpy(discr.norm(v, np.inf)) for v in resid.join()]


def generate_and_distribute_mesh(dist_ctx, generate_mesh, *, comm=None):
    """Generate a mesh and distribute it among all ranks in *dist_ctx*.

    Generate the mesh with the user-supplied mesh generation function
    *generate_mesh*, partition the mesh, and distribute it to every
    rank in the provided :class:`mirgecom.mpi.DistributedContext`.

    .. note::
        This is a collective routine and must be called by all ranks.

    Parameters
    ----------
    dist_ctx: mirgecom.mpi.DistributedContext
        Distributed context over which to partition the mesh.
    generate_mesh:
        Callable of zero arguments returning a :class:`meshmode.mesh.Mesh`.
        Will only be called on one (undetermined) rank.

    Returns
    -------
    local_mesh: :class:`meshmode.mesh.Mesh`
        The local partition of the the mesh returned by *generate_mesh*.
    global_nelements: :class:`int`
        The number of elements in the serial mesh
    """
    from mirgecom.mpi import (
        DistributedContext,
        MPILikeDistributedContext,
        MPIDistributedContext)
    if dist_ctx is not None:
        if not isinstance(dist_ctx, DistributedContext):
            # May have passed an MPI comm positionally
            dist_ctx = MPIDistributedContext(dist_ctx)
            from warnings import warn
            warn("comm argument is deprecated and will disappear in Q2 2022. "
                 "Use dist_ctx instead.", DeprecationWarning, stacklevel=2)
    else:
        if comm is not None:
            from warnings import warn
            warn("comm argument is deprecated and will disappear in Q2 2022. "
                 "Use dist_ctx instead.", DeprecationWarning, stacklevel=2)
        from mirgecom.mpi import make_mpi_context
        dist_ctx = make_mpi_context(comm)

    if dist_ctx.size > 1:
        if not isinstance(dist_ctx, MPILikeDistributedContext):
            raise TypeError("Distributed context must be MPI-like.")

        from meshmode.distributed import (
            MPIMeshDistributor,
            get_partition_by_pymetis,
        )
        num_parts = dist_ctx.size
        mesh_dist = MPIMeshDistributor(dist_ctx.comm)
        global_nelements = 0

        if mesh_dist.is_mananger_rank():

            mesh = generate_mesh()

            global_nelements = mesh.nelements

            part_per_element = get_partition_by_pymetis(mesh, num_parts)
            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
            del mesh

        else:
            local_mesh = mesh_dist.receive_mesh_part()

        dist_ctx.bcast(global_nelements)

    else:
        local_mesh = generate_mesh()
        global_nelements = local_mesh.nelements

    return local_mesh, global_nelements
