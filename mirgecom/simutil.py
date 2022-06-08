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
.. autofunction:: componentwise_norms
.. autofunction:: max_component_norm
.. autofunction:: check_naninf_local
.. autofunction:: check_range_local
.. autofunction:: boundary_report

Mesh utilities
--------------

.. autofunction:: generate_and_distribute_mesh

Simulation support utilities
----------------------------

.. autofunction:: limit_species_mass_fractions
.. autofunction:: species_fraction_anomaly_relaxation

Lazy eval utilities
-------------------

.. autofunction:: force_evaluation
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

from arraycontext import map_array_container, flatten

from functools import partial

from meshmode.dof_array import DOFArray

from typing import List
from grudge.discretization import DiscretizationCollection


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


def get_sim_timestep(discr, state, t, dt, cfl, t_final, constant_cfl=False):
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
    state: :class:`~mirgecom.gas_model.FluidState`
        The full fluid conserved and thermal state
    t: float
        Current time
    t_final: float
        Final time
    dt: float
        The current timestep
    cfl: float
        The current CFL number
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
                get_viscous_timestep(discr=discr, state=state)))[()]
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

    comm = discr.mpi_communicator
    rank = 0

    if comm:
        rank = comm.Get_rank()

    rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)

    if rank == 0:
        import os
        viz_dir = os.path.dirname(rank_fn)
        if viz_dir and not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

    if comm:
        comm.barrier()

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


def global_reduce(local_values, op, *, comm=None):
    """Perform a global reduction (allreduce if MPI comm is provided).

    This routine is a convenience wrapper for the MPI AllReduce operation
    that also works outside of an MPI context.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    local_values:
        The (:mod:`mpi4py`-compatible) value or array of values on which the
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
    if comm is not None:
        from mpi4py import MPI
        op_to_mpi_op = {
            "min": MPI.MIN,
            "max": MPI.MAX,
            "sum": MPI.SUM,
            "prod": MPI.PROD,
            "lor": MPI.LOR,
            "land": MPI.LAND,
        }
        return comm.allreduce(local_values, op=op_to_mpi_op[op])
    else:
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


def allsync(local_values, comm=None, op=None):
    """
    Perform allreduce if MPI comm is provided.

    Deprecated. Do not use in new code.
    """
    from warnings import warn
    warn("allsync is deprecated and will disappear in Q1 2022. "
         "Use global_reduce instead.", DeprecationWarning, stacklevel=2)

    if comm is None:
        return local_values

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

    return global_reduce(local_values, op_string, comm=comm)


def check_range_local(discr: DiscretizationCollection, dd: str, field: DOFArray,
                      min_value: float, max_value: float) -> List[float]:
    """Return the values that are outside the range [min_value, max_value]."""
    actx = field.array_context
    local_min = np.asscalar(actx.to_numpy(op.nodal_min_loc(discr, dd, field)))
    local_max = np.asscalar(actx.to_numpy(op.nodal_max_loc(discr, dd, field)))

    failing_values = []

    if local_min < min_value:
        failing_values.append(local_min)
    if local_max > max_value:
        failing_values.append(local_max)

    return failing_values


def check_naninf_local(discr: DiscretizationCollection, dd: str,
                       field: DOFArray) -> bool:
    """Return True if there are any NaNs or Infs in the field."""
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
    resid_errs = actx.to_numpy(
        flatten(componentwise_norms(discr, resid, order=np.inf), actx))

    return resid_errs.tolist()


def componentwise_norms(discr, fields, order=np.inf):
    """Return the *order*-norm for each component of *fields*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    if not isinstance(fields, DOFArray):
        return map_array_container(
            partial(componentwise_norms, discr, order=order), fields)
    return discr.norm(fields, order)


def max_component_norm(discr, fields, order=np.inf):
    """Return the max *order*-norm over the components of *fields*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    actx = fields.array_context
    return max(actx.to_numpy(flatten(
        componentwise_norms(discr, fields, order), actx)))


def generate_and_distribute_mesh(comm, generate_mesh):
    """Generate a mesh and distribute it among all ranks in *comm*.

    Generate the mesh with the user-supplied mesh generation function
    *generate_mesh*, partition the mesh, and distribute it to every
    rank in the provided MPI communicator *comm*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

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

    global_nelements = comm.bcast(global_nelements)
    return local_mesh, global_nelements


def boundary_report(discr, boundaries, outfile_name):
    """Generate a report of the grid boundaries."""
    comm = discr.mpi_communicator
    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.Get_size()
        rank = comm.Get_rank()

    local_header = f"nproc: {nproc}\nrank: {rank}\n"
    from io import StringIO
    local_report = StringIO(local_header)
    local_report.seek(0, 2)

    for btag in boundaries:
        boundary_discr = discr.discr_from_dd(btag)
        nnodes = sum([grp.ndofs for grp in boundary_discr.groups])
        local_report.write(f"{btag}: {nnodes}\n")

    if nproc > 1:
        from meshmode.mesh import BTAG_PARTITION
        from grudge.trace_pair import connected_ranks
        remote_ranks = connected_ranks(discr)
        local_report.write(f"remote_ranks: {remote_ranks}\n")
        rank_nodes = []
        for remote_rank in remote_ranks:
            boundary_discr = discr.discr_from_dd(BTAG_PARTITION(remote_rank))
            nnodes = sum([grp.ndofs for grp in boundary_discr.groups])
            rank_nodes.append(nnodes)
        local_report.write(f"nnodes_pb: {rank_nodes}\n")

    local_report.write("-----\n")
    local_report.seek(0)

    for irank in range(nproc):
        if irank == rank:
            f = open(outfile_name, "a+")
            f.write(local_report.read())
            f.close()
        if comm is not None:
            comm.barrier()


def create_parallel_grid(comm, generate_grid):
    """Generate and distribute mesh compatibility interface."""
    from warnings import warn
    warn("Do not call create_parallel_grid; use generate_and_distribute_mesh "
         "instead. This function will disappear August 1, 2021",
         DeprecationWarning, stacklevel=2)
    return generate_and_distribute_mesh(comm=comm, generate_mesh=generate_grid)


def limit_species_mass_fractions(cv):
    """Keep the species mass fractions from going negative."""
    from mirgecom.fluid import make_conserved
    if cv.nspecies > 0:
        y = cv.species_mass_fractions
        actx = cv.array_context
        new_y = 1.*y
        zero = 0 * y[0]
        one = zero + 1.

        for i in range(cv.nspecies):
            new_y[i] = actx.np.where(actx.np.less(new_y[i], 1e-14),
                                     zero, new_y[i])
            new_y[i] = actx.np.where(actx.np.greater(new_y[i], 1.),
                                     one, new_y[i])
        new_rho_y = cv.mass*new_y

        for i in range(cv.nspecies):
            new_rho_y[i] = actx.np.where(actx.np.less(new_rho_y[i], 1e-16),
                                         zero, new_rho_y[i])
            new_rho_y[i] = actx.np.where(actx.np.greater(new_rho_y[i], 1.),
                                         one, new_rho_y[i])

        return make_conserved(dim=cv.dim, mass=cv.mass,
                              momentum=cv.momentum, energy=cv.energy,
                              species_mass=new_rho_y)
    return cv


def species_fraction_anomaly_relaxation(cv, alpha=1.):
    """Pull negative species fractions back towards 0 with a RHS contribution."""
    from mirgecom.fluid import make_conserved
    if cv.nspecies > 0:
        y = cv.species_mass_fractions
        actx = cv.array_context
        new_y = 1.*y
        zero = 0. * y[0]
        for i in range(cv.nspecies):
            new_y[i] = actx.np.where(actx.np.less(new_y[i], 0.),
                                     -new_y[i], zero)
            # y_spec = actx.np.where(y_spec > 1., y_spec-1., zero)
        return make_conserved(dim=cv.dim, mass=0.*cv.mass,
                              momentum=0.*cv.momentum, energy=0.*cv.energy,
                              species_mass=alpha*cv.mass*new_y)
    return 0.*cv


def force_evaluation(actx, expn):
    """Wrap freeze/thaw forcing evaluation of expressions."""
    from arraycontext import thaw, freeze
    return thaw(freeze(expn, actx), actx)
