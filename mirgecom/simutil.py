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
                  step=0, t=0, overwrite=False, vis_timer=None,
                  mpi_communicator=None):
    """Write parallel VTK output for the fields specified in *io_fields*.

    This routine writes a parallel-compatible unstructured VTK visualization
    file set in (vtu/pvtu) format. One file per MPI rank is written with the
    following naming convention: *vizname*_*step*_<mpi-rank>.vtu, and a single
    file manifest with naming convention: *vizname*_*step*.pvtu.  Users are
    advised to visualize the data using _Paraview_, _VisIt_, or other
    VTU-compatible visualization software by opening the PVTU files.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    visualizer:
        A :class:`meshmode.discretization.visualization.Visualizer`
        VTK output object.
    io_fields:
        List of tuples indicating the (name, data) for each field to write.
    vizname: str
        Root part of the visualization file name to write
    step: int
        The step number to use in the file names
    t: float
        The simulation time to write into the visualization files
    overwrite: bool
        Option whether to overwrite existing files (True) or fail if files
        exist (False=default).
    mpi_communicator:
        An MPI Communicator is required for parallel writes. If no
        mpi_communicator is provided, then the write is assumed to be serial.
        (deprecated behavior: pull an MPI communicator from the discretization
         collection.  This will stop working in Fall 2022.)
    """
    from contextlib import nullcontext
    from mirgecom.io import make_rank_fname, make_par_fname

    if mpi_communicator is None:  # None is OK for serial writes!
        mpi_communicator = discr.mpi_communicator
        if mpi_communicator is not None:  # It's *not* OK to get comm from discr
            from warning import warn
            warn("Using `write_visfile` in parallel without an MPI communicator is "
                 "deprecated and will stop working in Fall 2022. For parallel "
                 "writes, specify an MPI communicator with the `mpi_communicator` "
                 "argument.")
    rank = 0

    if mpi_communicator is not None:
        rank = mpi_communicator.Get_rank()

    rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)

    if rank == 0:
        import os
        viz_dir = os.path.dirname(rank_fn)
        if viz_dir and not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

    if mpi_communicator is not None:
        mpi_communicator.barrier()

    if vis_timer:
        ctm = vis_timer.start_sub_timer()
    else:
        ctm = nullcontext()

    with ctm:
        visualizer.write_parallel_vtk_file(
            mpi_communicator, rank_fn, io_fields,
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
    local_min = actx.to_numpy(op.nodal_min_loc(discr, dd, field)).item()
    local_max = actx.to_numpy(op.nodal_max_loc(discr, dd, field)).item()

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
    if len(fields) > 0:
        return op.norm(discr, fields, order)
    else:
        # FIXME: This work-around for #575 can go away after #569
        return 0


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

    return local_mesh, global_nelements


def create_parallel_grid(comm, generate_grid):
    """Generate and distribute mesh compatibility interface."""
    from warnings import warn
    warn("Do not call create_parallel_grid; use generate_and_distribute_mesh "
         "instead. This function will disappear August 1, 2021",
         DeprecationWarning, stacklevel=2)
    return generate_and_distribute_mesh(comm=comm, generate_mesh=generate_grid)
