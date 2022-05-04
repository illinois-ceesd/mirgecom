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

.. autofunction:: distribute_mesh

Simulation support utilities
----------------------------

.. autofunction:: limit_species_mass_fractions
.. autofunction:: species_fraction_anomaly_relaxation
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
from grudge.dof_desc import DD_VOLUME_ALL


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


def get_sim_timestep(
        discr, state, t, dt, cfl, t_final, constant_cfl=False,
        fluid_volume_dd=DD_VOLUME_ALL):
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
                discr, fluid_volume_dd,
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
    from warnings import warn
    warn(
        "generate_and_distribute_mesh is deprecated and will go away Q4 2022. "
        "Use distribute_mesh instead.", DeprecationWarning, stacklevel=2)
    return distribute_mesh(comm, generate_mesh)


def distribute_mesh(comm, get_mesh_data, partition_generator_func=None):
    # FIXME: Out of date
    """Distribute a mesh among all ranks in *comm*.

    Retrieve the global mesh data with the user-supplied function *get_mesh_data*,
    partition the mesh, and distribute it to every rank in the provided MPI
    communicator *comm*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the mesh
    get_mesh_data:
        Callable of zero arguments returning *mesh* or *(mesh, volume_to_elements)*,
        where *mesh* is a :class:`meshmode.mesh.Mesh` and *volume_to_elements* is a
        :class:`dict` mapping volume tags to :class:`numpy.ndarray`s of element
        numbers.

    Returns
    -------
    local_mesh_data : :class:`meshmode.mesh.Mesh` or :class:`dict`
        If *get_mesh_data* returns only a mesh, *local_mesh_data* is the local mesh.
        If *get_mesh_data* also returns *volume_to_elements*, *local_mesh_data* will
        be a :class:`dict` mapping volume tags to corresponding local meshes.
    global_nelements : :class:`int`
        The number of elements in the global mesh
    """
    from meshmode.distributed import mpi_distribute

    num_ranks = comm.Get_size()

    if partition_generator_func is None:
        def partition_generator_func(mesh, tag_to_elements, num_ranks):
            from meshmode.distributed import get_partition_by_pymetis
            return get_partition_by_pymetis(mesh, num_ranks)

    if comm.Get_rank() == 0:
        global_data = get_mesh_data()

        from meshmode.mesh import Mesh
        if isinstance(global_data, Mesh):
            mesh = global_data
            tag_to_elements = None
            volume_to_tags = None
        elif isinstance(global_data, tuple) and len(global_data) == 3:
            mesh, tag_to_elements, volume_to_tags = global_data
        else:
            raise TypeError("Unexpected result from get_mesh_data")

        from meshmode.mesh.processing import partition_mesh

        rank_per_element = partition_generator_func(mesh, tag_to_elements, num_ranks)

        if tag_to_elements is None:
            rank_to_elements = {
                rank: np.where(rank_per_element == rank)[0]
                for rank in range(num_ranks)}

            rank_to_mesh_data = partition_mesh(mesh, rank_to_elements)

        else:
            tag_to_volume = {
                tag: vol
                for vol, tags in volume_to_tags.items()
                for tag in tags}

            volumes = list(volume_to_tags.keys())

            volume_index_per_element = np.full(mesh.nelements, -1, dtype=int)
            for tag, elements in tag_to_elements.items():
                volume_index_per_element[elements] = volumes.index(
                    tag_to_volume[tag])

            if np.any(volume_index_per_element < 0):
                raise ValueError("Missing volume specification for some elements.")

            part_id_to_elements = {
                (rank, volumes[vol_idx]):
                    np.where(
                        (volume_index_per_element == vol_idx)
                        & (rank_per_element == rank))[0]
                for vol_idx in range(len(volumes))
                for rank in range(num_ranks)}

            # FIXME: Find a better way to do this
            part_id_to_part_index = {
                part_id: part_index
                for part_id, part_index in zip(
                    part_id_to_elements.keys(),
                    range(len(part_id_to_elements)))}
            from meshmode.mesh.processing import _compute_global_elem_to_part_elem
            global_elem_to_part_elem = _compute_global_elem_to_part_elem(
                mesh.nelements, part_id_to_elements, part_id_to_part_index,
                mesh.element_id_dtype)

            tag_to_global_to_part = {
                tag: global_elem_to_part_elem[elements, :]
                for tag, elements in tag_to_elements.items()}

            part_id_to_tag_to_elements = {}
            for part_id in part_id_to_elements.keys():
                part_idx = part_id_to_part_index[part_id]
                part_tag_to_elements = {}
                for tag, global_to_part in tag_to_global_to_part.items():
                    part_tag_to_elements[tag] = global_to_part[
                        global_to_part[:, 0] == part_idx, 1]
                part_id_to_tag_to_elements[part_id] = part_tag_to_elements

            part_id_to_mesh = partition_mesh(mesh, part_id_to_elements)

            rank_to_mesh_data = {
                rank: {
                    vol: (
                        part_id_to_mesh[rank, vol],
                        part_id_to_tag_to_elements[rank, vol])
                    for vol in volumes}
                for rank in range(num_ranks)}

        local_mesh_data = mpi_distribute(
            comm, source_rank=0, source_data=rank_to_mesh_data)

        global_nelements = comm.bcast(mesh.nelements, root=0)

    else:
        local_mesh_data = mpi_distribute(comm, source_rank=0)

        global_nelements = comm.bcast(None, root=0)

    return local_mesh_data, global_nelements


def boundary_report(discr, boundaries, outfile_name, volume_dd=DD_VOLUME_ALL):
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

    for bdtag in boundaries:
        boundary_discr = discr.discr_from_dd(bdtag)
        nnodes = sum([grp.ndofs for grp in boundary_discr.groups])
        local_report.write(f"{bdtag}: {nnodes}\n")

    from meshmode.mesh import BTAG_PARTITION
    from meshmode.distributed import get_connected_partitions
    connected_part_ids = get_connected_partitions(
        discr.discr_from_dd(volume_dd).mesh)
    local_report.write(f"connected_part_ids: {connected_part_ids}\n")
    part_nodes = []
    for connected_part_id in connected_part_ids:
        boundary_discr = discr.discr_from_dd(BTAG_PARTITION(connected_part_id))
        nnodes = sum([grp.ndofs for grp in boundary_discr.groups])
        part_nodes.append(nnodes)
    if part_nodes:
        local_report.write(f"nnodes_pb: {part_nodes}\n")

    local_report.write("-----\n")
    local_report.seek(0)

    for irank in range(nproc):
        if irank == rank:
            f = open(outfile_name, "a+")
            f.write(local_report.read())
            f.close()
        if comm is not None:
            comm.barrier()


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
