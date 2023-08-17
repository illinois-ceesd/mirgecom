"""Provide some utilities for building simulation applications.

General utilities
-----------------

.. autofunction:: check_step
.. autofunction:: get_sim_timestep
.. autofunction:: write_visfile
.. autofunction:: global_reduce
.. autofunction:: get_reasonable_memory_pool

Diagnostic utilities
--------------------

.. autofunction:: compare_fluid_solutions
.. autofunction:: componentwise_norms
.. autofunction:: max_component_norm
.. autofunction:: check_naninf_local
.. autofunction:: check_range_local
.. autofunction:: boundary_report

Mesh and element utilities
--------------------------

.. autofunction:: geometric_mesh_partitioner
.. autofunction:: distribute_mesh
.. autofunction:: get_number_of_tetrahedron_nodes
.. autofunction:: get_box_mesh

Simulation support utilities
----------------------------

.. autofunction:: configurate

File comparison utilities
-------------------------

.. autofunction:: compare_files_vtu
.. autofunction:: compare_files_xdmf
.. autofunction:: compare_files_hdf5
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
import sys
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional

import grudge.op as op
import numpy as np
import pyopencl as cl
from arraycontext import flatten, map_array_container
from grudge.discretization import DiscretizationCollection, PartID
from grudge.dof_desc import DD_VOLUME_ALL
from meshmode.dof_array import DOFArray

from mirgecom.utils import normalize_boundaries
from mirgecom.viscous import get_viscous_timestep

logger = logging.getLogger(__name__)

if TYPE_CHECKING or getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    # pylint: disable=no-name-in-module
    from mpi4py.MPI import Comm


def get_number_of_tetrahedron_nodes(dim, order, include_faces=False):
    """Get number of nodes (modes) in *dim* Tetrahedron of *order*."""
    # number of {nodes, modes} see e.g.:
    # JSH/TW Nodal DG Methods, Section 10.1
    # DOI: 10.1007/978-0-387-72067-8
    nnodes = int(np.math.factorial(dim+order)
                 / (np.math.factorial(dim) * np.math.factorial(order)))
    if include_faces:
        nnodes = nnodes + (dim+1)*get_number_of_tetrahedron_nodes(dim-1, order)
    return nnodes


def get_box_mesh(dim, a, b, n, t=None, periodic=None):
    """
    Create a rectangular "box" like mesh with tagged boundary faces.

    The resulting mesh has boundary tags
    `"-i"` and `"+i"` for `i=1,...,dim`
    corresponding to lower and upper faces normal to coordinate dimension `i`.

    Parameters
    ----------
    dim: int
        The mesh topological dimension
    a: float or tuple
        The coordinates of the lower corner of the box. If scalar-valued, gets
        promoted to a uniform tuple.
    b: float or tuple
        The coordinates of the upper corner of the box. If scalar-valued, gets
        promoted to a uniform tuple.
    n: int or tuple
        The number of elements along a given dimension. If scalar-valued, gets
        promoted to a uniform tuple.
    t: str or None
        The mesh type. See
        :func:`meshmode.mesh.generation.generate_box_mesh` for details.
    periodic: bool or tuple or None
        Indicates whether the mesh is periodic in a given dimension. If
        scalar-valued, gets promoted to a uniform tuple.

    Returns
    -------
    :class:`meshmode.mesh.Mesh`
        The generated box mesh.
    """
    if np.isscalar(a):
        a = (a,)*dim
    if np.isscalar(b):
        b = (b,)*dim
    if np.isscalar(n):
        n = (n,)*dim
    if periodic is None:
        periodic = (False,)*dim
    elif np.isscalar(periodic):
        periodic = (periodic,)*dim

    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]

    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, nelements_per_axis=n,
               boundary_tag_to_face=bttf,
               mesh_type=t, periodic=periodic)


def check_step(step, interval):
    """
    Check step number against a user-specified interval.

    Utility is used typically for visualization.

    - Negative numbers mean 'never visualize'.
    - Zero means 'always visualize'.

    Useful for checking whether the current step is an output step,
    or anything else that occurs on fixed intervals.
    """
    if interval == 0:
        return True
    elif interval < 0:
        return False
    elif step % interval == 0:
        return True
    return False


def get_sim_timestep(
        dcoll, state, t, dt, cfl, t_final=0.0, constant_cfl=False,
        local_dt=False, fluid_dd=DD_VOLUME_ALL):
    r"""Return the maximum stable timestep for a typical fluid simulation.

    This routine returns a constraint-limited timestep size for a fluid
    simulation.  The returned timestep will be constrained by the specified
    Courant-Friedrichs-Lewy number, *cfl*, and the simulation max simulated time
    limit, *t_final*, and subject to the user's optional settings.

    The local fluid timestep, $\delta{t}_l$, is computed by
    :func:`~mirgecom.viscous.get_viscous_timestep`.  Users are referred to that
    routine for the details of the local timestep.

    With the remaining simulation time $\Delta{t}_r =
    \left(\mathit{t\_final}-\mathit{t}\right)$, three modes are supported
    for the returned timestep, $\delta{t}$:

    - "Constant DT" mode (default): $\delta{t} = \mathbf{\text{min}}
      \left(\textit{dt},~\Delta{t}_r\right)$
    - "Constant CFL" mode (constant_cfl=True): $\delta{t} =
      \mathbf{\text{min}}\left(\mathbf{\text{global\_min}}\left(\delta{t}\_l\right)
      ,~\Delta{t}_r\right)$
    - "Local DT" mode (local_dt=True): $\delta{t} = \mathbf{\text{cell\_local\_min}}
      \left(\delta{t}_l\right)$

    Note that for "Local DT" mode, *t_final* is ignored, and a
    :class:`~meshmode.dof_array.DOFArray` containing the local *cfl*-limited
    timestep, where $\mathbf{\text{cell\_local\_min}}\left(\delta{t}\_l\right)$ is
    defined as the minimum over the cell collocation points. This mode is useful for
    stepping to convergence of steady-state solutions.

    .. important::
        For "Constant CFL" mode, this routine calls the collective
        :func:`~grudge.op.nodal_min` on the inside which involves MPI collective
        functions.  Thus all MPI ranks on the
        :class:`~grudge.discretization.DiscretizationCollection` must call this
        routine collectively when using "Constant CFL" mode.

    Parameters
    ----------
    dcoll: :class:`~grudge.discretization.DiscretizationCollection`
        The DG discretization collection to use
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
    local_dt: bool
        True if running local DT mode. False by default.
    fluid_dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *state* lives. Must be a
        volume on the base discretization.

    Returns
    -------
    float or :class:`~meshmode.dof_array.DOFArray`
        The global maximum stable DT based on a viscous fluid.
    """
    if local_dt:
        actx = state.array_context
        data_shape = (state.cv.mass[0]).shape
        if actx.supports_nonscalar_broadcasting:
            return cfl * actx.np.broadcast_to(
                op.elementwise_min(
                    dcoll, fluid_dd,
                    get_viscous_timestep(dcoll, state, dd=fluid_dd)),
                data_shape)
        else:
            return cfl * op.elementwise_min(
                dcoll, fluid_dd, get_viscous_timestep(dcoll, state, dd=fluid_dd))

    my_dt = dt
    t_remaining = max(0, t_final - t)
    if constant_cfl:
        my_dt = state.array_context.to_numpy(
            cfl * op.nodal_min(
                dcoll, fluid_dd,
                get_viscous_timestep(dcoll=dcoll, state=state, dd=fluid_dd)))[()]

    return min(t_remaining, my_dt)


def write_visfile(dcoll, io_fields, visualizer, vizname,
                  step=0, t=0, overwrite=False, vis_timer=None,
                  comm: Optional["Comm"] = None):
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
    comm:
        An MPI Communicator is required for parallel writes. If no
        mpi_communicator is provided, then the write is assumed to be serial.
        (deprecated behavior: pull an MPI communicator from the discretization
        collection.  This will stop working in Fall 2022.)
    """
    from contextlib import nullcontext

    from mirgecom.io import make_par_fname, make_rank_fname

    if comm is None:  # None is OK for serial writes!
        comm = dcoll.mpi_communicator
        if comm is not None:  # It's *not* OK to get comm from dcoll
            from warnings import warn
            warn("Using `write_visfile` in parallel without an MPI communicator is "
                 "deprecated and will stop working in Fall 2022. For parallel "
                 "writes, specify an MPI communicator with the `mpi_communicator` "
                 "argument.")
    rank = 0

    if comm is not None:
        rank = comm.Get_rank()

    rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)

    if rank == 0:
        import os
        viz_dir = os.path.dirname(rank_fn)
        if viz_dir and not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

    if comm is not None:
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


def check_range_local(dcoll: DiscretizationCollection, dd: str, field: DOFArray,
                      min_value: float, max_value: float) -> List[float]:
    """Return the values that are outside the range [min_value, max_value]."""
    actx = field.array_context
    local_min = actx.to_numpy(op.nodal_min_loc(dcoll, dd, field)).item()
    local_max = actx.to_numpy(op.nodal_max_loc(dcoll, dd, field)).item()

    failing_values = []

    if local_min < min_value:
        failing_values.append(local_min)
    if local_max > max_value:
        failing_values.append(local_max)

    return failing_values


def check_naninf_local(dcoll: DiscretizationCollection, dd: str,
                       field: DOFArray) -> bool:
    """Return True if there are any NaNs or Infs in the field."""
    actx = field.array_context
    s = actx.to_numpy(op.nodal_sum_loc(dcoll, dd, field))
    return not np.isfinite(s)


def compare_fluid_solutions(dcoll, red_state, blue_state, *, dd=DD_VOLUME_ALL):
    """Return inf norm of (*red_state* - *blue_state*) for each component.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    actx = red_state.array_context
    resid = red_state - blue_state
    resid_errs = actx.to_numpy(
        flatten(componentwise_norms(dcoll, resid, order=np.inf, dd=dd), actx))

    return resid_errs.tolist()


def componentwise_norms(dcoll, fields, order=np.inf, *, dd=DD_VOLUME_ALL):
    """Return the *order*-norm for each component of *fields*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    if not isinstance(fields, DOFArray):
        return map_array_container(
            partial(componentwise_norms, dcoll, order=order, dd=dd), fields)
    if len(fields) > 0:
        return op.norm(dcoll, fields, order, dd=dd)
    else:
        # FIXME: This work-around for #575 can go away after #569
        return 0


def max_component_norm(dcoll, fields, order=np.inf, *, dd=DD_VOLUME_ALL):
    """Return the max *order*-norm over the components of *fields*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    actx = fields.array_context
    return max(actx.to_numpy(flatten(
        componentwise_norms(dcoll, fields, order, dd=dd), actx)))


class PartitioningError(Exception):
    """Error tossed to indicate an error with domain decomposition."""

    pass


def geometric_mesh_partitioner(mesh, num_ranks=None, *, nranks_per_axis=None,
                               auto_balance=False, imbalance_tolerance=.01,
                               debug=False):
    """Partition a mesh uniformly along the X coordinate axis.

    The intent is to partition the mesh uniformly along user-specified
    directions. In this current interation, the capability is demonstrated
    by splitting along the X axis.

    Parameters
    ----------
    mesh: :class:`meshmode.mesh.Mesh`
        The serial mesh to partition
    num_ranks: int
        The number of partitions to make (deprecated)
    nranks_per_axis: numpy.ndarray
        How many partitions per specified axis.
    auto_balance: bool
        Indicates whether to perform automatic balancing.  If true, the
        partitioner will try to balance the number of elements over
        the partitions.
    imbalance_tolerance: float
        If *auto_balance* is True, this parameter indicates the acceptable
        relative difference to the average number of elements per partition.
        It defaults to balance within 1%.
    debug: bool
        En/disable debugging/diagnostic print reporting.

    Returns
    -------
    elem_to_rank: numpy.ndarray
        Array indicating the MPI rank for each element
    """
    mesh_dimension = mesh.dim
    if nranks_per_axis is None or num_ranks is not None:
        from warnings import warn
        warn("num_ranks is deprecated, use nranks_per_axis instead.")
        num_ranks = num_ranks or 1
        nranks_per_axis = np.ones(mesh_dimension, dtype=np.int32)
        nranks_per_axis[0] = num_ranks
    if len(nranks_per_axis) != mesh_dimension:
        raise ValueError("nranks_per_axis must match mesh dimension.")
    num_ranks = np.prod(nranks_per_axis)
    if np.prod(nranks_per_axis[1:]) != 1:
        raise NotImplementedError("geometric_mesh_partitioner currently only "
                                "supports partitioning in the X-dimension."
                                "(only nranks_per_axis[0] should be > 1).")
    mesh_verts = mesh.vertices
    mesh_x = mesh_verts[0]

    x_min = np.min(mesh_x)
    x_max = np.max(mesh_x)
    x_interval = x_max - x_min
    part_loc = np.linspace(x_min, x_max, num_ranks+1)

    part_interval = x_interval / nranks_per_axis[0]

    all_elem_group_centroids = []
    for group in mesh.groups:
        elem_group_x = mesh_verts[0, group.vertex_indices]
        elem_group_centroids = np.sum(elem_group_x, axis=1)/elem_group_x.shape[1]
        all_elem_group_centroids.append(elem_group_centroids)
    elem_centroids = np.concatenate(all_elem_group_centroids)
    global_nelements = len(elem_centroids)

    aver_part_nelem = global_nelements / num_ranks

    if debug:
        print(f"Partitioning {global_nelements} elements in"
              f" [{x_min},{x_max}]/{num_ranks}")
        print(f"Average nelem/part: {aver_part_nelem}")
        print(f"Initial part locs: {part_loc=}")

    # Create geometrically even partitions
    elem_to_rank = ((elem_centroids-x_min) / part_interval).astype(int)

    print(f"{elem_to_rank=}")

    # map partition id to list of elements in that partition
    part_to_elements = {r: set(np.where(elem_to_rank == r)[0])
                        for r in range(num_ranks)}
    # make an array of the geometrically even partition sizes
    # avoids calling "len" over and over on the element sets
    nelem_part = [len(part_to_elements[r]) for r in range(num_ranks)]

    if debug:
        print(f"Initial: {nelem_part=}")

    # Automatic load-balancing
    if auto_balance:

        for r in range(num_ranks-1):

            # find the element reservoir (next part with elements in it)
            # adv_part = r + 1
            # while nelem_part[adv_part] == 0:
            #    adv_part = adv_part + 1

            num_elem_needed = aver_part_nelem - nelem_part[r]
            part_imbalance = np.abs(num_elem_needed) / float(aver_part_nelem)

            if debug:
                print(f"Processing part({r=})")
                print(f"{part_loc[r]=}")
                print(f"{num_elem_needed=}, {part_imbalance=}")
                print(f"{nelem_part=}")

            niter = 0
            total_change = 0
            moved_elements = set()

            adv_part = r + 1
            # while ((part_imbalance > imbalance_tolerance)
            #       and (adv_part < num_ranks)):
            while part_imbalance > imbalance_tolerance:
                # This partition needs to keep changing in size until it meets the
                # specified imbalance tolerance, or gives up trying

                # seek out the element reservoir
                while nelem_part[adv_part] == 0:
                    adv_part = adv_part + 1
                    if adv_part >= num_ranks:
                        raise PartitioningError("Ran out of elements to partition.")

                if debug:
                    print(f"-{nelem_part[r]=}, adv_part({adv_part}),"
                          f" {nelem_part[adv_part]=}")
                    print(f"-{part_loc[r+1]=},{part_loc[adv_part+1]=}")
                    print(f"-{num_elem_needed=},{part_imbalance=}")

                if niter > 100:
                    raise PartitioningError("Detected too many iterations in"
                                            " partitioning.")

                # The purpose of the next block is to populate the "moved_elements"
                # data structure. Then those elements will be moved between the
                # current partition being processed and the "reservoir,"
                # *and* to adjust the position of the "right" side of the current
                # partition boundary.
                moved_elements = set()
                num_elements_added = 0

                if num_elem_needed > 0:

                    # Partition is SMALLER than it should be, grab elements from
                    # the reservoir
                    if debug:
                        print(f"-Grabbing elements from reservoir({adv_part})"
                              f", {nelem_part[adv_part]=}")

                    portion_needed = (float(abs(num_elem_needed))
                                      / float(nelem_part[adv_part]))
                    portion_needed = min(portion_needed, 1.0)

                    if debug:
                        print(f"--Chomping {portion_needed*100}% of"
                              f" reservoir({adv_part}) [by nelem].")

                    if portion_needed == 1.0:  # Chomp
                        new_loc = part_loc[adv_part+1]
                        moved_elements.update(part_to_elements[adv_part])

                    else:  # Bite
                        # This is the spatial size of the reservoir
                        reserv_interval = part_loc[adv_part+1] - part_loc[r+1]

                        # Find what portion of the reservoir to grab spatially
                        # This part is needed because the elements are not
                        # distributed uniformly in space.
                        fine_tuned = False
                        trial_portion_needed = portion_needed
                        while not fine_tuned:
                            pos_update = trial_portion_needed*reserv_interval
                            new_loc = part_loc[r+1] + pos_update

                            moved_elements = set()
                            num_elem_mv = 0
                            for e in part_to_elements[adv_part]:
                                if elem_centroids[e] <= new_loc:
                                    moved_elements.add(e)
                                    num_elem_mv = num_elem_mv + 1
                            if num_elem_mv < num_elem_needed:
                                fine_tuned = True
                            else:
                                ovrsht = (num_elem_mv - num_elem_needed)
                                rel_ovrsht = ovrsht/float(num_elem_needed)
                                if rel_ovrsht > 0.8:
                                    # bisect the space grabbed and try again
                                    trial_portion_needed = trial_portion_needed/2.0
                                else:
                                    fine_tuned = True

                        portion_needed = trial_portion_needed
                        new_loc = part_loc[r+1] + pos_update
                        if debug:
                            print(f"--Tuned: {portion_needed=} [by spatial volume]")
                            print(f"--Advancing part({r}) by +{pos_update}")

                    num_elements_added = len(moved_elements)
                    if debug:
                        print(f"--Adding {num_elements_added} to part({r}).")

                else:

                    # Partition is LARGER than it should be
                    # Grab the spatial size of the current partition
                    # to estimate the portion we need to shave off
                    # assuming uniform element density
                    part_interval = part_loc[r+1] - part_loc[r]
                    num_to_move = -num_elem_needed
                    portion_needed = num_to_move/float(nelem_part[r])

                    if debug:
                        print(f"--Shaving off {portion_needed*100}% of"
                              f" partition({r}) [by nelem].")

                    # Tune the shaved portion to account for
                    # non-uniform element density
                    fine_tuned = False
                    while not fine_tuned:
                        pos_update = portion_needed*part_interval
                        new_pos = part_loc[r+1] - pos_update
                        moved_elements = set()
                        num_elem_mv = 0
                        for e in part_to_elements[r]:
                            if elem_centroids[e] > new_pos:
                                moved_elements.add(e)
                                num_elem_mv = num_elem_mv + 1
                        if num_elem_mv < num_to_move:
                            fine_tuned = True
                        else:
                            ovrsht = (num_elem_mv - num_to_move)
                            rel_ovrsht = ovrsht/float(num_to_move)
                            if rel_ovrsht > 0.8:
                                # bisect and try again
                                portion_needed = portion_needed/2.0
                            else:
                                fine_tuned = True

                    # new "right" wall location of shranken part
                    # and negative num_elements_added for removal
                    new_loc = new_pos
                    num_elements_added = -len(moved_elements)
                    if debug:
                        print(f"--Reducing partition size by {portion_needed*100}%"
                              " [by nelem].")
                        print(f"--Removing {-num_elements_added} from part({r}).")

                # Now "moved_elements", "num_elements_added", and "new_loc"
                # are computed.  Update the partition, and reservoir.
                if debug:
                    print(f"--Number of elements to ADD: {num_elements_added}.")

                if num_elements_added > 0:
                    part_to_elements[r].update(moved_elements)
                    part_to_elements[adv_part].difference_update(
                        moved_elements)
                    for e in moved_elements:
                        elem_to_rank[e] = r
                else:
                    part_to_elements[r].difference_update(moved_elements)
                    part_to_elements[adv_part].update(moved_elements)
                    for e in moved_elements:
                        elem_to_rank[e] = adv_part

                total_change = total_change + num_elements_added
                part_loc[r+1] = new_loc
                if debug:
                    print(f"--Before: {nelem_part=}")
                nelem_part[r] = nelem_part[r] + num_elements_added
                nelem_part[adv_part] = nelem_part[adv_part] - num_elements_added
                if debug:
                    print(f"--After: {nelem_part=}")

                # Compute new nelem_needed and part_imbalance
                num_elem_needed = num_elem_needed - num_elements_added
                part_imbalance = \
                    np.abs(num_elem_needed) / float(aver_part_nelem)
                niter = niter + 1

            # Summarize the total change and state of the partition
            # and reservoir
            if debug:
                print(f"-Part({r}): {total_change=}")
                print(f"-Part({r=}): {nelem_part[r]=}, {part_imbalance=}")
                print(f"-Part({adv_part}): {nelem_part[adv_part]=}")

    # Validate the partitioning before returning
    total_partitioned_elements = sum([len(part_to_elements[r])
                                      for r in range(num_ranks)])
    total_nelem_part = sum([nelem_part[r] for r in range(num_ranks)])

    if debug:
        print("Validating mesh parts.")

    if total_partitioned_elements != total_nelem_part:
        raise PartitioningError("Validator: parted element counts dont match")
    if total_partitioned_elements != global_nelements:
        raise PartitioningError("Validator: global element counts dont match.")
    if len(elem_to_rank) != global_nelements:
        raise PartitioningError("Validator: elem-to-rank wrong size.")
    if np.any(nelem_part) <= 0:
        raise PartitioningError("Validator: empty partitions.")

    for e in range(global_nelements):
        part = elem_to_rank[e]
        if e not in part_to_elements[part]:
            raise PartitioningError("Validator: part/element/part map mismatch.")

    part_counts = np.zeros(global_nelements)
    for part_elements in part_to_elements.values():
        for element in part_elements:
            part_counts[element] = part_counts[element] + 1

    if np.any(part_counts > 1):
        raise PartitioningError("Validator: degenerate elements")
    if np.any(part_counts < 1):
        raise PartitioningError("Validator: orphaned elements")

    return elem_to_rank


def generate_and_distribute_mesh(comm, generate_mesh, **kwargs):
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
    r"""Distribute a mesh among all ranks in *comm*.

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
        Callable of zero arguments returning *mesh* or
        *(mesh, tag_to_elements, volume_to_tags)*, where *mesh* is a
        :class:`meshmode.mesh.Mesh`, *tag_to_elements* is a
        :class:`dict` mapping mesh volume tags to :class:`numpy.ndarray`\ s of
        element numbers, and *volume_to_tags* is a :class:`dict` that maps volumes
        in the resulting distributed mesh to volume tags in *tag_to_elements*.
    partition_generator_func:
        Optional callable that takes *mesh*, *tag_to_elements*, and *comm*'s size,
        and returns a :class:`numpy.ndarray` indicating to which rank each element
        belongs.

    Returns
    -------
    local_mesh_data: :class:`meshmode.mesh.Mesh` or :class:`dict`
        If the result of calling *get_mesh_data* specifies a single volume,
        *local_mesh_data* is the local mesh.  If it specifies multiple volumes,
        *local_mesh_data* will be a :class:`dict` mapping volume tags to
        tuples of the form *(local_mesh, local_tag_to_elements)*.
    global_nelements: :class:`int`
        The number of elements in the global mesh
    """
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

            rank_to_mesh_data_dict = partition_mesh(mesh, rank_to_elements)

            rank_to_mesh_data = [
                rank_to_mesh_data_dict[rank]
                for rank in range(num_ranks)]

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
                PartID(volumes[vol_idx], rank):
                    np.where(
                        (volume_index_per_element == vol_idx)
                        & (rank_per_element == rank))[0]
                for vol_idx in range(len(volumes))
                for rank in range(num_ranks)}

            # TODO: Add a public function to meshmode to accomplish this? So we're
            # not depending on meshmode internals
            part_id_to_part_index = {
                part_id: part_index
                for part_index, part_id in enumerate(part_id_to_elements.keys())}
            from meshmode.mesh.processing import \
                _compute_global_elem_to_part_elem
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

            rank_to_mesh_data = [
                {
                    vol: (
                        part_id_to_mesh[PartID(vol, rank)],
                        part_id_to_tag_to_elements[PartID(vol, rank)])
                    for vol in volumes}
                for rank in range(num_ranks)]

        local_mesh_data = comm.scatter(rank_to_mesh_data, root=0)

        global_nelements = comm.bcast(mesh.nelements, root=0)

    else:
        local_mesh_data = comm.scatter(None, root=0)

        global_nelements = comm.bcast(None, root=0)

    return local_mesh_data, global_nelements


def extract_volumes(mesh, tag_to_elements, selected_tags, boundary_tag):
    r"""
    Create a mesh containing a subset of another mesh's volumes.

    Parameters
    ----------
    mesh: :class:`meshmode.mesh.Mesh`
        The original mesh.
    tag_to_elements:
        A :class:`dict` mapping mesh volume tags to :class:`numpy.ndarray`\ s
        of element numbers in *mesh*.
    selected_tags:
        A sequence of tags in *tag_to_elements* representing the subset of volumes
        to be included.
    boundary_tag:
        Tag to assign to the boundary that was previously the interface between
        included/excluded volumes.

    Returns
    -------
    in_mesh: :class:`meshmode.mesh.Mesh`
        The resulting mesh.
    tag_to_in_elements:
        A :class:`dict` mapping the tags from *selected_tags* to
        :class:`numpy.ndarray`\ s of element numbers in *in_mesh*.
    """
    is_in_element = np.full(mesh.nelements, False)
    for tag, elements in tag_to_elements.items():
        if tag in selected_tags:
            is_in_element[elements] = True

    from meshmode.mesh.processing import partition_mesh
    in_mesh = partition_mesh(mesh, {
        "_in": np.where(is_in_element)[0],
        "_out": np.where(~is_in_element)[0]})["_in"]

    # partition_mesh creates a partition boundary for "_out"; replace with a
    # normal boundary
    new_facial_adjacency_groups = []
    from meshmode.mesh import BoundaryAdjacencyGroup, InterPartAdjacencyGroup
    for grp_list in in_mesh.facial_adjacency_groups:
        new_grp_list = []
        for fagrp in grp_list:
            if (
                    isinstance(fagrp, InterPartAdjacencyGroup)
                    and fagrp.part_id == "_out"):
                new_fagrp = BoundaryAdjacencyGroup(
                    igroup=fagrp.igroup,
                    boundary_tag=boundary_tag,
                    elements=fagrp.elements,
                    element_faces=fagrp.element_faces)
            else:
                new_fagrp = fagrp
            new_grp_list.append(new_fagrp)
        new_facial_adjacency_groups.append(new_grp_list)
    in_mesh = in_mesh.copy(facial_adjacency_groups=new_facial_adjacency_groups)

    element_to_in_element = np.where(
        is_in_element,
        np.cumsum(is_in_element) - 1,
        np.full(mesh.nelements, -1))

    tag_to_in_elements = {
        tag: element_to_in_element[tag_to_elements[tag]]
        for tag in selected_tags}

    return in_mesh, tag_to_in_elements


def boundary_report(dcoll, boundaries, outfile_name, *, dd=DD_VOLUME_ALL,
                    mesh=None):
    """Generate a report of the grid boundaries."""
    boundaries = normalize_boundaries(boundaries)

    comm = dcoll.mpi_communicator
    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.Get_size()
        rank = comm.Get_rank()

    if mesh is not None:
        nelem = 0
        for grp in mesh.groups:
            nelem = nelem + grp.nelements
        local_header = f"nproc: {nproc}\nrank: {rank}\nnelem: {nelem}\n"
    else:
        local_header = f"nproc: {nproc}\nrank: {rank}\n"

    from io import StringIO
    local_report = StringIO(local_header)
    local_report.seek(0, 2)

    for bdtag in boundaries:
        boundary_discr = dcoll.discr_from_dd(bdtag)
        nnodes = sum([grp.ndofs for grp in boundary_discr.groups])
        local_report.write(f"{bdtag}: {nnodes}\n")

    from meshmode.distributed import get_connected_parts
    from meshmode.mesh import BTAG_PARTITION
    connected_part_ids = get_connected_parts(dcoll.discr_from_dd(dd).mesh)
    local_report.write(f"num_nbr_parts: {len(connected_part_ids)}\n")
    local_report.write(f"connected_part_ids: {connected_part_ids}\n")
    part_nodes = []
    for connected_part_id in connected_part_ids:
        boundary_discr = dcoll.discr_from_dd(
            dd.trace(BTAG_PARTITION(connected_part_id)))
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


def force_evaluation(actx, expn):
    """Wrap freeze/thaw forcing evaluation of expressions.

    Deprecated; use :func:`mirgecom.utils.force_evaluation` instead.
    """
    from warnings import warn
    warn("simutil.force_evaluation is deprecated and will disappear in Q3 2023. "
         "Use utils.force_evaluation instead.", DeprecationWarning, stacklevel=2)
    return actx.thaw(actx.freeze(expn))


def get_reasonable_memory_pool(ctx: cl.Context, queue: cl.CommandQueue,
                               force_buffer: bool = False,
                               force_non_pool: bool = False):
    """Return an SVM or buffer memory pool based on what the device supports.

    By default, it prefers SVM allocations over CL buffers, and memory
    pools over direct allocations.
    """
    import pyopencl.tools as cl_tools
    from pyopencl.characterize import has_coarse_grain_buffer_svm

    if force_buffer and force_non_pool:
        logger.info(f"Using non-pooled CL buffer allocations on {queue.device}.")
        return cl_tools.DeferredAllocator(ctx)

    if force_buffer:
        logger.info(f"Using pooled CL buffer allocations on {queue.device}.")
        return cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))

    if force_non_pool and has_coarse_grain_buffer_svm(queue.device):
        logger.info(f"Using non-pooled SVM allocations on {queue.device}.")
        return cl_tools.SVMAllocator(  # pylint: disable=no-member
            ctx, alignment=0, queue=queue)

    if has_coarse_grain_buffer_svm(queue.device) and hasattr(cl_tools, "SVMPool"):
        logger.info(f"Using SVM-based memory pool on {queue.device}.")
        return cl_tools.SVMPool(cl_tools.SVMAllocator(  # pylint: disable=no-member
            ctx, alignment=0, queue=queue))
    else:
        from warnings import warn
        if not has_coarse_grain_buffer_svm(queue.device):
            warn(f"No SVM support on {queue.device}, returning a CL buffer-based "
                  "memory pool. If you are running with PoCL-cuda, please update "
                  "your PoCL installation.")
        else:
            warn("No SVM memory pool support with your version of PyOpenCL, "
                 f"returning a CL buffer-based memory pool on {queue.device}. "
                 "Please update your PyOpenCL version.")
        return cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))


def configurate(config_key, config_object=None, default_value=None):
    """Return a configured item from a configuration object."""
    if config_object is not None:
        d = config_object if isinstance(config_object, dict) else\
            config_object.__dict__
        if config_key in d:
            value = d[config_key]
            if default_value is not None:
                return type(default_value)(value)
            return value
    return default_value


def compare_files_vtu(
        first_file: str,
        second_file: str,
        file_type: str,
        tolerance: float = 1e-12,
        field_tolerance: Optional[Dict[str, float]] = None
        ) -> None:
    """Compare files of vtu type.

    Parameters
    ----------
    first_file:
        First file to compare
    second_file:
        Second file to compare
    file_type:
        Vtu files
    tolerance:
        Max acceptable absolute difference
    field_tolerance:
        Dictionary of individual field tolerances

    Returns
    -------
    True:
        If it passes the files contain data within the given tolerance.
    False:
        If it fails the files contain data outside the given tolerance.
    """
    import xml.etree.ElementTree as Et

    import vtk

    # read files:
    if file_type == "vtu":
        reader1 = vtk.vtkXMLUnstructuredGridReader()  # pylint: disable=no-member
        reader2 = vtk.vtkXMLUnstructuredGridReader()  # pylint: disable=no-member
    else:
        reader1 = vtk.vtkXMLPUnstructuredGridReader()  # pylint: disable=no-member
        reader2 = vtk.vtkXMLPUnstructuredGridReader()  # pylint: disable=no-member

    reader1.SetFileName(first_file)
    reader1.Update()
    output1 = reader1.GetOutput()

    # Check rank number
    def numranks(filename: str) -> int:
        tree = Et.parse(filename)
        root = tree.getroot()
        return len(root.findall(".//Piece"))

    if file_type == "pvtu":
        rank1 = numranks(first_file)
        rank2 = numranks(second_file)
        if rank1 != rank2:
            raise ValueError(f"File '{first_file}' has {rank1} ranks, "
                f"but file '{second_file}' has {rank2} ranks.")

    reader2.SetFileName(second_file)
    reader2.Update()
    output2 = reader2.GetOutput()

    # check fidelity
    point_data1 = output1.GetPointData()
    point_data2 = output2.GetPointData()

    # verify same number of PointData arrays in both files
    if point_data1.GetNumberOfArrays() != point_data2.GetNumberOfArrays():
        print("File 1:", point_data1.GetNumberOfArrays(), "\n",
              "File 2:", point_data2.GetNumberOfArrays())
        raise ValueError("Fidelity test failed: Mismatched data array count")

    nfields = point_data1.GetNumberOfArrays()
    max_field_errors = [0 for _ in range(nfields)]

    if field_tolerance is None:
        field_tolerance = {}
    field_specific_tols = [configurate(point_data1.GetArrayName(i),
        field_tolerance, tolerance) for i in range(nfields)]

    for i in range(nfields):
        arr1 = point_data1.GetArray(i)
        arr2 = point_data2.GetArray(i)

        # verify both files contain same arrays
        if point_data1.GetArrayName(i) != point_data2.GetArrayName(i):
            print("File 1:", point_data1.GetArrayName(i), "\n",
                  "File 2:", point_data2.GetArrayName(i))
            raise ValueError("Fidelity test failed: Mismatched data array names")

        # verify arrays are same sizes in both files
        if arr1.GetSize() != arr2.GetSize():
            print("File 1, DataArray", i, ":", arr1.GetSize(), "\n",
                  "File 2, DataArray", i, ":", arr2.GetSize())
            raise ValueError("Fidelity test failed: Mismatched data array sizes")

        # verify individual values w/in given tolerance
        fieldname = point_data1.GetArrayName(i)
        print(f"Field: {fieldname}", end=" ")
        for j in range(arr1.GetSize()):
            test_err = abs(arr1.GetValue(j) - arr2.GetValue(j))
            if test_err > max_field_errors[i]:
                max_field_errors[i] = test_err
        print(f"Max Error: {max_field_errors[i]}", end=" ")
        print(f"Tolerance: {field_specific_tols[i]}")

    violated_tols = []
    for i in range(nfields):
        if max_field_errors[i] > field_specific_tols[i]:
            violated_tols.append(field_specific_tols[i])

    if violated_tols:
        raise ValueError("Fidelity test failed: Mismatched data array "
                                 f"values {violated_tols=}.")

    print("VTU Fidelity test completed successfully")


class _Hdf5Reader:
    def __init__(self, filename):
        import h5py

        self.file_obj = h5py.File(filename, "r")

    def read_specific_data(self, datapath):
        return self.file_obj[datapath]


class _XdmfReader:
    # CURRENTLY DOES NOT SUPPORT MULTIPLE Grids

    def __init__(self, filename):
        from xml.etree import ElementTree

        tree = ElementTree.parse(filename)
        root = tree.getroot()

        domains = tuple(root)
        self.domain = domains[0]
        self.grids = tuple(self.domain)
        self.uniform_grid = self.grids[0]

    def get_topology(self):
        connectivity = None

        for a in self.uniform_grid:
            if a.tag == "Topology":
                connectivity = a

        if connectivity is None:
            raise ValueError("File is missing grid connectivity data")

        return connectivity

    def get_geometry(self):
        geometry = None

        for a in self.uniform_grid:
            if a.tag == "Geometry":
                geometry = a

        if geometry is None:
            raise ValueError("File is missing grid node location data")

        return geometry

    def read_data_item(self, data_item):
        # CURRENTLY DOES NOT SUPPORT 'DataItem' THAT STORES VALUES DIRECTLY

        # check that data stored as separate hdf5 file
        if data_item.get("Format") != "HDF":
            raise TypeError("Data stored in unrecognized format")

        # get corresponding hdf5 file
        source_info = data_item.text
        split_source_info = source_info.partition(":")

        h5_filename = split_source_info[0]
        # TODO: change file name to match actual mirgecom output directory later
        h5_filename = "examples/" + h5_filename
        h5_datapath = split_source_info[2]

        # read data from corresponding hdf5 file
        h5_reader = _Hdf5Reader(h5_filename)
        return h5_reader.read_specific_data(h5_datapath)


def compare_files_xdmf(first_file: str, second_file: str, tolerance: float = 1e-12):
    """Compare files of xdmf type.

    Parameters
    ----------
    first_file:
        First file to compare
    second_file:
        Second file to compare
    file_type:
        Xdmf files
    tolerance:
        Max acceptable absolute difference

    Returns
    -------
    True:
        If it passes the file type test or contains same data.
    False:
        If it fails the file type test or contains different data.
    """
    # read files
    file_reader1 = _XdmfReader(first_file)
    file_reader2 = _XdmfReader(second_file)

    # check same number of grids
    if len(file_reader1.grids) != len(file_reader2.grids):
        print("File 1:", len(file_reader1.grids), "\n",
              "File 2:", len(file_reader2.grids))
        raise ValueError("Fidelity test failed: Mismatched grid count")

    # check same number of cells in gridTrue:
    if len(file_reader1.uniform_grid) != len(file_reader2.uniform_grid):
        print("File 1:", len(file_reader1.uniform_grid), "\n",
              "File 2:", len(file_reader2.uniform_grid))
        raise ValueError("Fidelity test failed: Mismatched cell count in "
                         "uniform grid")

    # compare Topology:
    top1 = file_reader1.get_topology()
    top2 = file_reader2.get_topology()

    # check TopologyType
    if top1.get("TopologyType") != top2.get("TopologyType"):
        print("File 1:", top1.get("TopologyType"), "\n",
              "File 2:", top2.get("TopologyType"))
        raise ValueError("Fidelity test failed: Mismatched topology type")

    # check number of connectivity values
    connectivities1 = file_reader1.read_data_item(tuple(top1)[0])
    connectivities2 = file_reader2.read_data_item(tuple(top2)[0])

    connectivities1 = np.array(connectivities1)
    connectivities2 = np.array(connectivities2)

    if connectivities1.shape != connectivities2.shape:
        print("File 1:", connectivities1.shape, "\n",
              "File 2:", connectivities2.shape)
        raise ValueError("Fidelity test failed: Mismatched connectivities count")

    if not np.allclose(connectivities1, connectivities2, atol=tolerance):
        print("Tolerance:", tolerance)
        raise ValueError("Fidelity test failed: Mismatched connectivity values "
                         "with given tolerance")

    # compare Geometry:
    geo1 = file_reader1.get_geometry()
    geo2 = file_reader2.get_geometry()

    # check GeometryType
    if geo1.get("GeometryType") != geo2.get("GeometryType"):
        print("File 1:", geo1.get("GeometryType"), "\n",
              "File 2:", geo2.get("GeometryType"))
        raise ValueError("Fidelity test failed: Mismatched geometry type")

    # check number of node values
    nodes1 = file_reader1.read_data_item(tuple(geo1)[0])
    nodes2 = file_reader2.read_data_item(tuple(geo2)[0])

    nodes1 = np.array(nodes1)
    nodes2 = np.array(nodes2)

    if nodes1.shape != nodes2.shape:
        print("File 1:", nodes1.shape, "\n", "File 2:", nodes2.shape)
        raise ValueError("Fidelity test failed: Mismatched nodes count")

    if not np.allclose(nodes1, nodes2, atol=tolerance):
        print("Tolerance:", tolerance)
        raise ValueError("Fidelity test failed: Mismatched node values with "
                         "given tolerance")

    # compare other Attributes:
    maxerrorvalue = 0
    for i in range(len(file_reader1.uniform_grid)):
        curr_cell1 = file_reader1.uniform_grid[i]
        curr_cell2 = file_reader2.uniform_grid[i]

        # skip already checked cells
        if curr_cell1.tag == "Geometry" or curr_cell1.tag == "Topology":
            continue

        # check AttributeType
        if curr_cell1.get("AttributeType") != curr_cell2.get("AttributeType"):
            print("File 1:", curr_cell1.get("AttributeType"), "\n",
                  "File 2:", curr_cell2.get("AttributeType"))
            raise ValueError("Fidelity test failed: Mismatched cell type")

        # check Attribtue name
        if curr_cell1.get("Name") != curr_cell2.get("Name"):
            print("File 1:", curr_cell1.get("Name"), "\n",
                  "File 2:", curr_cell2.get("Name"))
            raise ValueError("Fidelity test failed: Mismatched cell name")

        # check number of Attribute values
        values1 = file_reader1.read_data_item(tuple(curr_cell1)[0])
        values2 = file_reader2.read_data_item(tuple(curr_cell2)[0])

        if len(values1) != len(values2):
            print("File 1,", curr_cell1.get("Name"), ":", len(values1), "\n",
                  "File 2,", curr_cell2.get("Name"), ":", len(values2))
            raise ValueError("Fidelity test failed: Mismatched data values count")

        # check values w/in tolerance
        for i in range(len(values1)):
            if abs(values1[i] - values2[i]) > tolerance:
                print("Tolerance:", tolerance, "\n", "Cell:", curr_cell1.get("Name"))
                if maxerrorvalue < abs(values1[i] - values2[i]):
                    maxerrorvalue = abs(values1[i] - values2[i])

    if not maxerrorvalue == 0:
        raise ValueError("Fidelity test failed: Mismatched data array "
                                 "values with given tolerance. "
                                 "Max Error Value:", maxerrorvalue)

    print("XDMF Fidelity test completed successfully with tolerance", tolerance)


def compare_files_hdf5(first_file: str, second_file: str, tolerance: float = 1e-12):
    """Compare files of hdf5 type.

    Parameters
    ----------
    first_file:
        First file to compare
    second_file:
        Second file to compare
    file_type:
        Hdf5 files
    tolerance:
        Max acceptable absolute difference

    Returns
    -------
    True:
        If it passes the file type test or contains same data.
    False:
        If it fails the file type test or contains different data.
    """
    file_reader1 = _Hdf5Reader(first_file)
    file_reader2 = _Hdf5Reader(second_file)
    f1 = file_reader1.file_obj
    f2 = file_reader2.file_obj

    objects1 = list(f1.keys())
    objects2 = list(f2.keys())

    # check number of Grids
    if len(objects1) != len(objects2):
        print("File 1:", len(objects1), "\n", "File 2:", len(objects2))
        raise ValueError("Fidelity test failed: Mismatched grid count")

    # loop through Grids
    maxvalueerror = 0
    for i in range(len(objects1)):
        obj_name1 = objects1[i]
        obj_name2 = objects2[i]

        if obj_name1 != obj_name2:
            print("File 1:", obj_name1, "\n", "File 2:", obj_name2)
            raise ValueError("Fidelity test failed: Mismatched grid names")

        curr_o1 = list(f1[obj_name1])
        curr_o2 = list(f2[obj_name2])

        if len(curr_o1) != len(curr_o2):
            print("File 1,", obj_name1, ":", len(curr_o1), "\n",
                  "File 2,", obj_name2, ":", len(curr_o2))
            raise ValueError("Fidelity test failed: Mismatched group count")

        # loop through Groups
        for j in range(len(curr_o1)):
            subobj_name1 = curr_o1[j]
            subobj_name2 = curr_o2[j]

            if subobj_name1 != subobj_name2:
                print("File 1:", subobj_name1, "\n", "File 2:", subobj_name2)
                raise ValueError("Fidelity test failed: Mismatched group names")

            subpath1 = obj_name1 + "/" + subobj_name1
            subpath2 = obj_name2 + "/" + subobj_name2

            data_arrays_list1 = list(f1[subpath1])
            data_arrays_list2 = list(f2[subpath2])

            if len(data_arrays_list1) != len(data_arrays_list2):
                print("File 1,", subobj_name1, ":", len(data_arrays_list1), "\n",
                      "File 2,", subobj_name2, ":", len(data_arrays_list2))
                raise ValueError("Fidelity test failed: Mismatched data list count")

            # loop through data arrays
            for k in range(len(data_arrays_list1)):
                curr_listname1 = data_arrays_list1[k]
                curr_listname2 = data_arrays_list2[k]

                if curr_listname1 != curr_listname2:
                    print("File 1:", curr_listname1, "\n", "File 2:", curr_listname2)
                    raise ValueError("Fidelity test failed: Mismatched data "
                                     "list names")

                curr_listname1 = subpath1 + "/" + curr_listname1
                curr_listname2 = subpath2 + "/" + curr_listname2

                curr_datalist1 = np.array(list(f1[curr_listname1]))
                curr_datalist2 = np.array(list(f2[curr_listname2]))

                if curr_datalist1.shape != curr_datalist2.shape:
                    print("File 1,", curr_listname1, ":", curr_datalist1.shape, "\n",
                          "File 2,", curr_listname2, ":", curr_datalist2.shape)
                    raise ValueError("Fidelity test failed: Mismatched data "
                                     "list size")

                if not np.allclose(curr_datalist1, curr_datalist2, atol=tolerance):
                    print("Tolerance:", tolerance, "\n",
                          "Data List:", curr_listname1)
                    if maxvalueerror < abs(curr_datalist1 - curr_datalist2):
                        maxvalueerror = abs(curr_datalist1 - curr_datalist2)

    if not maxvalueerror == 0:
        raise ValueError("Fidelity test failed: Mismatched data "
                             "values with given tolerance. "
                             "Max Value Error: ", maxvalueerror)

    print("HDF5 Fidelity test completed successfully with tolerance", tolerance)
