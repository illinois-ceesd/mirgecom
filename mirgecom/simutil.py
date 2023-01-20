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

Mesh utilities
--------------

.. autofunction:: generate_and_distribute_mesh

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
import numpy as np
import grudge.op as op

from arraycontext import map_array_container, flatten

from functools import partial

from meshmode.dof_array import DOFArray

from typing import List, Dict, Optional
from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import DD_VOLUME_ALL
from mirgecom.viscous import get_viscous_timestep

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
                  comm=None):
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
    from mirgecom.io import make_rank_fname, make_par_fname

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


def get_reasonable_memory_pool(ctx, queue):
    """Return an SVM or buffer memory pool based on what the device supports."""
    from pyopencl.characterize import has_coarse_grain_buffer_svm
    import pyopencl.tools as cl_tools

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
            return d[config_key]
    return default_value


def compare_files_vtu(
        first_file: str,
        second_file: str,
        file_type: str,
        tolerance: float = 1e-12,
        field_tolerance: Optional[Dict[str, float]] = None
        ):
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
    import vtk
    import xml.etree.ElementTree as Et

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
