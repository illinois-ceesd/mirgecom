"""Provide some utilities for restarting simulations.

.. autofunction:: read_restart_data
.. autofunction:: write_restart_file
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

import pickle
from meshmode.dof_array import array_context_for_pickling, DOFArray
from grudge.discretization import PartID
from dataclasses import is_dataclass, asdict
from collections import defaultdict
from mirgecom.simutil import (
    invert_decomp,
    interdecomposition_overlap,
    multivolume_interdecomposition_overlap,
    copy_mapped_dof_array_data
)


class PathError(RuntimeError):
    """Use RuntimeError for filesystem path errors."""

    pass


def read_restart_data(actx, filename):
    """Read the raw restart data dictionary from the given pickle restart file."""
    with array_context_for_pickling(actx):
        with open(filename, "rb") as f:
            return pickle.load(f)


def write_restart_file(actx, restart_data, filename, comm=None):
    """Pickle the simulation data into a file for use in restarting."""
    rank = 0
    if comm:
        rank = comm.Get_rank()
    if rank == 0:
        import os
        rst_dir = os.path.dirname(filename)
        if rst_dir:
            if os.path.exists(rst_dir) and not os.path.isdir(rst_dir):
                raise PathError(f"{rst_dir} exists and is not a directory.")
            elif not os.path.exists(rst_dir):
                os.makedirs(rst_dir, exist_ok=True)
    if comm:
        comm.barrier()
    with array_context_for_pickling(actx):
        with open(filename, "wb") as f:
            pickle.dump(restart_data, f)


def redistribute_restart_data(actx, comm, source_decomp_map_file, input_path,
                              target_decomp_map_file, output_path,
                              mesh_filename):
    """Redistribute a restart dataset M-to-N."""
    from mpi4py.util import pkl5

    def _create_zeros_like_dofarray(actx, nel, ary):
        # Get nnodes from the shape of the sample DOFArray
        _, nnodes = ary[0].shape
        zeros_array = actx.zeros(shape=(nel, nnodes), dtype=ary[0].dtype)
        return DOFArray(actx, (zeros_array,))

    # Traverse the restart_data and copy data from the src
    # into the target DOFArrays in-place, so that the trg data
    # is persistent across multiple calls with different src data.
    def _recursive_map_and_copy(trg_item, src_item, elem_map):
        """Recursively map and copy DOFArrays from src_item."""
        if trg_item is None:
            print(f"{src_item=}")
            raise ValueError("trg_item is None, but src_item is not.")
        if src_item is None:
            print(f"{trg_item=}")
            raise ValueError("src_item is None, but trg_item is not.")
        # print(f"{trg_item=}")
        # print(f"{src_item=}")
        if isinstance(src_item, DOFArray):
            if trg_item is None:
                raise ValueError("No corresponding target DOFArray found.")
            return copy_mapped_dof_array_data(trg_item, src_item, elem_map)
        elif isinstance(src_item, dict):
            return {k: _recursive_map_and_copy(trg_item.get(k, None), v, elem_map)
                    for k, v in src_item.items()}
        elif isinstance(src_item, (list, tuple)):
            return type(src_item)(_recursive_map_and_copy(t, v, elem_map)
                                  for t, v in zip(trg_item, src_item))
        elif is_dataclass(src_item):
            trg_dict = asdict(trg_item)
            return type(src_item)(**{
                k: _recursive_map_and_copy(trg_dict.get(k, None), v,
                                           elem_map)
                for k, v in asdict(src_item).items()})
        else:
            return src_item  # dupe non-dof data outright

    # Creates a restart data set w/zeros of (trg=part-specific) size for
    # all the DOFArray data.  To be updated in-place with data from each
    # src part.
    def _recursive_init_with_zeros(sample_item, trg_zeros):
        """Recursively initialize data structures with zeros or original data."""
        black_list = ["volume_to_local_mesh_data", "mesh"]
        if isinstance(sample_item, DOFArray):
            return 1. * trg_zeros
        elif isinstance(sample_item, dict):
            return {k: _recursive_init_with_zeros(v, trg_zeros)
                    for k, v in sample_item.items() if k not in black_list}
        elif isinstance(sample_item, (list, tuple)):
            return type(sample_item)(_recursive_init_with_zeros(v, trg_zeros)
                                     for v in sample_item)
        elif is_dataclass(sample_item):
            return type(sample_item)(**{k: _recursive_init_with_zeros(v,
                                                                      trg_zeros)
                                        for k, v in asdict(sample_item).items()})
        else:
            return sample_item

    comm_wrapper = pkl5.Intracomm(comm)
    my_rank = comm_wrapper.Get_rank()

    with open(source_decomp_map_file, "rb") as pkl_file:
        src_dcmp = pickle.load(pkl_file)
    with open(target_decomp_map_file, "rb") as pkl_file:
        trg_dcmp = pickle.load(pkl_file)

    trg_parts = invert_decomp(trg_dcmp)
    trg_nparts = len(trg_parts)

    writer_color = 1 if my_rank < trg_nparts else 0
    writer_comm = comm_wrapper.Split(writer_color, my_rank)
    writer_comm_wrapper = pkl5.Intracomm(writer_comm)

    if writer_color:
        writer_nprocs = writer_comm_wrapper.Get_size()
        writer_rank = writer_comm_wrapper.Get_rank()
        nparts_per_writer = int(writer_nprocs / trg_nparts)
        nleftover = trg_nparts - (nparts_per_writer * writer_nprocs)
        nparts_this_writer = nparts_per_writer + (1 if writer_rank
                                                    < nleftover else 0)
        my_starting_rank = nparts_per_writer * writer_rank
        my_starting_rank = my_starting_rank + (writer_rank if writer_rank
                                               < nleftover else nleftover)
        my_ending_rank = my_starting_rank + nparts_this_writer - 1
        parts_to_write = list(range(my_starting_rank, my_ending_rank+1))
        xdo = interdecomposition_overlap(trg_dcmp, src_dcmp,
                                         return_parts=parts_to_write)

        # Read one source restart file to get the 'order' and a sample DOFArray
        mesh_data_item = "volume_to_local_mesh_data"
        sample_restart_file = f"{input_path}-{writer_rank:04d}.pkl"
        with array_context_for_pickling(actx):
            with open(sample_restart_file, "rb") as f:
                sample_data = pickle.load(f)
        if "mesh" in sample_data:
            mesh_data_item = "mesh"  # check mesh data return type instead

        sample_dof_array = \
                    next(val for key, val in
                         sample_data.items() if isinstance(val, DOFArray))

        for trg_part, olaps in xdo.items():
            # Number of elements in each target partition
            trg_nel = len(trg_parts[trg_part])

            # Create representative DOFArray with zeros using sample for order
            # But with nelem = trg_nel
            trg_zeros = _create_zeros_like_dofarray(
                actx, trg_nel, sample_dof_array)
            # Use trg_zeros to reset all dof arrays in the out_rst_data to
            # the current (trg_part) size made of zeros.
            with array_context_for_pickling(actx):
                out_rst_data = _recursive_init_with_zeros(sample_data, trg_zeros)

            # Read and Map DOFArrays from source to target
            for src_part, elem_map in olaps.items():
                # elem_map={trg-local-index : src-local-index} for trg,src parts
                src_restart_file = f"{input_path}-{src_part:04d}.pkl"
                with array_context_for_pickling(actx):
                    with open(src_restart_file, "rb") as f:
                        src_rst_data = pickle.load(f)
                src_rst_data.pop("volume_to_local_mesh_data", None)
                src_rst_data.pop("mesh", None)
                with array_context_for_pickling(actx):
                    out_rst_data = _recursive_map_and_copy(
                        out_rst_data, src_rst_data, elem_map)

            # Read new mesh data and stack it in the restart file
            mesh_pkl_filename = f"{mesh_filename}_np{trg_nparts}_rank{trg_part}.pkl"
            with array_context_for_pickling(actx):
                with open(mesh_pkl_filename, "rb") as pkl_file:
                    global_nelements, volume_to_local_mesh_data = \
                        pickle.load(pkl_file)
                out_rst_data[mesh_data_item] = volume_to_local_mesh_data

            # Write out the trg_part-specific redistributed pkl restart file
            output_filename = f"{output_path}-{trg_part:04d}.pkl"
            with array_context_for_pickling(actx):
                with open(output_filename, "wb") as f:
                    pickle.dump(out_rst_data, f)

    return


def _find_rank_with_all_volumes(multivol_decomp_map):
    """Find a rank that has data for all volumes."""
    # Collect all volume tags from the PartIDs
    all_volumes = \
        {partid.volume_tag for partid in multivol_decomp_map.keys()}

    # Track which ranks have seen which volumes
    ranks_seen_volumes = defaultdict(set)

    for partid, elements in multivol_decomp_map.items():
        if elements:  # non-empty means this rank has data for this volume
            ranks_seen_volumes[partid.rank].add(partid.volume_tag)

    # Now, find a rank that has seen all volumes
    for rank, seen_volumes in ranks_seen_volumes.items():
        if seen_volumes == all_volumes:
            return rank

    raise ValueError("No rank found with data for all volumes.")


# Traverse the restart_data and copy data from the src
# into the target DOFArrays in-place, so that the trg data
# is persistent across multiple calls with different src data.
def _recursive_map_and_copy(trg_item, src_item, trg_partid_to_index_map,
                            src_volume_sizes):
    """
    Recursively map and copy DOFArrays from the source item to the target item.

    Parameters
    ----------
    trg_item: object
        The target item where data will be mapped and copied.
    src_item: object
        The source item from which data will be copied.
    trg_partid_to_index_map: dict
        A mapping from PartID to index for the target.
    src_volume_sizes: dict
        Dictionary of volume sizes for the source.

    Returns
    -------
    object:
        The target item after mapping and copying the data from the source item.
    """
    if trg_item is None:
        print(f"{src_item=}")
        raise ValueError("trg_item is None, but src_item is not.")
    if src_item is None:
        print(f"{trg_item=}")
        raise ValueError("src_item is None, but trg_item is not.")
    trg_rank = next(iter(trg_partid_to_index_map)).rank

    # print(f"{trg_item=}")
    # print(f"{src_item=}")
    src_nel, src_nnodes = src_item[0].shape
    volume_tag = next((vol_tag for vol_tag, size in src_volume_sizes.items()
                       if size == src_nel), None)
    target_partid = PartID(volume_tag=volume_tag, rank=trg_rank)
    elem_map = trg_partid_to_index_map[target_partid]

    if isinstance(src_item, DOFArray):
        if trg_item is None:
            raise ValueError("No corresponding target DOFArray found.")
        return copy_mapped_dof_array_data(trg_item, src_item, elem_map)
    elif isinstance(src_item, dict):
        return {k: _recursive_map_and_copy(
            trg_item.get(k, None), v, trg_partid_to_index_map,
            src_volume_sizes) for k, v in src_item.items()}
    elif isinstance(src_item, (list, tuple)):
        return type(src_item)(_recursive_map_and_copy(
            t, v, trg_partid_to_index_map, src_volume_sizes)
            for t, v in zip(trg_item, src_item))
    elif is_dataclass(src_item):
        trg_dict = asdict(trg_item)
        return type(src_item)(**{
            k: _recursive_map_and_copy(
                trg_dict.get(k, None), v, trg_partid_to_index_map,
                src_volume_sizes) for k, v in asdict(src_item).items()})
    else:
        return src_item  # dupe non-dof data outright


def _ensure_unique_nelems(mesh_data_dict):
    seen_nelems = set()
    for volid, mesh_data in mesh_data_dict.items():
        if mesh_data.nelements in seen_nelems:
            raise ValueError(f"Multiple volumes {volid} found with same "
                             "number of elements.")
        seen_nelems.add(mesh_data.nelem)


def _get_volume_sizes_on_each_rank(multivol_decomp_map):
    volume_sizes = defaultdict(int)
    for partid, elements in multivol_decomp_map.items():
        volume_sizes[(partid.rank, partid.volume_tag)] = len(elements)
    return volume_sizes


def _recursive_resize_reinit_with_zeros(actx, sample_item, target_volume_sizes,
                                         sample_volume_sizes):
    """
    Recursively initialize a composite data structure based on a sample.

    DOFArray data items are initialized with zeros of the appropriate
    target partid size. Non DOFArray items are copied from original sample.

    Parameters
    ----------
    actx: :class:`arraycontext.ArrayContext`
        The array context used for operations.
    sample_item: object
        A sample item to base the initialization on.
    target_volume_sizes: dict
        Target volume sizes.
    sample_volume_sizes: dict
        Sample volume sizes.

    Returns
    -------
    object:
        Initialized data structure.
    """
    if isinstance(sample_item, DOFArray):
        sample_nel, sample_nnodes = sample_item[0].shape
        volume_tag = next((vol_tag for vol_tag, size in sample_volume_sizes.items()
                           if size == sample_nel), None)
        trg_nel = target_volume_sizes[volume_tag]
        _, nnodes = sample_item[0].shape
        zeros_array = actx.zeros(shape=(trg_nel, sample_nnodes),
                                 dtype=sample_item[0].dtype)
        return DOFArray(actx, (zeros_array,))
    elif isinstance(sample_item, dict):
        return {k: _recursive_resize_reinit_with_zeros(actx, v, target_volume_sizes)
                for k, v in sample_item.items()}
    elif isinstance(sample_item, (list, tuple)):
        return type(sample_item)(_recursive_resize_reinit_with_zeros(actx, v,
                                                            target_volume_sizes)
                                 for v in sample_item)
    elif is_dataclass(sample_item):
        return type(sample_item)(**{k: _recursive_resize_reinit_with_zeros(
            actx, v, target_volume_sizes) for k, v in asdict(sample_item).items()})
    else:
        return sample_item  # retain non-dof data outright


def _get_volume_sizes_for_rank(target_rank, multivol_decomp_map):
    volume_sizes = {}

    for partid, elements in multivol_decomp_map.items():
        if partid.rank == target_rank:
            volume_sizes[partid.volume_tag] = len(elements)

    return volume_sizes


def _get_restart_data_for_target_rank(
        actx, trg_rank, sample_rst_data, sample_vol_sizes, src_overlaps,
        target_multivol_decomp_map, source_multivol_decomp_map,
        input_path):
    trg_vol_sizes = _get_volume_sizes_for_rank(
        trg_rank, target_multivol_decomp_map)

    with array_context_for_pickling(actx):
        out_rst_data = _recursive_resize_reinit_with_zeros(
            actx, sample_rst_data, trg_vol_sizes, sample_vol_sizes)

        # Read and Map DOFArrays from source to target
        for src_rank, trg_partid_to_idx_map in src_overlaps.items():
            src_vol_sizes = _get_volume_sizes_for_rank(
                src_rank, source_multivol_decomp_map)

            src_restart_file = f"{input_path}-{src_rank:04d}.pkl"
            with array_context_for_pickling(actx):
                with open(src_restart_file, "rb") as f:
                    src_rst_data = pickle.load(f)
                    mesh_data = src_rst_data.pop("volume_to_local_mesh_data", None)
                    _ensure_unique_nelems(mesh_data)

                with array_context_for_pickling(actx):
                    out_rst_data = _recursive_map_and_copy(
                        out_rst_data, src_rst_data, trg_partid_to_idx_map,
                        src_vol_sizes)

        return out_rst_data


def redistribute_multivolume_restart_data(
        actx, comm, source_idecomp_map, target_idecomp_map,
        source_multivol_decomp_map, target_multivol_decomp_map,
        src_input_path, output_path, mesh_filename):
    """
    Redistribute (m-to-n) multi-volume restart data.

    This function takes in src(m) and trg(n) decomps for multi-volume datasets.
    It then redistributes the restart data from src to match the trg decomposition.

    Parameters
    ----------
    actx: :class:`arraycontext.ArrayContext`
        The array context used for operations
    comm:
        Am MPI communicator object
    source_idecomp_map: dict
        Decomposition map of the source distribution without volume tags.
    target_idecomp_map: dict
        Decomposition map of the target distribution without volume tags.
    source_multivol_decomp_map: dict
        Decomposition map of the source with volume tags. It maps from src `PartID`
        objects to lists of elements.
    target_multivol_decomp_map: dict
        Decomposition map of the target with volume tags. It maps from trg 'PartID'
         objects to lists of elements.
    src_input_path: str
        Path to the source restart data files.
    output_path: str
        Path to save the redistributed restart data files.
    mesh_filename: str
        Base filename of the mesh data for the restart data

    Returns
    -------
    None
        This function doesn't return any value but writes the redistributed
        restart data to the specified `output_path`.
    """
    from mpi4py.util import pkl5

    comm_wrapper = pkl5.Intracomm(comm)
    my_rank = comm_wrapper.Get_rank()

    # Identify a source rank with data for all volumes
    sample_rank = _find_rank_with_all_volumes(source_multivol_decomp_map)
    if sample_rank is None:
        raise ValueError("No source rank found with data for all volumes.")

    mesh_data_item = "volume_to_local_mesh_data"
    sample_restart_file = f"{src_input_path}-{sample_rank:04d}.pkl"
    with array_context_for_pickling(actx):
        with open(sample_restart_file, "rb") as f:
            sample_rst_data = pickle.load(f)
    if "mesh" in sample_rst_data:
        mesh_data_item = "mesh"  # check mesh data return type instead
    vol_to_sample_mesh_data = \
        sample_rst_data.pop(mesh_data_item, None)
    _ensure_unique_nelems(vol_to_sample_mesh_data)
    # sample_vol_sizes, determine from mesh?
    sample_vol_sizes = _get_volume_sizes_for_rank(sample_rank,
                                                  source_multivol_decomp_map)

    trg_nparts = len(target_idecomp_map)

    writer_color = 1 if my_rank < trg_nparts else 0
    writer_comm = comm_wrapper.Split(writer_color, my_rank)
    writer_comm_wrapper = pkl5.Intracomm(writer_comm)

    if writer_color:
        writer_nprocs = writer_comm_wrapper.Get_size()
        writer_rank = writer_comm_wrapper.Get_rank()
        nparts_per_writer = int(writer_nprocs / trg_nparts)
        nleftover = trg_nparts - (nparts_per_writer * writer_nprocs)
        nparts_this_writer = nparts_per_writer + (1 if writer_rank
                                                    < nleftover else 0)
        my_starting_rank = nparts_per_writer * writer_rank
        my_starting_rank = my_starting_rank + (writer_rank if writer_rank
                                               < nleftover else nleftover)
        my_ending_rank = my_starting_rank + nparts_this_writer - 1
        parts_to_write = list(range(my_starting_rank, my_ending_rank+1))
        xdo = multivolume_interdecomposition_overlap(target_idecomp_map,
                                                     source_idecomp_map,
                                                     target_multivol_decomp_map,
                                                     source_multivol_decomp_map,
                                                     return_ranks=parts_to_write)

        for trg_rank, olaps in xdo.items():

            out_rst_data = _get_restart_data_for_target_rank(
                actx, trg_rank, sample_rst_data, sample_vol_sizes,
                olaps, target_multivol_decomp_map, source_multivol_decomp_map,
                src_input_path)

            # Read new mesh data and stack it in the restart file
            mesh_pkl_filename = f"{mesh_filename}_np{trg_nparts}_rank{trg_rank}.pkl"
            with array_context_for_pickling(actx):
                with open(mesh_pkl_filename, "rb") as pkl_file:
                    global_nelements, mesh_data = \
                        pickle.load(pkl_file)
                out_rst_data[mesh_data_item] = mesh_data

            # Write out the trg_part-specific redistributed pkl restart file
            output_filename = f"{output_path}-{trg_rank:04d}.pkl"
            with array_context_for_pickling(actx):
                with open(output_filename, "wb") as f:
                    pickle.dump(out_rst_data, f)

    return
