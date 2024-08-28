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
import numpy as np
from meshmode.dof_array import array_context_for_pickling, DOFArray
from grudge.discretization import PartID
from datetime import datetime
from dataclasses import is_dataclass, asdict
from collections import defaultdict
from mirgecom.simutil import (
    invert_decomp,
    interdecomposition_overlap,
    multivolume_interdecomposition_overlap,
    summarize_decomposition,
    copy_mapped_dof_array_data
)
from pprint import pprint


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
        if elements is None:
            continue
        if len(elements) > 0:  # non-empty means this rank has data for this volume
            ranks_seen_volumes[partid.rank].add(partid.volume_tag)

    # Now, find a rank that has seen all volumes
    for rank, seen_volumes in ranks_seen_volumes.items():
        if seen_volumes == all_volumes:
            return rank

    raise ValueError("No rank found with data for all volumes.")


# Traverse the restart_data and copy data from the src
# into the target DOFArrays in-place, so that the trg data
# is persistent across multiple calls with different src data.
def _recursive_map_and_copy(trg_item, src_item, trsrs_idx_maps,
                            src_volume_sizes, trg_rank, src_rank,
                            data_path=None):
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
    if data_path is None:
        data_path = []
    path_string = f"{' -> '.join(map(str, data_path))}"
    if trg_item is None:
        print(f"Target {trg_rank=} item is None at path: {path_string}.")
        print(f"Source {src_rank=} item at this path is: {type(src_item)}")
        # return trg_item
        raise ValueError("trg_item is None, but src_item is not.")

    if src_item is None:
        print(f"{trg_item=}")
        raise ValueError("src_item is None, but trg_item is not.")

    # print(f"{trsrs_idx_maps=}")
    # trg_rank = next(iter(trsrs_idx_maps)).rank
    # print(f"{trg_item=}")
    # print(f"{src_item=}")

    if isinstance(src_item, DOFArray):
        if trg_item is None:
            raise ValueError(
                f"No corresponding target DOFArray found {src_item=}.")
        src_nel, src_nnodes = src_item[0].shape
        volume_tag = next((vol_tag for vol_tag, size in src_volume_sizes.items()
                           if size == src_nel), None)
        if not volume_tag:
            raise ValueError(f"Could not resolve src volume for {path_string}.")
        trg_partid = PartID(volume_tag=volume_tag, rank=trg_rank)
        if trg_partid in trsrs_idx_maps:
            trvs_mapping = trsrs_idx_maps[trg_partid]
            src_partid = PartID(volume_tag=volume_tag, rank=src_rank)
            elem_map = trvs_mapping[src_partid]
            # print(f"dbg Copying data {src_partid=}, {trg_partid=}")
            return copy_mapped_dof_array_data(trg_item, src_item, elem_map)
        else:
            # print("dbg Skipping copy for non-target volume.")
            return trg_item
    elif isinstance(src_item, np.ndarray):
        if trg_item is None:
            raise ValueError(
                f"No corresponding target ndarray found for {src_item=}.")
        # Create a new ndarray with the same shape as the src_item
        result_array = np.empty_like(src_item, dtype=object)
        for idx, array_element in np.ndenumerate(src_item):
            result_array[idx] = _recursive_map_and_copy(
                trg_item[idx], array_element, trsrs_idx_maps,
                src_volume_sizes, trg_rank, src_rank, data_path + [str(idx)])

        return result_array
    elif isinstance(src_item, dict):
        return {k: _recursive_map_and_copy(
            trg_item.get(k, None), v, trsrs_idx_maps,
            src_volume_sizes, trg_rank, src_rank,
            data_path + [k])for k, v in src_item.items()}
    elif isinstance(src_item, (list, tuple)):
        return type(src_item)(_recursive_map_and_copy(
            t, v, trsrs_idx_maps, src_volume_sizes, trg_rank,
            src_rank,
            data_path + [str(idx)]) for idx, (t, v) in enumerate(zip(trg_item,
                                                                     src_item)))
    elif is_dataclass(src_item):
        trg_dict = asdict(trg_item)
        return type(src_item)(**{
            k: _recursive_map_and_copy(
                trg_dict.get(k, None), v, trsrs_idx_maps,
                src_volume_sizes, trg_rank, src_rank, data_path + [k])
            for k, v in asdict(src_item).items()})
    else:
        return src_item  # dupe non-dof data outright


def _ensure_unique_nelems(mesh_data_dict):
    seen_nelems = set()
    for volid, mesh_data in mesh_data_dict.items():
        if mesh_data[0].nelements in seen_nelems:
            raise ValueError(f"Multiple volumes {volid} found with same "
                             "number of elements.")
        seen_nelems.add(mesh_data[0].nelements)


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
    elif isinstance(sample_item, np.ndarray):
        if sample_item.size > 0 and isinstance(sample_item.flat[0], DOFArray):
            # handle ndarray containing DOFArrays
            new_shape = sample_item.shape
            new_arr = np.empty(new_shape, dtype=object)
            for idx, dof in np.ndenumerate(sample_item):
                new_arr[idx] = _recursive_resize_reinit_with_zeros(
                    actx, dof, target_volume_sizes, sample_volume_sizes)
            return new_arr
        else:
            return sample_item  # retain non-DOFArray ndarray data outright
    elif isinstance(sample_item, dict):
        return {k: _recursive_resize_reinit_with_zeros(
            actx, v, target_volume_sizes, sample_volume_sizes)
            for k, v in sample_item.items()}
    elif isinstance(sample_item, (list, tuple)):
        return type(sample_item)(_recursive_resize_reinit_with_zeros(
            actx, v, target_volume_sizes, sample_volume_sizes)
            for v in sample_item)
    elif is_dataclass(sample_item):
        return type(sample_item)(**{k: _recursive_resize_reinit_with_zeros(
            actx, v, target_volume_sizes, sample_volume_sizes)
            for k, v in asdict(sample_item).items()})
    else:
        return sample_item  # retain non-dof data outright


def _get_volume_sizes_for_rank(target_rank, multivol_decomp_map):
    volume_sizes = {}

    for partid, elements in multivol_decomp_map.items():
        if partid.rank == target_rank:
            volume_sizes[partid.volume_tag] = len(elements)

    return volume_sizes


def _extract_src_rank_specific_mapping(trs_olaps, src_rank):
    """Extract source rank-specific mappings from overlap mapping."""
    return {trg_partid: {src_partid: local_mapping
                         for src_partid, local_mapping in src_mappings.items()
                         if src_partid.rank == src_rank}
            for trg_partid, src_mappings in trs_olaps.items()
            if src_rank in {src_partid.rank for src_partid in src_mappings.keys()}}


def _get_item_structure(item):
    """Report data structure with field names."""
    if isinstance(item, DOFArray):
        return "DOFArray"
    elif isinstance(item, np.ndarray):
        # Create a new ndarray with the same shape as the item
        item_structure = np.empty_like(item, dtype=object)

        for idx, array_element in np.ndenumerate(item):
            item_structure[idx] = _get_item_structure(array_element)

        return f"ndarray({', '.join(map(str, item_structure.flatten()))})"
    elif isinstance(item, dict):
        return {k: _get_item_structure(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        item_structure = [_get_item_structure(v) for v in item]
        return f"{type(item).__name__}({', '.join(item_structure)})"
    elif is_dataclass(item):
        item_fields = asdict(item)
        field_structure = [f"{k}: {_get_item_structure(v)}"
                           for k, v in item_fields.items()]
        return f"{type(item).__name__}({', '.join(field_structure)})"
    else:
        return type(item).__name__


def _get_item_structure_and_size(item, ignore_keys=None):
    """Report data structure with field names and sizes for DOFArrays."""
    if ignore_keys is None:
        ignore_keys = {"volume_to_local_mesh_data"}
    if isinstance(item, DOFArray):
        shape = item[0].shape
        return f"DOFArray({shape[0]} elements, {shape[1]} nodes)"
    elif isinstance(item, dict):
        return {k: _get_item_structure_and_size(v)
                for k, v in item.items() if k not in ignore_keys}
    elif isinstance(item, (list, tuple)):
        item_structure = [_get_item_structure_and_size(v) for v in item]
        return f"{type(item).__name__}({', '.join(map(str, item_structure))})"
    elif is_dataclass(item):
        item_fields = asdict(item)
        field_structure = [f"{k}: {_get_item_structure_and_size(v)}"
                           for k, v in item_fields.items() if k not in ignore_keys]
        return f"{type(item).__name__}({', '.join(field_structure)})"
    elif isinstance(item, np.ndarray):
        array_structure = [_get_item_structure_and_size(sub_item)
                           for sub_item in item]
        return f"ndarray({', '.join(map(str, array_structure))})"
    else:
        return type(item).__name__


# Call this one with target-rank-specific mappings (trs_olaps) of the form:
# {targ_partid : { src_partid : {trg_el_index : src_el_index} } }
def _get_restart_data_for_target_rank(
        actx, trg_rank, sample_rst_data, sample_vol_sizes, trs_olaps,
        target_multivol_decomp_map, source_multivol_decomp_map,
        input_path):
    trg_vol_sizes = _get_volume_sizes_for_rank(
        trg_rank, target_multivol_decomp_map)
    # print(f"dbg Target rank = {trg_rank}")
    src_ranks_in_mapping = {k.rank for v in trs_olaps.values()
                            for k in v.keys()}
    # print(f"dbg olap src ranks: {src_ranks_in_mapping}")
    # Read and Map DOFArrays from each source rank to target
    with array_context_for_pickling(actx):
        inp_rst_structure = _get_item_structure(sample_rst_data)
        out_rst_data = _recursive_resize_reinit_with_zeros(
            actx, sample_rst_data, trg_vol_sizes, sample_vol_sizes)
        trg_rst_structure = _get_item_structure(out_rst_data)
        # print(f"Initial output restart data structure {trg_rank=}:")
        # pprint(trg_rst_structure)
        if inp_rst_structure != trg_rst_structure:
            print("Initial structure for input:")
            pprint(inp_rst_structure)
            print(f"Initial structure for {trg_rank=}:")
            pprint(trg_rst_structure)
            raise AssertionError("Input and output data structure mismatch.")

    for src_rank in src_ranks_in_mapping:
        # get src_rank-specific overlaps
        trsrs_idx_maps = _extract_src_rank_specific_mapping(trs_olaps, src_rank)
        # print(f"dbg {src_rank=},{trsrs_idx_maps=}")
        src_vol_sizes = _get_volume_sizes_for_rank(
            src_rank, source_multivol_decomp_map)
        src_restart_file = f"{input_path}-{src_rank:04d}.pkl"
        with array_context_for_pickling(actx):
            with open(src_restart_file, "rb") as f:
                src_rst_data = pickle.load(f)
        src_mesh_data = src_rst_data.pop("volume_to_local_mesh_data", None)
        _ensure_unique_nelems(src_mesh_data)

        # with array_context_for_pickling(actx):
        #    src_rst_structure = _get_item_structure(src_rst_data)
        #    print(f"Copying {src_rank=} data to {trg_rank=} with src structure:")
        #    pprint(src_rst_structure)

        # Copies data for all overlapping parts from src to trg rank data
        with array_context_for_pickling(actx):
            out_rst_data = \
                _recursive_map_and_copy(
                    out_rst_data, src_rst_data, trsrs_idx_maps,
                    src_vol_sizes, trg_rank, src_rank)
            # out_rst_structure = _get_item_structure(out_rst_data)
            # print(f"After copy of {src_rank=}, {trg_rank=} data structure is:")
            # pprint(out_rst_structure)

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
        An MPI communicator object
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

    if my_rank == 0:
        print("Redistributing restart data.")
        # Give some information about the current partitioning:
        print("Source decomp summary:")
        summarize_decomposition(source_idecomp_map, source_multivol_decomp_map)
        print("\nTarget decomp summary:")
        summarize_decomposition(target_idecomp_map, target_multivol_decomp_map)

    # Identify a source rank with data for all volumes
    sample_rank = _find_rank_with_all_volumes(source_multivol_decomp_map)
    if sample_rank is None:
        raise ValueError("No source rank found with data for all volumes.")
    if my_rank == 0:
        print(f"Found source rank {sample_rank} having data for all volumes.")
    sample_restart_file = f"{src_input_path}-{sample_rank:04d}.pkl"
    with array_context_for_pickling(actx):
        with open(sample_restart_file, "rb") as f:
            sample_rst_data = pickle.load(f)
    mesh_data_item = "volume_to_local_mesh_data"
    if "mesh" in sample_rst_data:
        mesh_data_item = "mesh"  # check mesh data return type instead
    vol_to_sample_mesh_data = \
        sample_rst_data.pop(mesh_data_item, None)
    _ensure_unique_nelems(vol_to_sample_mesh_data)
    with array_context_for_pickling(actx):
        inp_rst_structure = _get_item_structure(sample_rst_data)
        if my_rank == 0:
            print("Restart data structure:")
            pprint(inp_rst_structure)

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
        nparts_per_writer = max(1, trg_nparts // writer_nprocs)
        nleftover = trg_nparts - (nparts_per_writer * writer_nprocs)
        nparts_this_writer = nparts_per_writer + (1 if writer_rank
                                                    < nleftover else 0)
        my_starting_rank = nparts_per_writer * writer_rank
        my_starting_rank = my_starting_rank + (writer_rank if writer_rank
                                               < nleftover else nleftover)
        my_ending_rank = my_starting_rank + nparts_this_writer - 1
        parts_to_write = list(range(my_starting_rank, my_ending_rank+1))
        print(f"{my_rank=}, {writer_rank=}, "
              f"Parts[{my_starting_rank},{my_ending_rank}]")
        # print(f"{source_multivol_decomp_map=}")
        # print(f"{target_multivol_decomp_map=}")

        if writer_rank == 0:
            print(f"{datetime.now()}: Computing interdecomp mapping ...")
        xdo = multivolume_interdecomposition_overlap(source_idecomp_map,
                                                     target_idecomp_map,
                                                     source_multivol_decomp_map,
                                                     target_multivol_decomp_map,
                                                     return_ranks=parts_to_write)
        if writer_rank == 0:
            print(f"{datetime.now()}: Computing interdecomp mapping (done)")

        # for trg_partid, olaps in xdo.items():
        #    print(f"dbg {trg_partid=}, "
        #          "f{len(target_multivol_map[trg_partid])=}")
        #    for src_partid, elem_map in olaps.items():
        #        print(f"dbg {src_partid=}, {len(elem_map)=}")

        for trg_rank in parts_to_write:
            if writer_rank == 0:
                print(f"{datetime.now()}: Processing rank {trg_rank}.")
            trs_olaps = {k: v for k, v in xdo.items() if k.rank == trg_rank}
            out_rst_data = _get_restart_data_for_target_rank(
                actx, trg_rank, sample_rst_data, sample_vol_sizes, trs_olaps,
                target_multivol_decomp_map, source_multivol_decomp_map,
                src_input_path)

            with array_context_for_pickling(actx):
                if "nparts" in out_rst_data:   # reset nparts!
                    out_rst_data["nparts"] = trg_nparts
                if "num_parts" in out_rst_data:   # reset nparts!
                    out_rst_data["num_parts"] = trg_nparts
                # print("Output restart structure (sans mesh):")
                out_rst_structure = _get_item_structure(out_rst_data)
                # pprint(out_rst_structure)
                if inp_rst_structure != out_rst_structure:
                    print("Initial structure for input:")
                    pprint(inp_rst_structure)
                    print(f"Output structure for {trg_rank=}:")
                    pprint(out_rst_structure)
                    raise AssertionError("Input and output data structure mismatch.")

            # Read new mesh data and stack it in the restart file
            mesh_pkl_filename = \
                f"{mesh_filename}_mesh_np{trg_nparts}_rank{trg_rank}.pkl"
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

        if writer_rank == 0 and writer_nprocs > 1:
            print(f"{datetime.now()}: Waiting on other ranks to finish ...")
        writer_comm_wrapper.Barrier()

    return
