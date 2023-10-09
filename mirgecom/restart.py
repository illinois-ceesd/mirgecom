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
from meshmode.dof_array import array_context_for_pickling


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
    from meshmode.dof_array import DOFArray
    from dataclasses import is_dataclass, asdict
    from mirgecom.simutil import (
        invert_decomp,
        interdecomposition_overlap,
        copy_mapped_dof_array_data
    )
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


def redistribute_multivolume_restart_data(
        actx, comm, source_multivol_decomp_map_file, input_path,
        target_multivol_decomp_map_file, output_path, mesh_filename):
    """Redistribute a multi-volume restart dataset M-to-N."""
    from meshmode.dof_array import DOFArray
    from dataclasses import is_dataclass, asdict
    from mirgecom.simutil import (
        invert_decomp,
        interdecomposition_overlap,
        copy_mapped_dof_array_data
    )
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

    def _ensure_unique_nelems(mesh_data_dict):
        seen_nelems = set()

        for volid, mesh_data in mesh_data_dict.items():
            if mesh_data.nelements in seen_nelems:
                raise ValueError(f"Multiple volumes {volid} found with same "
                                 "number of elements.")
            seen_nelems.add(mesh_data.nelem)

    comm_wrapper = pkl5.Intracomm(comm)
    my_rank = comm_wrapper.Get_rank()

    with open(source_multivol_decomp_map_file, "rb") as pkl_file:
        src_dcmp = pickle.load(pkl_file)
    with open(target_multivol_decomp_map_file, "rb") as pkl_file:
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
                vol_to_src_mesh_data = \
                    src_rst_data.pop("volume_to_local_mesh_data", None)
                _ensure_unique_nelems(vol_to_src_mesh_data)
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
