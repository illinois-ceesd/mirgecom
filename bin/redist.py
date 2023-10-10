"""Re-distribute a mirgecom restart dump and create a new restart dataset."""

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
import argparse
import sys
import os
import glob
import pickle

# from pytools.obj_array import make_obj_array
# from functools import partial

from logpyle import (
    # IntervalTimer,
    set_dt
)
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
    logmgr_add_mempool_usage,
)

from mirgecom.simutil import (
    ApplicationOptionsError,
    distribute_mesh_pkl,
    invert_decomp
)
from mirgecom.mpi import mpi_entry_point


class SingleLevelFilter(logging.Filter):
    """Filter the logger."""

    def __init__(self, passlevel, reject):
        """Initialize the filter."""
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        """Execute the filter."""
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, mesh_source=None, ndist=None, mdist=None,
         output_path=None, input_path=None, log_path=None,
         casename=None, use_1d_part=None, use_wall=False,
         restart_file=None):
    """Redistribute a mirgecom restart dataset."""
    if mesh_source is None:
        raise ApplicationOptionsError("Missing mesh source file.")

    mesh_source.strip("'")

    if log_path is None:
        log_path = "log_data"

    log_path.strip("'")

    if output_path is None:
        output_path = "."
    output_path.strip("'")

    if input_path is None:
        raise ApplicationOptionsError("Input path/filename is required.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    from mpi4py import MPI
    from mpi4py.util import pkl5
    comm_world = MPI.COMM_WORLD
    comm = pkl5.Intracomm(comm_world)
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Default to decomp for one part per process
    if ndist is None:
        ndist = nprocs

    if mdist is None:
        # Try to detect it. If can't then fail.
        if rank == 0:
            search_pattern = input_path + "*"
            print(f"Searching input path {search_pattern}.")
            files = glob.glob(search_pattern)
            print(f"Found files: {files}")
            xps = ["_decomp_", "_mesh_"]
            ffiles = [f for f in files if not any(xc in f for xc in xps)]
            mdist = len(ffiles)
            if mdist <= 0:
                ffiles = [f for f in files if "_decomp_" not in f]
                mdist = len(ffiles)
                if mdist <= 0:
                    mdist = len(files)
        mdist = comm.bcast(mdist, root=0)
        if mdist <= 0:
            raise ApplicationOptionsError("Cannot detect number of parts "
                                          "for input data.")
        else:
            if rank == 0:
                print(f"Automatically detected {mdist} input parts.")

    # We need a decomp map for the input data
    # If can't find, then generate one.
    input_data_directory = os.path.dirname(input_path)
    output_filename = os.path.basename(input_path)
    casename = casename or output_filename
    casename.strip("'")

    if os.path.exists(output_path):
        if not os.path.isdir(output_path):
            raise ApplicationOptionsError(
                "Mesh dist mode requires 'output'"
                " parameter to be a directory for output.")
    if rank == 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    output_directory = output_path
    output_path = output_directory + "/" + output_filename
    mesh_filename = output_directory + "/" + casename + "_mesh"

    if output_path == input_data_directory:
        raise ApplicationOptionsError("Output path must be different than input"
                                      " location because of filename collisions.")

    decomp_map_file_search_pattern = \
        input_data_directory + f"/*_decomp_np{mdist}.pkl"
    input_decomp_map_files = glob.glob(decomp_map_file_search_pattern)
    source_decomp_map_file = \
        input_decomp_map_files[0] if input_decomp_map_files else None

    generate_source_decomp = \
        True if source_decomp_map_file is None else False
    if source_decomp_map_file is None:
        source_decomp_map_file = input_path + f"_decomp_np{mdist}.pkl"
    if generate_source_decomp:
        print("Unable to find source decomp map, generating from scratch.")
    else:
        print("Found existing source decomp map.")
    print(f"Source decomp map file: {source_decomp_map_file}.")

    if rank == 0:
        print(f"Redist on {nprocs} procs: {mdist}->{ndist} parts")
        print(f"Casename: {casename}")
        print(f"Mesh source file: {mesh_source}")

    # logging and profiling
    logname = log_path + "/" + casename + ".sqlite"

    if rank == 0:
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    comm.Barrier()

    logmgr = initialize_logmgr(True,
        filename=logname, mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    alloc = getattr(actx, "allocator", None)

    monitor_memory = True

    logmgr_add_cl_device_info(logmgr, queue)

    if monitor_memory:
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_mempool_usage(logmgr, alloc)

        logmgr.add_watches([
            ("memory_usage_python.max",
             "| Memory:\n| \t python memory: {value:7g} Mb\n")
        ])

        try:
            logmgr.add_watches([
                ("memory_usage_gpu.max",
                 "| \t gpu memory: {value:7g} Mb\n")
            ])
        except KeyError:
            pass

        logmgr.add_watches([
            ("memory_usage_hwm.max",
             "| \t memory hwm: {value:7g} Mb\n")])

    if rank == 0:
        print(f"Reading mesh from {mesh_source}.")
        print(f"Writing {ndist} mesh pkl files to {output_path}.")

    def get_mesh_data():
        from meshmode.mesh.io import read_gmsh
        mesh, tag_to_elements = read_gmsh(
            mesh_source,
            return_tag_to_elements_map=True)
        volume_to_tags = {
            "fluid": ["fluid"]}
        if use_wall:
            volume_to_tags["wall"] = ["wall_insert", "wall_surround"]
        else:
            from mirgecom.simutil import extract_volumes
            mesh, tag_to_elements = extract_volumes(
                mesh, tag_to_elements, volume_to_tags["fluid"],
                "wall_interface")
        return mesh, tag_to_elements, volume_to_tags

    def my_partitioner(mesh, tag_to_elements, num_ranks):
        if use_1d_part:
            from mirgecom.simutil import geometric_mesh_partitioner
            return geometric_mesh_partitioner(
                mesh, num_ranks, auto_balance=True, debug=False)
        else:
            from meshmode.distributed import get_partition_by_pymetis
            return get_partition_by_pymetis(mesh, num_ranks)

    part_func = my_partitioner

    # This bit will write the source decomp's (M) partition table if it
    # didn't already exist.  We need this table to create the
    # N-parted restart data from the M-parted data.
    if generate_source_decomp:
        if rank == 0:
            print("Generating source decomp...")
            source_mesh_data = get_mesh_data()
            from meshmode.mesh import Mesh
            if isinstance(source_mesh_data, Mesh):
                multivolume_dataset = False
                source_mesh = source_mesh_data
                tag_to_elements = None
                volume_to_tags = None
            elif isinstance(source_mesh_data, tuple):
                source_mesh, tag_to_elements, volume_to_tags = source_mesh_data
                multivolume_dataset = True
            else:
                raise TypeError("Unexpected result from get_mesh_data")
            comm.bcast(multivolume_dataset)
            rank_per_element = my_partitioner(source_mesh, tag_to_elements, mdist)
            with open(source_decomp_map_file, "wb") as pkl_file:
                pickle.dump(rank_per_element, pkl_file)
            print("Done generating source decomp.")
        else:
            multivolume_dataset = comm.bcast(None)

    comm.Barrier()

    if rank == 0:
        meshtype = " multi-volume " if multivolume_dataset else ""
        print(f"Partitioning {meshtype} mesh to {ndist} parts, "
              f"writing to {mesh_filename}...")

    # This bit creates the N-parted mesh pkl files and partition table
    # Should only do this if the n decomp map is not found in the output path.
    distribute_mesh_pkl(
        comm, get_mesh_data, filename=mesh_filename,
        num_target_ranks=ndist, partition_generator_func=part_func, logmgr=logmgr)

    comm.Barrier()

    if rank == 0:
        print("Done partitioning target mesh, mesh pkl files written.")
        print(f"Generating the restart data for {ndist} parts...")

    if multivolume_dataset:
        target_decomp_map_file = mesh_filename + f"_decomp_np{ndist}.pkl"
        target_multivol_decomp_map_file = \
            mesh_filename + f"_multivol_idecomp_np{ndist}.pkl"
        with open(target_decomp_map_file, "rb") as pkl_file:
            trg_dcmp = pickle.load(pkl_file)
            trg_idcmp = invert_decomp(trg_dcmp)
        with open(target_multivol_decomp_map_file, "rb") as pkl_file:
            trg_mv_dcmp = pickle.load(pkl_file)
        source_decomp_map_file = input_path + f"_decomp_np{mdist}.pkl"
        source_multivol_decomp_map_file = \
            input_path + f"_multivol_idcomp_np{mdist}.pkl"
        with open(source_decomp_map_file, "rb") as pkl_file:
            src_dcmp = pickle.load(pkl_file)
            src_idcmp = invert_decomp(src_dcmp)
        with open(source_multivol_decomp_map_file, "rb") as pkl_file:
            src_mv_dcmp = pickle.load(pkl_file)
        from mirgecom.restart import redistribute_multivolume_restart_data
        redistribute_multivolume_restart_data(
            actx, comm, src_idcmp, trg_idcmp,
            src_mv_dcmp, trg_mv_dcmp, input_path,
            output_path, mesh_filename)
    else:
        target_decomp_map_file = mesh_filename + f"_decomp_np{ndist}.pkl"
        from mirgecom.restart import redistribute_restart_data
        redistribute_restart_data(actx, comm, source_decomp_map_file, input_path,
                                  target_decomp_map_file, output_path, mesh_filename)

    if rank == 0:
        print("Done generating restart data.")
        print(f"Restart data for {ndist} parts is in {output_path}")

    logmgr_set_time(logmgr, 0, 0)
    logmgr
    logmgr.tick_before()
    set_dt(logmgr, 0.)
    logmgr.tick_after()
    logmgr.close()


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="MIRGE-Com Mesh Distribution")
    parser.add_argument("-w", "--wall", dest="use_wall",
                        action="store_true", help="Include wall domain in mesh.")
    parser.add_argument("-1", "--1dpart", dest="one_d_part",
                        action="store_true", help="Use 1D partitioner.")
    parser.add_argument("-n", "--ndist", type=int, dest="ndist",
                        nargs="?", action="store",
                        help="Number of distributed parts")
    parser.add_argument("-m", "--mdist", type=int, dest="mdist",
                        nargs="?", action="store",
                        help="Number of source data parts")
    parser.add_argument("-s", "--source", type=str, dest="source",
                        nargs="?", action="store", help="Gmsh mesh source file")
    parser.add_argument("-o", "--ouput-path", type=str, dest="output_path",
                        nargs="?", action="store",
                        help="Output directory for distributed mesh pkl files")
    parser.add_argument("-i", "--input-path", type=str, dest="input_path",
                        nargs="?", action="store",
                        help="Input path/root filename for restart pkl files")
    parser.add_argument("-c", "--casename", type=str, dest="casename", nargs="?",
                        action="store",
                        help="Root name of distributed mesh pkl files.")
    parser.add_argument("-g", "--logpath", type=str, dest="log_path", nargs="?",
                        action="store", help="simulation case name")

    args = parser.parse_args()

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=False, distributed=True, profiling=False, numpy=False)

    main(actx_class, mesh_source=args.source,
         output_path=args.output_path, ndist=args.ndist,
         input_path=args.input_path, mdist=args.mdist,
         log_path=args.log_path, casename=args.casename,
         use_1d_part=args.one_d_part, use_wall=args.use_wall)
