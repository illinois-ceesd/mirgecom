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


def _is_mesh_data_multivol(mesh_filename, npart):
    mesh_root = mesh_filename + "_mesh"
    mvdecomp_filename = mesh_root + f"_multivol_idecomp_np{npart}.pkl"
    return os.path.exists(mvdecomp_filename)


def _validate_mesh_data_files(mesh_filename, npart, mv=False):
    mesh_root = mesh_filename + "_mesh"
    part_file_root = mesh_root + f"_np{npart}_rank"
    decomp_filename = mesh_root + f"_decomp_np{npart}.pkl"
    idecomp_filename = mesh_root + f"_idecomp_np{npart}.pkl"
    mvdecomp_filename = mesh_root + f"_multivol_idecomp_np{npart}.pkl"

    data_is_good = True
    if not os.path.exists(decomp_filename):
        print(f"Failed to find mesh decomp file: {decomp_filename}.")
        data_is_good = False
    if not os.path.exists(idecomp_filename):
        print(f"Failed to find mesh idecomp file: {idecomp_filename}.")
        data_is_good = False
    if mv:
        if not os.path.exists(mvdecomp_filename):
            print(f"Failed to find multivol decomp file: {mvdecomp_filename}.")
            data_is_good = False
    bad_ranks = []
    for r in range(npart):
        part_filename = part_file_root + f"{r}.pkl"
        if not os.path.exists(part_filename):
            bad_ranks.append(r)
            data_is_good = False
    if len(bad_ranks) > 0:
        print(f"Failed to find mesh data for ranks {bad_ranks}.")
    if not data_is_good:
        raise ApplicationOptionsError(
            f"Could not find expected mesh data for {mesh_filename}.")


@mpi_entry_point
def main(actx_class, mesh_source=None, ndist=None, mdist=None,
         output_path=None, input_path=None, log_path=None,
         src_mesh_filename=None, trg_mesh_filename=False):
    """Redistribute a mirgecom restart dataset."""
    from mpi4py import MPI
    from mpi4py.util import pkl5
    comm_world = MPI.COMM_WORLD
    comm = pkl5.Intracomm(comm_world)
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if log_path is None:
        log_path = "log_data"

    log_path.strip("'")

    if output_path is None:
        output_path = "."
    output_path.strip("'")

    if input_path is None:
        raise ApplicationOptionsError("Input path/filename is required.")

    if src_mesh_filename is None:
        raise ApplicationOptionsError("Source mesh filename must be specified.")

    if trg_mesh_filename is None:
        raise ApplicationOptionsError("Target mesh filename must be specified.")

    # Default to decomp for one part per process
    if ndist is None:
        ndist = nprocs

    if mdist is None:
        raise ApplicationOptionsError("Number of src ranks (m) is unspecified.")

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

    input_data_directory = os.path.dirname(input_path)
    output_filename = os.path.basename(input_path)

    if os.path.exists(output_path):
        if not os.path.isdir(output_path):
            raise ApplicationOptionsError(
                "Redist mode requires 'output'"
                " parameter to be a directory for output.")
    if rank == 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    output_directory = output_path
    output_path = output_directory + "/" + output_filename

    if output_directory == input_data_directory:
        raise ApplicationOptionsError("Output path must be different than input"
                                      " location because of filename collisions.")

    if rank == 0:
        print(f"Redist on {nprocs} procs: {mdist}->{ndist} MPI ranks")
        print(f"Source mesh: {src_mesh_filename}")
        print(f"Target mesh: {trg_mesh_filename}")
        print(f"Input restart data: {input_path}")
        print(f"Output restart data: {output_path}")

    # logging and profiling
    logname = log_path + f"/redist-{mdist}-to-{ndist}.sqlite"

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

    multivolume_dataset = _is_mesh_data_multivol(src_mesh_filename, mdist)
    _validate_mesh_data_files(src_mesh_filename, mdist)
    _validate_mesh_data_files(trg_mesh_filename, ndist)

    src_mesh_root = src_mesh_filename + "_mesh"
    src_decomp_filename = src_mesh_root + f"_decomp_np{mdist}.pkl"
    src_idecomp_filename = src_mesh_root + f"_idecomp_np{mdist}.pkl"
    src_mvdecomp_filename = src_mesh_root + f"_multivol_idecomp_np{mdist}.pkl"

    trg_mesh_root = trg_mesh_filename + "_mesh"
    trg_decomp_filename = trg_mesh_root + f"_decomp_np{ndist}.pkl"
    trg_idecomp_filename = trg_mesh_root + f"_idecomp_np{ndist}.pkl"
    trg_mvdecomp_filename = trg_mesh_root + f"_multivol_idecomp_np{ndist}.pkl"

    with open(src_idecomp_filename, "rb") as pkl_file:
        src_idcmp = pickle.load(pkl_file)
    with open(trg_idecomp_filename, "rb") as pkl_file:
        trg_idcmp = pickle.load(pkl_file)

    if rank == 0:
        print("Generating new restart data.")

    if multivolume_dataset:
        with open(trg_mvdecomp_filename, "rb") as pkl_file:
            trg_mv_dcmp = pickle.load(pkl_file)
        with open(src_mvdecomp_filename, "rb") as pkl_file:
            src_mv_dcmp = pickle.load(pkl_file)
        from mirgecom.restart import redistribute_multivolume_restart_data
        redistribute_multivolume_restart_data(
            actx, comm, src_idcmp, trg_idcmp,
            src_mv_dcmp, trg_mv_dcmp, input_path,
            output_path, trg_mesh_filename)
    else:
        from mirgecom.restart import redistribute_restart_data
        redistribute_restart_data(actx, comm, src_decomp_filename, input_path,
                                  trg_decomp_filename, output_path, trg_mesh_root)

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
        description="MIRGE-Com Restart redistribution utility.")
    parser.add_argument("-n", "--ndist", type=int, dest="ndist",
                        nargs="?", action="store",
                        help="Number of distributed parts")
    parser.add_argument("-m", "--mdist", type=int, dest="mdist",
                        nargs="?", action="store",
                        help="Number of source data parts")
    parser.add_argument("-o", "--ouput-path", type=str, dest="output_path",
                        nargs="?", action="store",
                        help="Output directory for distributed mesh pkl files")
    parser.add_argument("-s", "--source-mesh", type=str, dest="src_mesh",
                        nargs="?", action="store",
                        help="Path/filename for source (m parts) mesh.")
    parser.add_argument("-t", "--target-mesh", type=str, dest="trg_mesh",
                        nargs="?", action="store",
                        help="Path/filename for target (n parts) mesh.")
    parser.add_argument("-i", "--input-path", type=str, dest="input_path",
                        nargs="?", action="store",
                        help="Input path/root filename for restart pkl files")
    parser.add_argument("-g", "--logpath", type=str, dest="log_path", nargs="?",
                        action="store", help="simulation case name")

    args = parser.parse_args()

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=False, distributed=True, profiling=False, numpy=False)

    main(actx_class, ndist=args.ndist, mdist=args.mdist,
         output_path=args.output_path, input_path=args.input_path,
         src_mesh_filename=args.src_mesh,
         trg_mesh_filename=args.trg_mesh,
         log_path=args.log_path)
