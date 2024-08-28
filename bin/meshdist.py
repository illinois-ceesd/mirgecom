"""Read gmsh mesh, partition it, and create a pkl file per mesh partition."""

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

from logpyle import set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
    logmgr_add_mempool_usage,
)

from mirgecom.simutil import (
    ApplicationOptionsError,
    distribute_mesh_pkl
)
from mirgecom.mpi import mpi_entry_point


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, mesh_source=None, ndist=None, dim=None,
         output_path=None, log_path=None,
         casename=None, use_1d_part=None, use_wall=False,
         imba_tol=0.01):
    """The main function."""
    if mesh_source is None:
        raise ApplicationOptionsError("Missing mesh source file.")

    mesh_source.strip("'")

    if dim is None:
        dim = 3

    if log_path is None:
        log_path = "log_data"

    log_path.strip("'")

    if output_path is None:
        output_path = "."
    output_path.strip("'")

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
    nparts = comm.Get_size()

    if ndist is None:
        ndist = nparts

    if casename is None:
        casename = f"mirgecom_np{ndist}"
    casename.strip("'")

    if rank == 0:
        print(f"Distributing on {nparts} ranks into {ndist} parts.")
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
            mesh_source, force_ambient_dim=dim,
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
        from mirgecom.simutil import geometric_mesh_partitioner
        return geometric_mesh_partitioner(
            mesh, num_ranks, auto_balance=True, debug=True,
            imbalance_tolerance=imba_tol)

    part_func = my_partitioner if use_1d_part else None

    if os.path.exists(output_path):
        if not os.path.isdir(output_path):
            raise ApplicationOptionsError(
                "Mesh dist mode requires 'output'"
                " parameter to be a directory for output.")
    if rank == 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    comm.Barrier()
    mesh_filename = output_path + "/" + casename + "_mesh"

    if rank == 0:
        print(f"Writing mesh pkl files to {mesh_filename}.")

    distribute_mesh_pkl(
        comm, get_mesh_data, filename=mesh_filename,
        num_target_ranks=ndist,
        partition_generator_func=part_func, logmgr=logmgr)

    comm.Barrier()

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
    parser.add_argument("-d", "--dimen", type=int, dest="dim",
                        nargs="?", action="store",
                        help="Number dimensions")
    parser.add_argument("-n", "--ndist", type=int, dest="ndist",
                        nargs="?", action="store",
                        help="Number of distributed parts")
    parser.add_argument("-s", "--source", type=str, dest="source",
                        nargs="?", action="store", help="Gmsh mesh source file")
    parser.add_argument("-o", "--ouput-path", type=str, dest="output_path",
                        nargs="?", action="store",
                        help="Output path for distributed mesh pkl files")
    parser.add_argument("-c", "--casename", type=str, dest="casename", nargs="?",
                        action="store",
                        help="Root name of distributed mesh pkl files.")
    parser.add_argument("-g", "--logpath", type=str, dest="log_path", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("-z", "--imbatol", type=float, dest="imbalance_tolerance",
                        nargs="?", action="store",
                        help="1d partioner imabalance tolerance")

    args = parser.parse_args()

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=False, distributed=True, profiling=False, numpy=False)

    main(actx_class, mesh_source=args.source, dim=args.dim,
         output_path=args.output_path, ndist=args.ndist,
         log_path=args.log_path, casename=args.casename,
         use_1d_part=args.one_d_part, use_wall=args.use_wall,
         imba_tol=args.imbalance_tolerance)
