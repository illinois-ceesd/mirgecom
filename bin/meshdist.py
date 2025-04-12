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
import numpy as np
import pickle

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
         output_path=None, log_path=None, periodic=False,
         use_quads=True, geom_scale=1., casename=None,
         use_1d_part=None, use_wall=False, part_axis=None,
         imba_tol=None, use_meshmode=False, gen_only=False,
         nowrite=False):
    """The main function."""
    if mesh_source is None and not use_meshmode:
        raise ApplicationOptionsError("Missing mesh source file or "
                                      "meshmode option")

    if imba_tol is None:
        imba_tol = .01

    if mesh_source is not None:
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
        if use_meshmode:
            print("Generating mesh with meshmode.")
        else:
            print(f"Gmsh mesh source file: {mesh_source}")

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

    def get_mesh_mm():

        """Generate a grid using `gmsh`."""
        size = .001
        angle = 0.
        height = 0.02*geom_scale
        fluid_length = 0.1
        wall_length = 0.05
        bottom_inflow = np.zeros(shape=(dim,))
        top_inflow = np.zeros(shape=(dim,))
        bottom_interface = np.zeros(shape=(dim,))
        top_interface = np.zeros(shape=(dim,))
        bottom_wall = np.zeros(shape=(dim,))
        top_wall = np.zeros(shape=(dim,))

        # rotate the mesh around the bottom-left corner
        theta = angle/180.*np.pi
        bottom_inflow[0] = 0.0
        bottom_inflow[1] = -0.01*geom_scale
        top_inflow[0] = bottom_inflow[0] - height*np.sin(theta)
        top_inflow[1] = bottom_inflow[1] + height*np.cos(theta)

        bottom_interface[0] = bottom_inflow[0] + fluid_length*np.cos(theta)
        bottom_interface[1] = bottom_inflow[1] + fluid_length*np.sin(theta)
        top_interface[0] = top_inflow[0] + fluid_length*np.cos(theta)
        top_interface[1] = top_inflow[1] + fluid_length*np.sin(theta)

        bottom_wall[0] = bottom_interface[0] + wall_length*np.cos(theta)
        bottom_wall[1] = bottom_interface[1] + wall_length*np.sin(theta)
        top_wall[0] = top_interface[0] + wall_length*np.cos(theta)
        top_wall[1] = top_interface[1] + wall_length*np.sin(theta)
        from meshmode.mesh.generation import generate_regular_rect_mesh

        # this only works for non-slanty meshes
        def get_meshmode_mesh(a, b, nelements_per_axis, boundary_tag_to_face):

            from meshmode.mesh import TensorProductElementGroup
            dim = len(a)
            group_cls = TensorProductElementGroup if use_quads else None

            if dim == 2:
                peri = (False, periodic)
            else:
                peri = (False, True, periodic)

            mesh = generate_regular_rect_mesh(
                a=a, b=b, nelements_per_axis=nelements_per_axis,
                group_cls=group_cls, periodic=peri,
                boundary_tag_to_face=boundary_tag_to_face
                )

            mgrp = mesh.groups[0]
            x = mgrp.nodes[0, :, :]
            x_avg = np.sum(x, axis=1)/x.shape[1]
            tag_to_elements = {
                "fluid": np.where(x_avg < fluid_length)[0],
                "wall_insert": np.where(x_avg > fluid_length)[0]}

            return mesh, tag_to_elements

        boundary_tag_to_face = {
            "inflow": ["-x"],
            "outflow": ["+x"],
            "flow": ["-x", "+x"]
        }

        if periodic:
            boundary_tag_to_face["wall_farfield"] = ["+x"]
        if dim == 2:
            a = (bottom_inflow[0], bottom_inflow[1])
            b = (top_wall[0], top_wall[1])
            if not periodic:
                boundary_tag_to_face["wall_farfield"] = \
                    ["+x", "-y", "+y"]
                boundary_tag_to_face["isothermal_wall"] = \
                    ["-y", "+y"]
            nelements_per_axis = (int(fluid_length/size) + int(wall_length/size),
                                  int(height/size))
        else:
            a = (bottom_inflow[0], bottom_inflow[1], 0.)
            b = (top_wall[0], top_wall[1], 0.02)
            if not periodic:  # For 3D meshmode Y is *always* periodic
                boundary_tag_to_face["wall_farfield"] = \
                    ["+x", "-z", "+z"]
                boundary_tag_to_face["isothermal_wall"] = \
                    ["-z", "+z"]
            nelements_per_axis = (int((fluid_length+wall_length)/size),
                                  int(height/size), int(.02/size))
        volume_to_tags = {"fluid": ["fluid"]}
        if use_wall:
            volume_to_tags["wall"] = ["wall_insert"]

        mesh, tag_to_elements = get_meshmode_mesh(
            a=a, b=b, boundary_tag_to_face=boundary_tag_to_face,
            nelements_per_axis=nelements_per_axis)
        return mesh, tag_to_elements, volume_to_tags

    def get_mesh_gmsh():
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

    read_mesh_pkl = False
    if mesh_source is not None:
        read_mesh_pkl = mesh_source.endswith(".pkl")
        if read_mesh_pkl:
            nowrite = True

    def get_mesh_pkl():
        with open(mesh_source, "rb") as pkl_file:
            global_data = pickle.load(pkl_file)
        return global_data

    def my_partitioner(mesh, tag_to_elements, num_ranks):
        from mirgecom.simutil import geometric_mesh_partitioner
        return geometric_mesh_partitioner(
            mesh, num_ranks, auto_balance=True, debug=True,
            imbalance_tolerance=imba_tol, part_axis=part_axis)

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

    mesh_func = get_mesh_mm if use_meshmode else \
        get_mesh_pkl if read_mesh_pkl else get_mesh_gmsh
    write_serial_mesh = not nowrite
    distribute_mesh_pkl(
        comm, mesh_func, filename=mesh_filename,
        num_target_ranks=ndist,
        partition_generator_func=part_func, logmgr=logmgr,
        write_mesh_to_file=write_serial_mesh, gen_only=gen_only)

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
    parser.add_argument("-1", "--1dpart", dest="one_d_part",
                        action="store_true", help="Use 1D partitioner.")
    parser.add_argument("-a", "--axis", type=str, dest="part_axis", nargs="?",
                        action="store", help="Partitioning axis for 1dpart")
    parser.add_argument("-c", "--casename", type=str, dest="casename", nargs="?",
                        action="store",
                        help="Root name of distributed mesh pkl files.")
    parser.add_argument("-d", "--dimen", type=int, dest="dim",
                        nargs="?", action="store", help="Number dimensions")
    parser.add_argument("--gen-only", dest="gen_only",
                        action="store_true", help="generate mesh only")
    parser.add_argument("-g", "--logpath", type=str, dest="log_path", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("-m", "--meshmode", dest="meshmode",
                        action="store_true", help="Use meshmode mesh gen.")
    parser.add_argument("-n", "--ndist", type=int, dest="ndist",
                        nargs="?", action="store",
                        help="Number of distributed parts")
    parser.add_argument("--nowrite", action="store_true", dest="nowrite",
                        help="Do not write serial mesh pkl file.")
    parser.add_argument("-o", "--ouput-path", type=str, dest="output_path",
                        nargs="?", action="store",
                        help="Output path for distributed mesh pkl files")
    parser.add_argument("-p", "--periodic", dest="periodic",
                        action="store_true", help="Generate periodic mesh.")
    parser.add_argument("-s", "--source", type=str, dest="source",
                        nargs="?", action="store", help="Gmsh mesh source file")
    parser.add_argument("--geom-scale", type=float, default=1.,
                        help="Scale the geometry in the Y by this factor.")
    parser.add_argument("-w", "--wall", dest="use_wall",
                        action="store_true", help="Include wall domain in mesh.")
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
         imba_tol=args.imbalance_tolerance, periodic=args.periodic,
         use_meshmode=args.meshmode, geom_scale=args.geom_scale,
         part_axis=args.part_axis, gen_only=args.gen_only,
         nowrite=args.nowrite)
