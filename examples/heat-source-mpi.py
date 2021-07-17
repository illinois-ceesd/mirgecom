"""Demonstrate heat source example."""

__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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
import numpy.linalg as la  # noqa
import pyopencl as cl

from meshmode.array_context import (
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from mirgecom.profiling import PyOpenCLProfilingArrayContext

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_BASE, DTAG_BOUNDARY
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_device_name,
                                         logmgr_add_device_memory_usage)

from logpyle import IntervalTimer, set_dt


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=PyOpenCLArrayContext):
    """Run the example."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename="heat-source.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 2
    nel_1d = 16

    t = 0
    t_final = 0.01
    istep = 0

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={
                "dirichlet": ["+x", "-x"],
                "neumann": ["+y", "-y"]
                }
            )

        print("%d elements" % mesh.nelements)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 3

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                    mpi_communicator=comm)

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.0025/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    source_width = 0.2

    from meshmode.array_context import thaw
    nodes = thaw(actx, discr.nodes())

    boundaries = {
        DTAG_BOUNDARY("dirichlet"): DirichletDiffusionBoundary(0.),
        DTAG_BOUNDARY("neumann"): NeumannDiffusionBoundary(0.)
    }

    u = discr.zeros(actx)

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches(["step.max", "t_step.max", "t_log.max"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    vis = make_visualizer(discr)

    def rhs(t, u):
        return (
            diffusion_operator(
                discr, quad_tag=DISCR_TAG_BASE,
                alpha=1, boundaries=boundaries, u=u)
            + actx.np.exp(-np.dot(nodes, nodes)/source_width**2))

    rank = comm.Get_rank()

    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        if istep % 10 == 0:
            print(istep, t, discr.norm(u))
            vis.write_vtk_file("fld-heat-source-mpi-%03d-%04d.vtu" % (rank, istep),
                    [
                        ("u", u)
                        ])

        u = rk4_step(u, t, dt, rhs)
        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()


if __name__ == "__main__":
    import argparse
    casename = "heat-source"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    if args.profiling:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")
        actx_class = PyOpenCLProfilingArrayContext
    else:
        actx_class = PytatoPyOpenCLArrayContext if args.lazy \
            else PyOpenCLArrayContext

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class)

# vim: foldmethod=marker
