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

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
import grudge.op as op
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary,
    PrescribedFluxDiffusionBoundary
    )
from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage)

from logpyle import IntervalTimer, set_dt


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):
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

    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 2
    nel_1d = 11

    t = 0
    t_final = 10.0
    istep = 0

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.06,)*dim,
            b=(+0.06,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={
                "neumann": ["+y", "-y"],
                "dirichlet": ["-x"],
                "prescribed": ["+x"],
                }
            )

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 3

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    alpha = 1e-3

    dt = 0.0005
    print('dt=',dt)

    nodes = actx.thaw(dcoll.nodes())

    def my_presc_func(kappa_tpair, grad_u_tpair, normal, **kwargs):
        time = kwargs['time']
        return -actx.np.sin(2.0*np.pi*2.0*time/10.0)

    boundaries = {
        BoundaryDomainTag("neumann"): NeumannDiffusionBoundary(0.),
        BoundaryDomainTag("dirichlet"): DirichletDiffusionBoundary(250.),
        BoundaryDomainTag("prescribed"): PrescribedFluxDiffusionBoundary(my_presc_func),
    }

    u = dcoll.zeros(actx) + 300.0

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", "t_log.max"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    vis = make_visualizer(dcoll)

    def rhs(t, u):
        return (
            diffusion_operator(dcoll, kappa=alpha, boundaries=boundaries, u=u, time=t)
        )

    compiled_rhs = actx.compile(rhs)

    rank = comm.Get_rank()

    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        if istep % 1000 == 0:
            print(istep, t, actx.to_numpy(actx.np.linalg.norm(u[0])))
            vis.write_vtk_file("./viz_data/PrescFlux-%06d.vtu" % (istep),
                [("u", u)], overwrite=True)

        u = rk4_step(u, t, dt, compiled_rhs)
        u = force_evaluation(actx, u)

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
    lazy = args.lazy
    if args.profiling:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_logmgr=args.log, use_leap=args.leap, lazy=lazy,
         use_profiling=args.profiling, casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
