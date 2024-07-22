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

import grudge.op as op
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from mirgecom.mpi import mpi_entry_point
from mirgecom.simutil import write_visfile, check_step, check_naninf_local
from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage)

from logpyle import IntervalTimer, set_dt


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, use_esdg=False,
         use_overintegration=False,
         use_leap=False, casename=None, rst_filename=None):
    """Run the example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from functools import partial
    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename="heat-source.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 2
    nel_1d = 16

    t = 0
    t_final = 0.0002
    istep = 0

    nviz = 10
    viz_path = "viz_data/"
    vizname = viz_path+casename

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

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    from grudge.dof_desc import DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None  # noqa

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.0025/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    source_width = 0.2

    nodes = actx.thaw(dcoll.nodes())

    boundaries = {
        BoundaryDomainTag("dirichlet"): DirichletDiffusionBoundary(0.),
        BoundaryDomainTag("neumann"): NeumannDiffusionBoundary(0.)
    }

    u = dcoll.zeros(actx)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
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

    visualizer = make_visualizer(dcoll)

    def my_write_viz(step, t, state):
        viz_fields = [("u", state)]
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_health_check(u):
        health_error = False
        if check_naninf_local(dcoll, "vol", u):
            health_error = True
            logger.info(f"{rank=}: Invalid field data found.")

        return health_error

    from mirgecom.simutil import componentwise_norms

    def my_pre_step(step, t, dt, state):
        if logmgr:
            logmgr.tick_before()

        try:

            if istep % 10 == 0:
                norm = actx.to_numpy(componentwise_norms(dcoll, state))
                print(istep, t, norm)

            health_errors = global_reduce(my_health_check(state), op="lor")
            if health_errors:
                if rank == 0:
                    logger.info("Solution failed health check.")
                raise MyRuntimeError("Failed simulation health check.")

            do_viz = check_step(step=step, interval=nviz)
            if do_viz:
                my_write_viz(step=step, t=t, state=state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            raise

        return state, dt

    def my_rhs(t, state):
        return (
            diffusion_operator(dcoll, kappa=1, boundaries=boundaries, u=state,
                               quadrature_tag=quadrature_tag)
            + actx.np.exp(-np.dot(nodes, nodes)/source_width**2))

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    from mirgecom.steppers import advance_state
    current_step, current_t, advanced_state = \
        advance_state(rhs=my_rhs, timestepper=rk4_step,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=dt, state=u, t=t,
                      t_final=t_final, istep=istep, force_eval=True)

    final_answer = actx.to_numpy(op.norm(dcoll, advanced_state, np.inf))
    resid = abs(final_answer - 0.0002062062188374177)
    if resid > 1e-15:
        raise ValueError(f"Run did not produce the expected result {resid=}")

    if logmgr:
        logmgr.close()


if __name__ == "__main__":
    import argparse
    casename = "heat-source"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--overintegration", action="store_true",
        help="turn on overintegration.")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_leap=args.leap,
         use_overintegration=args.overintegration,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
