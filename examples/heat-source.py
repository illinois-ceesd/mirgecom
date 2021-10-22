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
import sys
import logging
import numpy as np
import numpy.linalg as la  # noqa
from functools import partial
import pyopencl as cl

from arraycontext import thaw
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
from mirgecom.steppers import advance_state
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from mirgecom.simutil import (
    generate_and_distribute_mesh,
    get_next_timestep,
    write_visfile
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_device_name,
                                         logmgr_add_device_memory_usage,
                                         logmgr_add_discretization_quantity,
                                         set_sim_state)

from logpyle import IntervalTimer, set_dt

logger = logging.getLogger(__name__)


class SimError(RuntimeError):
    """Simple exception for fatal driver errors."""

    pass


def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=PyOpenCLArrayContext):
    """Run the example."""
    cl_ctx = ctx_factory()

    if "mpi4py.MPI" in sys.modules:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nproc = comm.Get_size()
    else:
        comm = None
        rank = 0
        nproc = 1

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

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

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

    # Some discretization parameters
    dim = 2
    nel_1d = 16
    order = 3

    # {{{ Time stepping control

    # Time stepper selection
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    # Time loop control parameters
    t_final = 0.0002
    sim_dt = 1.e-5

    # i/o frequencies
    nhealth = 1
    nrestart = 10
    nviz = 10

    # }}}  Time stepping control

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == nproc
    else:  # generate the grid from scratch
        def generate_mesh():
            from meshmode.mesh.generation import generate_regular_rect_mesh
            return generate_regular_rect_mesh(
                a=(-0.5,)*dim, b=(0.5,)*dim,
                nelements_per_axis=(nel_1d,)*dim,
                boundary_tag_to_face={
                    "dirichlet": ["+x", "-x"],
                    "neumann": ["+y", "-y"]
                })
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(discr.nodes(), actx)

    boundaries = {
        DTAG_BOUNDARY("dirichlet"): DirichletDiffusionBoundary(0.),
        DTAG_BOUNDARY("neumann"): NeumannDiffusionBoundary(0.)
    }

    if rst_filename:
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        current_state = restart_data["state"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_step = 0
        current_t = 0
        current_state = discr.zeros(actx)

    vis_timer = None

    if logmgr:
        logmgr_add_discretization_quantity(
            logmgr, discr, lambda state: {"u": state}, lambda _: "", "u")

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_u", "------- u (min, max) = ({value:1.9e}, "),
            ("max_u",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr)

    init_message = make_init_message(
        casename=casename,
        dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        extra_params_dict={
            "Final time": t_final,
            "Timestep": sim_dt,
        })
    if rank == 0:
        logger.info(init_message)

    def write_viz(step, t, state):
        viz_fields = [
            ("u", state)
        ]
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nproc
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def health_check(state):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", state) \
           or check_range_local(discr, "vol", state, -1e5, 0.05):
            health_error = True
            logger.info(f"{rank=}: Invalid state data found.")
        return health_error

    def pre_step(step, t, state):
        try:
            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_health = check_step(step=step, interval=nhealth)
            do_restart = check_step(step=step, interval=nrestart)
            do_viz = check_step(step=step, interval=nviz)

            if do_health:
                health_errors = global_reduce(health_check(state), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Solution failed health check.")
                    raise SimError("Failed simulation health check.")

            if do_restart:
                write_restart(step=step, t=t, state=state)

            if do_viz:
                write_viz(step=step, t=t, state=state)

        except SimError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            write_restart(step=step, t=t, state=state)
            write_viz(step=step, t=t, state=state)
            raise

        return state, get_next_timestep(t, t_final, sim_dt)

    def post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, state)
            logmgr.tick_after()
        return state

    def rhs(t, state):
        return (
            diffusion_operator(
                discr, quad_tag=DISCR_TAG_BASE,
                alpha=1, boundaries=boundaries, u=state)
            + actx.np.exp(-np.dot(nodes, nodes)/0.2**2))

    current_step, current_t, current_state = \
        advance_state(rhs=rhs, timestepper=timestepper,
                      pre_step_callback=pre_step,
                      post_step_callback=post_step,
                      step=current_step, t=current_t, state=current_state,
                      t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    write_restart(step=current_step, t=current_t, state=current_state)
    write_viz(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="Heat Source")
    parser.add_argument("--mpi", action="store_true", help="run with MPI")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--casename", help="casename to use for i/o")
    parser.add_argument("--restart_file", help="root name of restart file")
    args = parser.parse_args()

    if args.mpi:
        main_func = mpi_entry_point(main)
    else:
        main_func = main

    if args.profiling:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")
        actx_class = PyOpenCLProfilingArrayContext
    else:
        actx_class = PytatoPyOpenCLArrayContext if args.lazy \
            else PyOpenCLArrayContext

    if args.casename:
        casename = args.casename
    else:
        casename = "heat-source"

    if args.restart_file:
        rst_filename = args.restart_file
    else:
        rst_filename = None

    main_func(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class)

# vim: foldmethod=marker
