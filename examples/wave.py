"""Demonstrate wave example."""

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
from functools import partial

import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl

from pytools.obj_array import flat_obj_array

from arraycontext import thaw
from meshmode.array_context import (PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext)

from mirgecom.profiling import PyOpenCLProfilingArrayContext  # noqa

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.simutil import (
    generate_and_distribute_mesh,
    get_next_timestep,
    write_visfile
)
from mirgecom.io import make_init_message
from mirgecom.wave import wave_operator

import pyopencl.tools as cl_tools

from logpyle import IntervalTimer, set_dt

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_device_name,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage,
                                         logmgr_add_discretization_quantity,
                                         set_sim_state)

logger = logging.getLogger(__name__)


class SimError(RuntimeError):
    """Simple exception for fatal driver errors."""

    pass


def bump(nodes, t=0):
    """Create a bump."""
    actx = nodes[0].array_context

    source_center = np.array([0.2, 0.35, 0.1])[:len(nodes)]
    source_width = 0.05
    source_omega = 3

    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(len(nodes))
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, constant_cfl=False,
         casename=None, rst_filename=None, actx_class=PyOpenCLArrayContext):
    """Drive the example."""
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
        filename="wave-mpi.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = actx_class(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = actx_class(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

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
    t_final = 1
    if constant_cfl:
        sim_dt = None
        sim_cfl = 0.485
    else:
        sim_dt = 0.00491
        sim_cfl = None

    # i/o frequencies
    nhealth = 1
    nstatus = 1
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
                nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(discr.nodes(), actx)

    wave_speed = 1.0

    if rst_filename:
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
        rst_order = restart_data["order"]
        rst_state = restart_data["state"]
        if order == rst_order:
            current_state = rst_state
        else:
            old_discr = EagerDGDiscretization(actx, local_mesh, order=rst_order,
                                              mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(actx, discr.discr_from_dd("vol"),
                                                   old_discr.discr_from_dd("vol"))
            current_state = connection(rst_state)
    else:
        # Set the current state from time 0
        current_step = 0
        current_t = 0
        current_state = flat_obj_array(
            bump(nodes),
            [discr.zeros(actx) for i in range(dim)])

    vis_timer = None

    if logmgr:
        logmgr_add_discretization_quantity(
            logmgr, discr, lambda state: {"u": state[0]}, lambda _: "", "u")

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
            "CFL": sim_cfl,
        })
    if rank == 0:
        logger.info(init_message)

    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, discr) / wave_speed
    from grudge.op import nodal_min
    min_nodal_dt = actx.to_numpy(nodal_min(discr, "vol", nodal_dt))[()]

    def get_timestep_and_cfl(t, state):
        if constant_cfl:
            dt = sim_cfl*min_nodal_dt
            cfl = sim_cfl
        else:
            dt = sim_dt
            cfl = sim_dt/min_nodal_dt
        return get_next_timestep(t, t_final, dt), cfl

    # FIXME: Can this be done with logging?
    def write_status(dt, cfl):
        status_msg = f"------ {dt=}" if constant_cfl else f"----- {cfl=}"
        if rank == 0:
            logger.info(status_msg)

    def write_viz(step, t, state):
        viz_fields = [
            ("u", state[0]),
            ("v", state[1:]),
        ]
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname == rst_filename:
            if rank == 0:
                logger.info("Skipping overwrite of restart file.")
        else:
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
        if check_naninf_local(discr, "vol", state[0]) \
           or check_range_local(discr, "vol", state[0], -1, 1):
            health_error = True
            logger.info(f"{rank=}: Invalid state data found.")
        return health_error

    def pre_step(step, t, state):
        try:
            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)
            do_restart = check_step(step=step, interval=nrestart)
            do_viz = check_step(step=step, interval=nviz)

            if do_health:
                health_errors = global_reduce(health_check(state), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise SimError("Failed simulation health check.")

            dt, cfl = get_timestep_and_cfl(t, state)

            if do_status:
                write_status(dt, cfl)

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

        return state, dt

    def post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, state)
            logmgr.tick_after()
        return state

    def rhs(t, w):
        return wave_operator(discr, c=wave_speed, w=w)

    current_step, current_t, current_state = \
        advance_state(rhs=rhs, timestepper=timestepper,
                      pre_step_callback=pre_step,
                      post_step_callback=post_step,
                      step=current_step, t=current_t, state=current_state,
                      t_final=t_final)

    current_dt, current_cfl = get_timestep_and_cfl(current_t, current_state)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    write_status(dt=current_dt, cfl=current_cfl)
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
    parser = argparse.ArgumentParser(description="Wave")
    parser.add_argument("--mpi", action="store_true", help="run with MPI")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--constant-cfl", action="store_true",
        help="maintain a constant CFL")
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
        casename = "wave"

    if args.restart_file:
        rst_filename = args.restart_file
    else:
        rst_filename = None

    main_func(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
        constant_cfl=args.constant_cfl, casename=casename, rst_filename=rst_filename,
        actx_class=actx_class)


# vim: foldmethod=marker
