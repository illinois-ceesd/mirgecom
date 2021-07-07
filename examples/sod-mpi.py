"""Demonstrate Sod's 1D shock example."""

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
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import SodShock1D
from mirgecom.eos import IdealSingleGas

from logpyle import set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state
)

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename="sod1d",
         rst_step=None, rst_name=None):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 1
    nel_1d = 24
    order = 1
    # tolerate large errors; case is unstable
    t_final = 0.01
    current_cfl = 1.0
    current_dt = .0001
    current_t = 0
    eos = IdealSingleGas()
    initializer = SodShock1D(dim=dim)
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    constant_cfl = False
    nstatus = 10
    nrestart = 5
    nviz = 10
    nhealth = 10
    rank = 0
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_step:  # read the grid from restart data
        rst_fname = rst_pattern.format(cname=casename, step=rst_step, rank=rank)

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_fname)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == num_parts
    else:  # generate the grid from scratch
        from meshmode.mesh.generation import generate_regular_rect_mesh
        box_ll = -5.0
        box_ur = 5.0
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    if rst_step:
        current_t = restart_data["t"]
        current_step = rst_step
        current_state = restart_data["state"]
    else:
        # Set the current state from time 0
        current_state = initializer(nodes)

    visualizer = make_visualizer(discr)

    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_graceful_exit(cv, step, t, do_viz=False, do_restart=False, message=None):
        if rank == 0:
            logger.info("Errors detected; attempting graceful exit.")
        if do_viz:
            my_write_viz(cv, step, t)
        if do_restart:
            my_write_restart(state=cv, step=step, t=t)
        if message is None:
            message = "Fatal simulation errors detected."
        raise RuntimeError(message)

    def my_write_viz(cv, step, t, dv=None, exact=None, resid=None):
        viz_fields = [("cv", cv)]
        if dv is not None:
            viz_fields.append(("dv", dv))
        if exact is not None:
            viz_fields.append(("exact_soln", exact))
        if resid is not None:
            viz_fields.append(("residual", resid))
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True)

    def my_write_restart(state, step, t):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        rst_data = {
            "local_mesh": local_mesh,
            "state": state,
            "t": t,
            "step": step,
            "global_nelements": global_nelements,
            "num_parts": num_parts
        }
        from mirgecom.restart import write_restart_file
        write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state, dv, exact):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure) \
           or check_range_local(discr, "vol", dv.pressure, .09, 1.1):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        from mirgecom.simutil import compare_fluid_solutions
        component_errors = compare_fluid_solutions(discr, state, exact)
        exittol = .09
        if max(component_errors) > exittol:
            health_error = True
            if rank == 0:
                logger.info("Solution diverged from exact soln.")

        return health_error

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_pre_step(step, t, dt, state):
        dv = None
        exact = None
        pre_step_errors = False

        if logmgr:
            logmgr.tick_before()

        from mirgecom.simutil import check_step
        do_viz = check_step(step=step, interval=nviz)
        do_restart = check_step(step=step, interval=nrestart)
        do_health = check_step(step=step, interval=nhealth)
        do_status = check_step(step=step, interval=nstatus)

        if step == rst_step:  # don't do viz or restart @ restart
            do_viz = False
            do_restart = False

        if do_health:
            dv = eos.dependent_vars(state)
            exact = initializer(x_vec=nodes, eos=eos, t=t)
            local_health_error = my_health_check(state, dv, exact)
            health_errors = False
            if comm is not None:
                health_errors = comm.allreduce(local_health_error, op=MPI.LOR)
            if health_errors and rank == 0:
                logger.info("Fluid solution failed health check.")
            pre_step_errors = pre_step_errors or health_errors

        if do_restart:
            my_write_restart(state, step, t)

        if do_viz:
            if dv is None:
                dv = eos.dependent_vars(state)
            if exact is None:
                exact = initializer(x_vec=nodes, eos=eos, t=t)
            resid = state - exact
            my_write_viz(cv=state, dv=dv, step=step, t=t,
                         exact=exact, resid=resid)

        if do_status:
            if exact is None:
                exact = initializer(x_vec=nodes, eos=eos, t=t)
            from mirgecom.simutil import compare_fluid_solutions
            component_errors = compare_fluid_solutions(discr, state, exact)
            status_msg = (
                "------- errors="
                + ", ".join("%.3g" % en for en in component_errors))
            if rank == 0:
                logger.info(status_msg)

        if pre_step_errors:
            my_graceful_exit(cv=state, step=step, t=t,
                             do_viz=(not do_viz), do_restart=(not do_restart),
                             message="Error detected at prestep, exiting.")

        return state, dt

    get_timestep = partial(inviscid_sim_timestep, discr=discr,
                           cfl=current_cfl, eos=eos, t_final=t_final,
                           constant_cfl=constant_cfl)

    def my_rhs(t, state):
        return euler_operator(discr, cv=state, t=t,
                              boundaries=boundaries, eos=eos)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step, dt=current_dt,
                      post_step_callback=my_post_step,
                      get_timestep=get_timestep, state=current_state,
                      t=current_t, t_final=t_final, eos=eos, dim=dim)

    finish_tol = 1e-16
    if np.abs(current_t - t_final) > finish_tol:
        my_graceful_exit(cv=current_state, step=current_step, t=current_t,
                         do_viz=True, do_restart=True,
                         message="Simulation timestepping did not complete.")

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    final_dv = eos.dependent_vars(current_state)
    final_exact = initializer(x_vec=nodes, eos=eos, t=current_t)
    final_resid = current_state - final_exact
    my_write_viz(cv=current_state, dv=final_dv, exact=final_exact,
                 resid=final_resid, step=current_step, t=current_t)
    my_write_restart(current_state, current_step, current_t)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main(use_leap=False)

# vim: foldmethod=marker
