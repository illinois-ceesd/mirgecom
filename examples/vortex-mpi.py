"""Demonstrate the isentropic vortex example."""

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

from mirgecom.profiling import PyOpenCLProfilingArrayContext

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
from mirgecom.initializers import Vortex2D
from mirgecom.eos import IdealSingleGas

from logpyle import IntervalTimer
from mirgecom.euler import extract_vars_for_logging, units_for_logging

from mirgecom.logging_quantities import (initialize_logmgr,
    logmgr_add_many_discretization_quantities, logmgr_add_device_name,
    logmgr_add_device_memory_usage)


logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_profiling=False, use_logmgr=False,
         use_leap=False):
    """Drive the example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    logmgr = initialize_logmgr(use_logmgr,
        filename="vortex.sqlite", mode="wu", mpi_comm=comm)

    cl_ctx = ctx_factory()
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

    dim = 2
    nel_1d = 16
    order = 3
    exittol = .1
    t_final = 0.1
    current_cfl = 1.0
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[:dim] = 1.0
    current_dt = .001
    current_t = 0
    eos = IdealSingleGas()
    initializer = Vortex2D(center=orig, velocity=vel)
    casename = "vortex"
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    constant_cfl = False
    nstatus = 10
    nviz = 10
    nhealth = 10
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    box_ll = -5.0
    box_ur = 5.0

    rank = comm.Get_rank()

    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, nelements_per_axis=(nel_1d,) * dim)
    local_mesh, global_nelements = generate_and_distribute_mesh(comm, generate_mesh)
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    current_state = initializer(nodes)

    vis_timer = None

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

        logmgr.add_watches(["step.max", "t_step.max",
                            "min_pressure", "max_pressure"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

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

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_rhs(t, state):
        return euler_operator(discr, cv=state, t=t,
                              boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        from mirgecom.simutil import check_step
        do_status = check_step(step=step, interval=nstatus)
        do_viz = check_step(step=step, interval=nviz)
        do_health = check_step(step=step, interval=nhealth)

        if do_status or do_viz or do_health:
            from mirgecom.simutil import compare_fluid_solutions
            dv = eos.dependent_vars(state)
            vortex_exact = initializer(x_vec=nodes, eos=eos, t=t)
            component_errors = compare_fluid_solutions(discr, state, vortex_exact)
            resid = state - vortex_exact
            io_fields = [
                ("cv", state),
                ("dv", dv),
                ("vortex_exact", vortex_exact),
                ("resid", resid)
            ]

        if do_status:
            status_msg = (
                "\n------- errors="
                + ", ".join("%.3g" % en for en in component_errors))
            if rank == 0:
                logger.info(status_msg)

        errored = False
        if do_health:
            from mirgecom.simutil import check_naninf_local, check_range_local
            if check_naninf_local(discr, "vol", dv.pressure) \
               or check_range_local(discr, "vol", dv.pressure, .2, 1.02):
                errored = True
                message = "Invalid pressure data found.\n"
            if np.max(component_errors) > exittol:
                errored = True
                message += "Solution errors exceed tolerance.\n"
            errored = discr.mpi_communicator.allreduce(errored, op=MPI.LOR)
            if errored:
                if rank == 0:
                    logger.info("Fluid solution failed health check.")
                logger.info(message)   # do this on all ranks

        if do_viz or errored:
            from mirgecom.simutil import sim_visualization
            sim_visualization(discr, io_fields, visualizer, vizname=casename,
                              step=step, t=t, overwrite=True)

        if errored:
            raise RuntimeError("Error detected by user checkpoint, exiting.")

        return state

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_checkpoint, logmgr=logmgr,
                      get_timestep=get_timestep, state=current_state,
                      t=current_t, t_final=t_final, eos=eos, dim=dim)

    if rank == 0:
        logger.info("Checkpointing final state ...")
    my_checkpoint(current_step, t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    use_profiling = True
    use_logging = True
    use_leap = False

    main(use_profiling=use_profiling, use_logmgr=use_logging, use_leap=use_leap)

# vim: foldmethod=marker
