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

from mirgecom.euler import (
    inviscid_operator,
    split_conserved,
    get_inviscid_timestep,
    InviscidTimestepError,
    get_inviscid_vis_fields,
)
from mirgecom.simutil import (
    create_parallel_grid,
    check_step,
)
from mirgecom.io import (
    make_init_message,
    write_visualization_file,
)
from mirgecom.mpi import mpi_entry_point, comm_any

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import Vortex2D
from mirgecom.eos import IdealSingleGas

from logpyle import IntervalTimer

from mirgecom.logging_quantities import initialize_logmgr, logmgr_add_device_name
from mirgecom.euler import logmgr_add_inviscid_quantities


logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_profiling=False, use_logmgr=False):
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
    exittol = .09
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
    nviz = 10
    rank = 0
    current_step = 0
    timestepper = rk4_step
    box_ll = -5.0
    box_ur = 5.0

    rank = comm.Get_rank()

    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_grid = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, n=(nel_1d,) * dim)
    local_mesh, global_nelements = create_parallel_grid(comm, generate_grid)
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    current_state = initializer(nodes)

    vis_timer = None

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_inviscid_quantities(logmgr, discr, eos)

        logmgr.add_watches(["step.max", "t_step.max", "t_log.max",
                            "min_temperature", "L2_norm_momentum1"])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr, order + 3 if dim == 2 else order)

    init_message = make_init_message(dim=dim, order=order, casename=casename,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements)
    if rank == 0:
        logger.info(init_message)

    def get_timestep(step, t, state):
        try:
            dt = get_inviscid_timestep(discr=discr, q=state, cfl=current_cfl,
                eos=eos) if constant_cfl else current_dt
        except InviscidTimestepError:
            write_vis(step, t, state)
            raise
        return min(dt, t_final - t)

    def rhs(t, state):
        return inviscid_operator(discr, q=state, t=t,
                                 boundaries=boundaries, eos=eos)

    def write_vis(step, t, state):
        io_fields = get_inviscid_vis_fields(dim, state, eos)
        io_fields.append(
            ("exact_cv", split_conserved(dim, initializer(nodes, t=t))))
        return write_visualization_file(visualizer, fields=io_fields,
                    basename=casename, step=step, t=t, comm=comm, timer=vis_timer)

    def checkpoint(step, t, dt, state):
        done = t >= t_final
        exact_state = initializer(nodes, t=t)
        if comm_any(comm, discr.norm(state - exact_state, np.inf) > exittol):
            write_vis(step, t, state)
            raise RuntimeError(f"Exact solution mismatch at step {step}.")
        if check_step(step, nviz) or done:
            write_vis(step, t, state)
        return done

    if current_step == 0:
        dt = get_timestep(0, current_t, current_state)
        done = checkpoint(0, current_t, dt, current_state)
        assert not done

    if rank == 0:
        logger.info("Timestepping started.")

    (current_step, current_t, current_state) = \
        advance_state(rhs=rhs, timestepper=timestepper, checkpoint=checkpoint,
            get_timestep=get_timestep, state=current_state, t=current_t,
            logmgr=logmgr)

    if rank == 0:
        logger.info("Timestepping finished.")

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    use_profiling = True
    use_logging = True

    main(use_profiling=use_profiling, use_logmgr=use_logging)

# vim: foldmethod=marker
