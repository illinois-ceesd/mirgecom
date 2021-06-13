"""Demonstrate simple mass lump advection."""

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
from mirgecom.initializers import Lump
from mirgecom.eos import IdealSingleGas


logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_leap=False):
    """Drive example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 3
    nel_1d = 16
    order = 3
    t_final = 0.01
    current_cfl = 1.0
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[:dim] = 1.0
    current_dt = .001
    current_t = 0
    eos = IdealSingleGas()
    initializer = Lump(dim=dim, center=orig, velocity=vel)
    casename = "lump"
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    constant_cfl = False
    nstatus = 1
    nhealth = 1
    nviz = 1
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

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
            exact_mix = initializer(x_vec=nodes, eos=eos, t=t)
            component_errors = compare_fluid_solutions(discr, state, exact_mix)
            resid = state - exact_mix
            io_fields = [
                ("cv", state),
                ("dv", dv),
                ("exact_mix", exact_mix),
                ("resid", resid)
            ]

        if do_status:  # This is bad, logging already completely replaces this
            from mirgecom.io import make_status_message
            status_msg = make_status_message(discr=discr, t=t, step=step, dt=dt,
                                             cfl=current_cfl, dependent_vars=dv)
            status_msg += (
                "\n------- errors="
                + ", ".join("%.3g" % en for en in component_errors))
            if rank == 0:
                logger.info(status_msg)

        errored = False
        if do_health:
            from mirgecom.simutil import check_naninf_local, check_range_local
            if check_naninf_local(discr, "vol", dv.pressure) \
               or check_range_local(discr, "vol", dv.pressure, .9, 1.1):
                errored = True
                message = "Invalid pressure data found.\n"
            exittol = .09
            if max(component_errors) > exittol:
                errored = True
                message += "Solution diverged from exact_mix.\n"
            comm = discr.mpi_communicator
            if comm is not None:
                errored = comm.allreduce(errored, op=MPI.LOR)
            if errored:
                if rank == 0:
                    logger.info("Fluid solution failed health check.")
                logger.info(message)   # do this on all ranks

        if do_viz or errored > 0:
            from mirgecom.simutil import write_visfile
            write_visfile(discr, io_fields, visualizer, vizname=casename,
                          step=step, t=t, overwrite=True)

        if errored:
            raise RuntimeError("Error detected by user checkpoint, exiting.")

        return state

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_checkpoint,
                      get_timestep=get_timestep, state=current_state,
                      t=current_t, t_final=t_final, eos=eos, dim=dim)

    #    if current_t != checkpoint_t:
    if rank == 0:
        logger.info("Checkpointing final state ...")
    my_checkpoint(current_step, t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main(use_leap=False)

# vim: foldmethod=marker
