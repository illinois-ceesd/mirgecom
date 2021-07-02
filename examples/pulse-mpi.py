"""Demonstrate acoustic pulse, and adiabatic slip wall."""

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
from mirgecom.mpi import mpi_entry_point
import numpy as np
from functools import partial
import pyopencl as cl
import pyopencl.tools as cl_tools

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    generate_and_distribute_mesh,
    sim_checkpoint,
    ExactSolutionMismatch,
)

from mirgecom.viscous import get_viscous_timestep

from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import (
    Lump,
    AcousticPulse
)
from mirgecom.eos import IdealSingleGas


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_leap=False):
    """Drive the example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    logger = logging.getLogger(__name__)

    dim = 2
    nel_1d = 16
    order = 1
    exittol = 2e-2
    exittol = 100.0
    t_final = .5
    current_cfl = 0.38
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    #    vel[:dim] = 1.0
    current_dt = .01
    current_t = 0
    eos = IdealSingleGas()
    initializer = Lump(dim=dim, center=orig, velocity=vel, rhoamp=0.0)
    casename = "pulse"
    wall = AdiabaticSlipBoundary()
    boundaries = {BTAG_ALL: wall}
    constant_cfl = True
    nstatus = 10
    nviz = 10
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    box_ll = -0.5
    box_ur = 0.5

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    num_parts = nproc

    from meshmode.mesh.generation import generate_regular_rect_mesh
    if num_parts > 1:
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                                b=(box_ur,) * dim,
                                nelements_per_axis=(nel_1d,) * dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
    else:
        local_mesh = generate_regular_rect_mesh(
            a=(box_ll,) * dim, b=(box_ur,) * dim, nelements_per_axis=(nel_1d,) * dim
        )
        global_nelements = local_mesh.nelements
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    uniform_state = initializer(nodes)
    acoustic_pulse = AcousticPulse(dim=dim, amplitude=1.0, width=.1,
                                   center=orig)
    current_state = acoustic_pulse(x_vec=nodes, cv=uniform_state, eos=eos)

    visualizer = make_visualizer(discr)

    initname = "pulse"
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

    def my_rhs(t, state):
        return euler_operator(discr, cv=state, t=t,
                              boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        t_remaining = max(0, t_final - t)
        checkpoint_cfl = current_cfl
        viz_fields = []

        if constant_cfl is True:
            dt = (
                current_cfl * get_viscous_timestep(discr=discr, eos=eos, cv=state)
            )
            viz_fields.append(("local_dt", dt))
            from grudge.op import nodal_min
            dt = nodal_min(discr, "vol", dt)
        else:
            from mirgecom.viscous import get_viscous_cfl
            cfl_field = get_viscous_cfl(discr, eos, dt, state)
            viz_fields.append(("cfl", cfl_field))
            from grudge.op import nodal_max
            checkpoint_cfl = nodal_max(discr, "vol", cfl_field)

        dt = min(dt, t_remaining)
        return sim_checkpoint(discr, visualizer, eos, cv=state,
                              vizname=casename, step=step, cfl=checkpoint_cfl,
                              t=t, dt=dt, nstatus=nstatus, nviz=nviz,
                              exittol=exittol, constant_cfl=constant_cfl,
                              comm=comm, viz_fields=viz_fields)

    try:
        (current_step, current_t, current_state) = \
            advance_state(rhs=my_rhs, timestepper=timestepper,
                checkpoint=my_checkpoint, state=current_state,
                dt=current_dt, t=current_t, t_final=t_final)
    except ExactSolutionMismatch as ex:
        current_step = ex.step
        current_t = ex.t
        current_state = ex.state

    #    if current_t != checkpoint_t:
    if rank == 0:
        logger.info("Checkpointing final state ...")
    my_checkpoint(current_step, t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state)

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    main(use_leap=False)

# vim: foldmethod=marker
