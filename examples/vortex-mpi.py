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
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from mpi4py import MPI

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.io import (
    make_io_fields,
    make_status_message,
    make_init_message,
    make_output_dump,
)

from mirgecom.euler import (
    get_inviscid_timestep,
    get_inviscid_cfl,
    inviscid_operator
)
from mirgecom.checkstate import compare_states
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import Vortex2D
from mirgecom.eos import IdealSingleGas


def check_step(step, n):
    if n == 0:
        return True
    elif n < 0:
        return False
    elif step % n == 0:
        return True
    return False


def main(ctx_factory=cl.create_some_context):

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    logger = logging.getLogger(__name__)

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
    casename = 'vortex'
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    constant_cfl = False
    nstatus = 10
    nviz = 10
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    box_ll = -5.0
    box_ur = 5.0

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    num_parts = nproc

    from meshmode.distributed import (
        MPIMeshDistributor,
        get_partition_by_pymetis,
    )

    mesh_dist = MPIMeshDistributor(comm)

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh

        mesh = generate_regular_rect_mesh(
            a=(box_ll,) * dim, b=(box_ur,) * dim, n=(nel_1d,) * dim
        )

        logging.info(f"Total {dim}d elements: {mesh.nelements}")

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    istate = initializer(0, nodes)
    current_state = istate

    visualizer = make_visualizer(discr, discr.order + 3
                                 if discr.dim == 2 else discr.order)
    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_mesh.nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_timestep(state):
        mydt = current_dt
        if constant_cfl is True:
            mydt = get_inviscid_timestep(discr=discr, q=state,
                                         cfl=current_cfl, eos=eos)
        if (current_t + mydt) > t_final:
            mydt = t_final - current_t
        return mydt

    def my_rhs(t, state):
        return inviscid_operator(discr, q=state, t=t, boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        # This stuff should be done by any/all checkpoint
        current_t = t
        current_step = step
        current_state = state
        current_dt = dt
        checkpoint_status = 0

        if constant_cfl is False:
            current_cfl = get_inviscid_cfl(discr=discr, q=state,
                                           eos=eos, dt=current_dt)

        # The rest of this checkpoint routine is customization
        do_status = check_step(step=step, n=nstatus)
        do_viz = check_step(step=step, n=nviz)
        if do_viz is False and do_status is False:
            return checkpoint_status

        dv = eos(q=current_state)
        expected_state = initializer(t=current_t, x_vec=nodes, eos=eos)

        if do_status is True:
            statusmesg = make_status_message(t=current_t, step=current_step,
                                             dt=current_dt,
                                             cfl=current_cfl, dv=dv)
            max_errors = compare_states(state, expected_state)
            statusmesg += f"\n------   Err({max_errors})"
            if rank == 0:
                logger.info(statusmesg)

            maxerr = np.max(max_errors)
            if maxerr > exittol:
                logger.error("Solution failed to follow expected result.")
                checkpoint_status = 1

        if do_viz:
            checkpoint_t = current_t
            io_fields = make_io_fields(dim, state, dv, eos)
            io_fields.append(("exact_soln", expected_state))
            result_resid = state - expected_state
            io_fields.append(("residual", result_resid))
            make_output_dump(visualizer, basename=casename, io_fields=io_fields,
                             comm=comm, step=step, t=checkpoint_t, overwrite=True)

        return checkpoint_status

    comm.Barrier()
    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper, checkpoint=my_checkpoint,
                    get_timestep=my_timestep, state=current_state,
                    t=current_t, t_final=t_final)

    comm.Barrier()

    if current_t != checkpoint_t:
        if rank == 0:
            logger.info("Checkpointing final state ...")
        my_checkpoint(current_step, t=current_t,
                      dt=(current_t - checkpoint_t),
                      state=current_state)

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

# vim: foldmethod=marker
