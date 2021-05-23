"""Demonstrate simple scalar lump advection."""

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
from pytools.obj_array import make_obj_array

from arraycontext.container.traversal import thaw
from arraycontext.impl.pyopencl import PyOpenCLArrayContext

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.discretization import DiscretizationCollection
from grudge.shortcuts import make_visualizer
import grudge.op as op


from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    generate_and_distribute_mesh,
    ExactSolutionMismatch
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import MulticomponentLump
from mirgecom.eos import IdealSingleGas


logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context):
    """Drive example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 3
    nel_1d = 16
    order = 3
    exittol = .09
    t_final = 0.01
    current_cfl = 1.0
    current_dt = .001
    current_t = 0
    constant_cfl = False
    nstatus = 1
    nviz = 1
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    box_ll = -5.0
    box_ur = 5.0

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nspecies = 4
    centers = make_obj_array([np.zeros(shape=(dim,)) for i in range(nspecies)])
    spec_y0s = np.ones(shape=(nspecies,))
    spec_amplitudes = np.ones(shape=(nspecies,))
    eos = IdealSingleGas()

    velocity = np.ones(shape=(dim,))

    initializer = MulticomponentLump(dim=dim, nspecies=nspecies,
                                     spec_centers=centers, velocity=velocity,
                                     spec_y0s=spec_y0s,
                                     spec_amplitudes=spec_amplitudes)
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}

    casename = "mixture-lump"

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, nelements_per_axis=(nel_1d,) * dim)
    local_mesh, global_nelements = generate_and_distribute_mesh(comm, generate_mesh)
    local_nelements = local_mesh.nelements

    dcoll = DiscretizationCollection(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(op.nodes(dcoll), actx)
    current_state = initializer(nodes)

    visualizer = make_visualizer(dcoll)
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

    get_timestep = partial(inviscid_sim_timestep, dcoll=dcoll, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_rhs(t, state):
        return euler_operator(dcoll, q=state, t=t,
                              boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        return sim_checkpoint(dcoll, visualizer, eos, q=state,
                              exact_soln=initializer, vizname=casename, step=step,
                              t=t, dt=dt, nstatus=nstatus, nviz=nviz,
                              exittol=exittol, constant_cfl=constant_cfl, comm=comm)

    try:
        (current_step, current_t, current_state) = \
            advance_state(rhs=my_rhs, timestepper=timestepper,
                          checkpoint=my_checkpoint,
                          get_timestep=get_timestep, state=current_state,
                          t=current_t, t_final=t_final)
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
    main()

# vim: foldmethod=marker
