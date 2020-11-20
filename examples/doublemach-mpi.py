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
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.euler import inviscid_operator
from mirgecom.artificial_viscosity import artificial_viscosity
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    create_parallel_grid,
    ExactSolutionMismatch,
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary, PrescribedBoundary
from mirgecom.initializers import DoubleMachReflection, SodShock1D
from mirgecom.eos import IdealSingleGas

from pytools.obj_array import obj_array_vectorize

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nel = (40,10)
    order = 3
    # tolerate large errors; case is unstable
    exittol = 2.0 #.2
    t_final = 1.0
    current_cfl = 0.1
    current_dt = .00031250
    current_t = 0
    eos = IdealSingleGas()
    initializer = DoubleMachReflection(dim)
    #initializer = SodShock1D(dim,x0=0.5)
    casename = "sod1d"
    #boundaries = {BTAG_ALL: AdiabaticSlipBoundary()}
    from grudge import sym
    boundaries = {sym.DTAG_BOUNDARY("ic1"): PrescribedBoundary(initializer),
                  sym.DTAG_BOUNDARY("ic2"): PrescribedBoundary(initializer),
                  sym.DTAG_BOUNDARY("ic3"): PrescribedBoundary(initializer),
                  sym.DTAG_BOUNDARY("wall"): AdiabaticSlipBoundary(),
                  sym.DTAG_BOUNDARY("out"): AdiabaticSlipBoundary()}
    constant_cfl = False
    nstatus = 10
    nviz = 25
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    box_ll = (0.0,0.0)
    box_ur = (4.0,1.0)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #from meshmode.mesh.generation import generate_regular_rect_mesh
    #generate_grid = partial(generate_regular_rect_mesh, a=box_ll,
    #                        b=box_ur, n=nel )
    #local_mesh, global_nelements = create_parallel_grid(comm, generate_grid)


    from meshmode.mesh.io import read_gmsh, generate_gmsh, ScriptWithFilesSource
    local_mesh = read_gmsh("doubleMach1.msh",force_ambient_dim=2)
    global_nelements = local_mesh.nelements

    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    current_state = initializer(0, nodes)

    #visualizer = make_visualizer(discr, discr.order + 3
    #                             if discr.dim == 2 else discr.order)
    visualizer = make_visualizer(discr, discr.order
                                 if discr.dim == 2 else discr.order)
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
        return (
               inviscid_operator(discr, q=state, t=t,boundaries=boundaries, eos=eos) 
               + artificial_viscosity(discr,t=t, r=state, eos=eos, boundaries=boundaries, alpha=4.0e-2)
                )
        #return (
        #       inviscid_operator(discr, q=state, t=t,boundaries=boundaries, eos=eos) 
        #       + artificial_viscosity(discr, r=state, eos=eos, boundaries=boundaries, alpha=1.0e-3)

    def my_checkpoint(step, t, dt, state):
        return sim_checkpoint(discr, visualizer, eos, q=state,
                              vizname=casename, step=step,
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
