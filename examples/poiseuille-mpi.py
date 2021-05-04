"""Demonstrate a planar Poiseuille flow example."""

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
from pytools.obj_array import make_obj_array
from functools import partial

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.symbolic import DTAG_BOUNDARY

from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    generate_and_distribute_mesh,
    ExactSolutionMismatch,
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedViscousBoundary,
    IsothermalNoSlipBoundary
)
from mirgecom.fluid import (
    split_conserved,
    join_conserved
)
from mirgecom.transport import SimpleTransport
from mirgecom.eos import IdealSingleGas


logger = logging.getLogger(__name__)


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None):
    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_profiling=False, use_logmgr=False):
    """Drive the example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue, allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )

    dim = 2
    order = 1
    exittol = 1.0
    t_final = 1e-7
    current_cfl = 1.0
    current_dt = 1e-8
    current_t = 0
    casename = "poiseuille"
    constant_cfl = False
    nstatus = 1
    nviz = 1
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    left_boundary_location = 0
    right_boundary_location = 0.1
    box_ll = (left_boundary_location, 0.0)
    box_ur = (right_boundary_location, 0.02)
    npts_axis = (50, 30)
    rank = comm.Get_rank()

    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")

    generate_mesh = partial(_get_box_mesh, 2, a=box_ll, b=box_ur, n=npts_axis)
    local_mesh, global_nelements = generate_and_distribute_mesh(comm, generate_mesh)
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    base_pressure = 100000.0
    pressure_ratio = 1.001

    def poiseuille_soln(nodes, eos, q=None, **kwargs):
        dim = len(nodes)
        x0 = left_boundary_location
        xmax = right_boundary_location
        xlen = xmax - x0
        p0 = base_pressure
        p1 = pressure_ratio*p0
        p_x = p1 + p0*(1 - pressure_ratio)*(nodes[0] - x0)/xlen
        ke = 0
        mass = nodes[0] + 1.0 - nodes[0]
        momentum = make_obj_array([0*mass for i in range(dim)])
        if q is not None:
            cv = split_conserved(dim, q)
            mass = cv.mass
            momentum = cv.momentum
            ke = .5*np.dot(cv.momentum, cv.momentum)/cv.mass
        energy_bc = p_x / (eos.gamma() - 1) + ke
        return join_conserved(dim, mass=mass, energy=energy_bc,
                              momentum=momentum)

    initializer = poiseuille_soln
    boundaries = {DTAG_BOUNDARY("-1"): PrescribedViscousBoundary(q_func=initializer),
                  DTAG_BOUNDARY("+1"): PrescribedViscousBoundary(q_func=initializer),
                  DTAG_BOUNDARY("-2"): IsothermalNoSlipBoundary(),
                  DTAG_BOUNDARY("+2"): IsothermalNoSlipBoundary()}
    eos = IdealSingleGas(transport_model=SimpleTransport(viscosity=1.0))

    current_state = initializer(nodes, eos)

    vis_timer = None

    visualizer = make_visualizer(discr, order)

    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_rhs(t, state):
        return ns_operator(discr, eos=eos, boundaries=boundaries, q=state, t=t)

    def my_checkpoint(step, t, dt, state):
        return sim_checkpoint(discr, visualizer, eos, q=state,
                              vizname=casename, step=step,
                              t=t, dt=dt, nstatus=nstatus, nviz=nviz,
                              exittol=exittol, constant_cfl=constant_cfl, comm=comm,
                              vis_timer=vis_timer, overwrite=True)

    try:
        (current_step, current_t, current_state) = \
            advance_state(rhs=my_rhs, timestepper=timestepper,
                          checkpoint=my_checkpoint,
                          get_timestep=get_timestep, state=current_state,
                          t=current_t, t_final=t_final, eos=eos,
                          dim=dim)
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
    use_profiling = False
    use_logging = False

    main(use_profiling=use_profiling, use_logmgr=use_logging)

# vim: foldmethod=marker
