"""Demonstrate double mach reflection."""

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
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.euler import inviscid_operator, split_conserved
from mirgecom.artificial_viscosity import (
    av_operator,
    smoothness_indicator
)
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
from mirgecom.boundary import AdiabaticSlipBoundary, PrescribedBoundary
from mirgecom.initializers import DoubleMachReflection
from mirgecom.eos import IdealSingleGas

logger = logging.getLogger(__name__)


def get_doublemach_mesh():
    """Generate or import a grid using `gmsh`.

    Input required:
        doubleMach.msh (read existing mesh)

    This routine will generate a new grid if it does
    not find the grid file (doubleMach.msh).
    """
    from meshmode.mesh.io import (
        read_gmsh,
        generate_gmsh,
        ScriptSource,
    )
    import os
    meshfile = "doubleMach.msh"
    if not os.path.exists(meshfile):
        mesh = generate_gmsh(
            ScriptSource("""
                x0=1.0/6.0;
                setsize=0.025;
                Point(1) = {0, 0, 0, setsize};
                Point(2) = {x0,0, 0, setsize};
                Point(3) = {4, 0, 0, setsize};
                Point(4) = {4, 1, 0, setsize};
                Point(5) = {0, 1, 0, setsize};
                Line(1) = {1, 2};
                Line(2) = {2, 3};
                Line(5) = {3, 4};
                Line(6) = {4, 5};
                Line(7) = {5, 1};
                Line Loop(8) = {-5, -6, -7, -1, -2};
                Plane Surface(8) = {8};
                Physical Surface('domain') = {8};
                Physical Curve('ic1') = {6};
                Physical Curve('ic2') = {7};
                Physical Curve('ic3') = {1};
                Physical Curve('wall') = {2};
                Physical Curve('out') = {5};
        """, "geo"), force_ambient_dim=2, dimensions=2, target_unit="M")
    else:
        mesh = read_gmsh(meshfile, force_ambient_dim=2)

    return mesh


@mpi_entry_point
def main(ctx_factory=cl.create_some_context):
    """Drive the example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue, allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )

    dim = 2
    order = 3
    # Too many steps for CI
    # t_final = 1.0e-2
    t_final = 1.0e-3
    current_cfl = 0.1
    current_dt = 1.0e-4
    current_t = 0
    eos = IdealSingleGas()
    initializer = DoubleMachReflection()
    casename = "doubleMach"

    boundaries = {
        DTAG_BOUNDARY("ic1"): PrescribedBoundary(initializer),
        DTAG_BOUNDARY("ic2"): PrescribedBoundary(initializer),
        DTAG_BOUNDARY("ic3"): PrescribedBoundary(initializer),
        DTAG_BOUNDARY("wall"): AdiabaticSlipBoundary(),
        DTAG_BOUNDARY("out"): AdiabaticSlipBoundary(),
    }
    constant_cfl = False
    nstatus = 10
    nviz = 100
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step

    s0 = -6.0
    kappa = 1.0
    alpha = 2.0e-2
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    gen_grid = partial(get_doublemach_mesh)

    local_mesh, global_nelements = generate_and_distribute_mesh(comm, gen_grid)

    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                                  mpi_communicator=comm)

    nodes = thaw(actx, discr.nodes())
    current_state = initializer(nodes)

    visualizer = make_visualizer(discr,
                                 discr.order if discr.dim == 2 else discr.order)
    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(
        dim=dim,
        order=order,
        nelements=local_nelements,
        global_nelements=global_nelements,
        dt=current_dt,
        t_final=t_final,
        nstatus=nstatus,
        nviz=nviz,
        cfl=current_cfl,
        constant_cfl=constant_cfl,
        initname=initname,
        eosname=eosname,
        casename=casename,
    )
    if rank == 0:
        logger.info(init_message)

    get_timestep = partial(
        inviscid_sim_timestep,
        discr=discr,
        t=current_t,
        dt=current_dt,
        cfl=current_cfl,
        eos=eos,
        t_final=t_final,
        constant_cfl=constant_cfl,
    )

    def my_rhs(t, state):
        return inviscid_operator(
            discr, q=state, t=t, boundaries=boundaries, eos=eos
        ) + av_operator(
            discr, q=state, boundaries=boundaries,
            boundary_kwargs={"t": t, "eos": eos}, alpha=alpha,
            s0=s0, kappa=kappa
        )

    def my_checkpoint(step, t, dt, state):
        cv = split_conserved(dim, state)
        tagged_cells = smoothness_indicator(discr, cv.mass, s0=s0, kappa=kappa)
        viz_fields = [("tagged cells", tagged_cells)]
        return sim_checkpoint(
            discr,
            visualizer,
            eos,
            q=state,
            vizname=casename,
            step=step,
            t=t,
            dt=dt,
            nstatus=nstatus,
            nviz=nviz,
            constant_cfl=constant_cfl,
            comm=comm,
            viz_fields=viz_fields,
            overwrite=True,
        )

    try:
        (current_step, current_t, current_state) = advance_state(
            rhs=my_rhs,
            timestepper=timestepper,
            checkpoint=my_checkpoint,
            get_timestep=get_timestep,
            state=current_state,
            t=current_t,
            t_final=t_final,
        )
    except ExactSolutionMismatch as ex:
        current_step = ex.step
        current_t = ex.t
        current_state = ex.state

    #    if current_t != checkpoint_t:
    if rank == 0:
        logger.info("Checkpointing final state ...")
        my_checkpoint(
            current_step,
            t=current_t,
            dt=(current_t - checkpoint_t),
            state=current_state,
        )

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

# vim: foldmethod=marker
