"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

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
from functools import partial
from mpi4py import MPI

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.euler import inviscid_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint
)
from mirgecom.io import (
    make_init_message,
)
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedBoundary,
    AdiabaticSlipBoundary,
    DummyBoundary
)
from mirgecom.initializers import (
    Lump,
    make_pulse
)
from mirgecom.eos import IdealSingleGas


def get_pseudo_y0_mesh():
    """Generate or import a grid using `gmsh`.

    Input required:
        data/pseudoY0.brep  (for mesh gen)
        -or-
        data/pseudoY0.msh   (read existing mesh)

    This routine will generate a new grid if it does
    not find the grid file (data/pseudoY0.msh), but
    note that if the grid is generated in millimeters,
    then the solution initialization and BCs need to be
    adjusted or the grid needs to be scaled up to meters
    before being used with the current main driver in this
    example.
    """
    from meshmode.mesh.io import read_gmsh, generate_gmsh, ScriptWithFilesSource
    import os
    if os.path.exists("data/pseudoY0.msh") is False:
        mesh = generate_gmsh(
            ScriptWithFilesSource("""
            Merge "data/pseudoY0.brep";
            Mesh.CharacteristicLengthMin = 1;
            Mesh.CharacteristicLengthMax = 10;
            Mesh.ElementOrder = 2;
            Mesh.CharacteristicLengthExtendFromBoundary = 0;

            // Inside and end surfaces of nozzle/scramjet
            Field[1] = Distance;
            Field[1].NNodesByEdge = 100;
            Field[1].FacesList = {5,7,8,9,10};
            Field[2] = Threshold;
            Field[2].IField = 1;
            Field[2].LcMin = 1;
            Field[2].LcMax = 10;
            Field[2].DistMin = 0;
            Field[2].DistMax = 20;

            // Edges separating surfaces with boundary layer
            // refinement from those without
            // (Seems to give a smoother transition)
            Field[3] = Distance;
            Field[3].NNodesByEdge = 100;
            Field[3].EdgesList = {5,10,14,16};
            Field[4] = Threshold;
            Field[4].IField = 3;
            Field[4].LcMin = 1;
            Field[4].LcMax = 10;
            Field[4].DistMin = 0;
            Field[4].DistMax = 20;

            // Min of the two sections above
            Field[5] = Min;
            Field[5].FieldsList = {2,4};

            Background Field = 5;
        """, ["data/pseudoY0.brep"]), 3, target_unit='MM')
    else:
        mesh = read_gmsh("data/pseudoY0.msh")

    return mesh


def main(ctx_factory=cl.create_some_context):
    """Drive the Y0 example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    logger = logging.getLogger(__name__)

    dim = 3
    order = 1
    exittol = .09
    t_final = 0.1
    current_cfl = 1.0
    vel_init = np.zeros(shape=(dim,))
    vel_inflow = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    orig[0] = 0.83
    orig[2] = 0.001
    #    vel[0] = 340.0
    vel_inflow[0] = 100.0  # m/s
    current_dt = 1e-8
    current_t = 0
    eos = IdealSingleGas()
    bulk_init = Lump(numdim=dim, rho0=1.225, p0=100000.0,
                     center=orig, velocity=vel_init, rhoamp=0.0)
    inflow_init = Lump(numdim=dim, rho0=1.225, p0=200000.0,
                       center=orig, velocity=vel_inflow, rhoamp=0.0)

    casename = "pseudoY0"
    wall = AdiabaticSlipBoundary()
    from grudge import sym
    #    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    boundaries = {sym.DTAG_BOUNDARY("Inflow"): PrescribedBoundary(inflow_init),
                  sym.DTAG_BOUNDARY("Outflow"): DummyBoundary,
                  sym.DTAG_BOUNDARY("Wall"): wall}
    constant_cfl = False
    nstatus = 10
    nviz = 10
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    num_parts = nproc

    from meshmode.distributed import (
        MPIMeshDistributor,
        get_partition_by_pymetis,
    )

    global_nelements = 0
    local_nelements = 0

    if nproc > 1:
        mesh_dist = MPIMeshDistributor(comm)
        if mesh_dist.is_mananger_rank():

            mesh = get_pseudo_y0_mesh()
            global_nelements = mesh.nelements
            logging.info(f"Total {dim}d elements: {global_nelements}")
            logging.info(f"Grid BTAGS: {mesh.boundary_tags}")
            #            print(f"btags = {mesh.boundary_tags}")

            logging.info("Partitioning grid.")
            part_per_element = get_partition_by_pymetis(mesh, num_parts)
            logging.info("Sending grid partitions")

            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
            del mesh
            logging.info("Grid partitions sent.")

        else:
            local_mesh = mesh_dist.receive_mesh_part()

        comm.Barrier()

        if rank == 0:
            logging.info("Grid distributed.")
    else:
        logging.info("Reading grid.")
        local_mesh = get_pseudo_y0_mesh()
        global_nelements = local_mesh.nelements
        logging.info("Done. Reading grid.")

    local_nelements = local_mesh.nelements

    comm.Barrier()
    if rank == 0:
        logging.info("Making discretization")
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    comm.Barrier()
    if rank == 0:
        logging.info("Initializing soln.")
    current_state = bulk_init(0, nodes)
    #    current_state = set_uniform_solution(t=0.0, x_vec=nodes)
    comm.Barrier()
    if rank == 0:
        logging.info("Adding pulse.")
    current_state[1] = current_state[1] + make_pulse(amp=50000.0, w=.002,
                                                     r0=orig, r=nodes)

    visualizer = make_visualizer(discr, discr.order + 3
                                 if discr.dim == 2 else discr.order)
    #    initname = initializer.__class__.__name__
    initname = "pseudoY0"
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
        return inviscid_operator(discr, q=state, t=t,
                                 boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        if rank == 0:
            logger.info(f"Checkpoint: {step}.")
        return sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                              logger=logger, q=state, vizname=casename, step=step,
                              t=t, dt=dt, nstatus=nstatus, nviz=nviz,
                              exittol=exittol, constant_cfl=constant_cfl, comm=comm)

    comm.Barrier()
    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper, checkpoint=my_checkpoint,
                    get_timestep=get_timestep, state=current_state,
                    t=current_t, t_final=t_final)

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
