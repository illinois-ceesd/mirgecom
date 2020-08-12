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
    check_step,
    make_status_message,
)
from mirgecom.io import (
    make_init_message,
    make_io_fields,
    make_output_dump,
)
from mirgecom.checkstate import compare_states
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
# from mirgecom.boundary import DummyBoundary
# from mirgecom.initializers import Uniform
from mirgecom.initializers import Lump
from mirgecom.eos import IdealSingleGas
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)


def sim_checkpoint(discr, visualizer, eos, logger, q, vizname, exact_soln=None,
                   step=0, t=0, dt=0, cfl=1.0, nstatus=-1, nviz=-1, exittol=1e-16,
                   constant_cfl=False, comm=None):
    r"""
    Checkpointing utility for runs with known exact solution generator
    """

    do_viz = check_step(step=step, n=nviz)
    do_status = check_step(step=step, n=nstatus)
    if do_viz is False and do_status is False:
        return 0

    actx = q[0].array_context
    nodes = thaw(actx, discr.nodes())
    rank = 0

    if comm is not None:
        rank = comm.Get_rank()
    checkpoint_status = 0

    dv = eos(q=q)

    have_exact = False

    if ((do_status is True or do_viz is True) and exact_soln is not None):
        have_exact = True
        expected_state = exact_soln(t=t, x_vec=nodes, eos=eos)

    if do_status is True:
        #        if constant_cfl is False:
        #            current_cfl = get_inviscid_cfl(discr=discr, q=q,
        #                                           eos=eos, dt=dt)
        statusmesg = make_status_message(t=t, step=step, dt=dt,
                                         cfl=cfl, dv=dv)
        if have_exact is True:
            max_errors = compare_states(red_state=q, blue_state=expected_state)
            statusmesg += f"\n------   Err({max_errors})"
            if rank == 0:
                logger.info(statusmesg)

            maxerr = np.max(max_errors)
            if maxerr > exittol:
                logger.error("Solution failed to follow expected result.")
                checkpoint_status = 1

    if do_viz:
        dim = discr.dim
        io_fields = make_io_fields(dim, q, dv, eos)
        if have_exact is True:
            io_fields.append(("exact_soln", expected_state))
            result_resid = q - expected_state
            io_fields.append(("residual", result_resid))
        make_output_dump(visualizer, basename=vizname, io_fields=io_fields,
                         comm=comm, step=step, t=t, overwrite=True)

    return checkpoint_status


# Surrogate for the currently non-functioning Uniform class
def set_uniform_solution(t, x_vec, eos=IdealSingleGas()):

    dim = len(x_vec)
    _rho = 1.0
    _p = 1.0
    _velocity = np.zeros(shape=(dim,))
    _gamma = 1.4

    mom0 = _rho * _velocity
    e0 = _p / (_gamma - 1.0)
    ke = 0.5 * np.dot(_velocity, _velocity) / _rho
    x_rel = x_vec[0]
    zeros = 0.0*x_rel
    ones = zeros + 1.0

    mass = zeros + _rho
    mom = make_obj_array([mom0 * ones for i in range(dim)])
    energy = e0 + ke + zeros

    return flat_obj_array(mass, energy, mom)


def import_pseudo_y0_mesh():

    from meshmode.mesh.io import read_gmsh, generate_gmsh, ScriptWithFilesSource
    import os
    if os.path.exists("pseudoY0.msh") is False:
        mesh = generate_gmsh(
            ScriptWithFilesSource("""
            Merge "pseudoY0.brep";
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
        """, ["pseudoY0.brep"]), 3, target_unit='MM')
    else:
        mesh = read_gmsh("pseudoY0.msh")

    return mesh


def main(ctx_factory=cl.create_some_context):

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    logger = logging.getLogger(__name__)

    dim = 3
    order = 1
    exittol = .09
    t_final = 0.01
    current_cfl = 1.0
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[2] = 1.0
    current_dt = 1e-5
    current_t = 0
    eos = IdealSingleGas()
    initializer = Lump(numdim=dim, center=orig, velocity=vel)
    #    initializer = Uniform(numdim=dim)
    casename = 'pseudoY0'
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    constant_cfl = False
    nstatus = 1
    nviz = 1
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

            mesh = import_pseudo_y0_mesh()
            global_nelements = mesh.nelements
            logging.info(f"Total {dim}d elements: {global_nelements}")

            part_per_element = get_partition_by_pymetis(mesh, num_parts)

            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
            del mesh

        else:
            local_mesh = mesh_dist.receive_mesh_part()
    else:
        local_mesh = import_pseudo_y0_mesh()
        global_nelements = local_mesh.nelements

    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    current_state = initializer(0, nodes)
    #    current_state = set_uniform_solution(t=0.0, x_vec=nodes)

    visualizer = make_visualizer(discr, discr.order + 3
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
        return inviscid_operator(discr, q=state, t=t,
                                 boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        return sim_checkpoint(discr, visualizer, eos, logger, exact_soln=initializer,
                              q=state, vizname=casename, step=step, t=t, dt=dt,
                              nstatus=nstatus, nviz=nviz, exittol=exittol,
                              constant_cfl=constant_cfl, comm=comm)

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
