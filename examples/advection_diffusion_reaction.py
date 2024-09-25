"""Demonstrate scalar advection-diffusion-reaction equation example."""

__copyright__ = "Copyright (C) 2024 University of Illinois Board of Trustees"

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

from meshmode.mesh import BTAG_ALL

from grudge.geometry import normal as normal_vector
from grudge.trace_pair import TracePair, interior_trace_pairs
from grudge.dof_desc import BoundaryDomainTag, DD_VOLUME_ALL
from grudge.shortcuts import make_visualizer
from grudge import op

from logpyle import set_dt

from mirgecom.io import make_init_message
from mirgecom.discretization import create_discretization_collection
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.simutil import (
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_step
)
from mirgecom.utils import force_evaluation
from mirgecom.diffusion import (
    diffusion_operator,
    NeumannDiffusionBoundary,
    RobinDiffusionBoundary)
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_set_time, logmgr_add_cl_device_info
)


logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def init_u_field(actx, nodes, t=0):
    """Initialize the field."""
    return actx.np.exp(-(nodes[0]+1.0)**2/0.1**2)*0.0


def _facial_flux(dcoll, u_tpair):

    actx = u_tpair.int.array_context
    normal = normal_vector(actx, dcoll, u_tpair.dd)
    flux_weak = u_tpair.avg*normal[0]

    return op.project(dcoll, u_tpair.dd, "all_faces", flux_weak)


class _OperatorTag:
    pass


def linear_advection_operator(dcoll, u, u_bc, *, comm_tag=None):
    """Compute the advection part of the transport.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    u: DOF array representing the independent variable
    u_bc: DOF array representing the Dirichlet boundary value
    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    numpy.ndarray
        an object array of DOF arrays, representing the ODE RHS
    """
    u_bnd = op.project(dcoll, "vol", BTAG_ALL, u)
    itp = interior_trace_pairs(dcoll, u, comm_tag=(_OperatorTag, comm_tag))

    dd = DD_VOLUME_ALL

    # boundary flux
    el_bnd_flux = (
        _facial_flux(dcoll, u_tpair=TracePair(dd=dd.with_domain_tag(BTAG_ALL),
                                              interior=u_bnd, exterior=u_bc))
        + sum([_facial_flux(dcoll, u_tpair=tpair) for tpair in itp]))

    # volume term
    vol_flux = op.weak_local_d_dx(dcoll, 0, u)
    return op.inverse_mass(dcoll, vol_flux - op.face_mass(dcoll, el_bnd_flux))


def get_source(actx, u, x):
    return -10.0*u


@mpi_entry_point
def main(actx_class, use_overintegration=False, casename=None, rst_filename=None):
    """Drive the Advection-Diffusion-Reaction Equation example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from functools import partial
    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True, filename=f"{casename}.sqlite",
                               mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)

    t_final = 0.1
    nviz = 100
    nstatus = 100

    dim = 1
    order = 2
    nel_1d = 100

    if dim != 1:
        raise NotImplementedError("Only works in 1D.")

    # ~~~
    if rst_filename is None:

        current_t = 0
        current_step = 0

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(
            generate_regular_rect_mesh,
            a=(-1.0,)*dim, b=(1.0,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={"inlet": ["-x"], "outlet": ["+x"]})
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
    else:
        from mirgecom.restart import read_restart_data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        assert comm.Get_size() == restart_data["nparts"]

        current_t = restart_data["t"]
        current_step = restart_data["step"]
        order = restart_data["order"]

    local_nelements = local_mesh.nelements
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    if rst_filename is None:
        u = init_u_field(actx, nodes, t=0)
    else:
        u = restart_data["u"]

    u = force_evaluation(actx, u)

    # ~~~
    constant_cfl = False
    inviscid_cfl = 0.485
    viscous_cfl = 0.15
    wave_speed = 1.0
    kappa = 0.01

    from grudge.dt_utils import characteristic_lengthscales
    dx = actx.to_numpy(actx.np.min(characteristic_lengthscales(actx, dcoll)))
    inv_dt = dx / wave_speed
    visc_dt = dx**2 / kappa

    dt_i = inviscid_cfl * inv_dt
    dt_v = viscous_cfl * visc_dt
    current_dt = np.minimum(dt_i, dt_v)
    current_cfl = inviscid_cfl if dt_i < dt_v else viscous_cfl

    # ~~~
    presc_value = 1.0
    boundaries = {
        BoundaryDomainTag("inlet"): RobinDiffusionBoundary(presc_value, 1.0),
        BoundaryDomainTag("outlet"): NeumannDiffusionBoundary(0.),
    }

    # ~~~
    visualizer = make_visualizer(dcoll)

    eosname = None
    initname = "advection-diffusion-reaction"
    init_message = make_init_message(
        dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=current_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=current_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.5e} s, "),
            ("t_sim.max", "sim time: {value:1.5e} s, "),
            ("t_step.max", "--- step walltime: {value:5g} s\n")
            ])

    def my_write_viz(step, t, u):
        viz_fields = [("u", u), ("x", nodes[0])]
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_health_check(u):
        health_error = False
        if check_naninf_local(dcoll, "vol", u):
            health_error = True
            logger.info(f"{rank=}: Invalid field data found.")

        return health_error

    def my_pre_step(step, t, dt, state):
        if logmgr:
            logmgr.tick_before()

        u = state

        try:
            health_errors = global_reduce(my_health_check(u), op="lor")
            if health_errors:
                if rank == 0:
                    logger.info("Fluid solution failed health check.")
                raise MyRuntimeError("Failed simulation health check.")

            if check_step(step=step, interval=nviz):
                my_write_viz(step=step, t=t, u=u)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, u):

        # inviscid flux
        u_bc = op.project(dcoll, "vol", BTAG_ALL, u)
        inv = linear_advection_operator(dcoll, u=u, u_bc=u_bc)

        # viscous flux
        visc = diffusion_operator(dcoll, kappa=kappa, boundaries=boundaries,
                                  u=u, penalty_amount=0.0)

        source = get_source(actx, u, nodes[0])

        return inv + visc + source

    if rank == 0:
        logging.info("Stepping.")

    from mirgecom.steppers import advance_state
    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=rk4_step, state=u,
                      pre_step_callback=my_pre_step, dt=current_dt,
                      post_step_callback=my_post_step,
                      istep=current_step, t=current_t, t_final=t_final)

    # ~~~ compute BC error
    dd = DD_VOLUME_ALL
    visc, grad = diffusion_operator(dcoll, kappa=kappa, boundaries=boundaries,
                                    u=current_cv, penalty_amount=0.0,
                                    return_grad_u=True)

    u_inlet = op.project(dcoll, "vol", dd.trace("inlet").domain_tag, current_cv)
    grad_u_inlet = op.project(dcoll, "vol", dd.trace("inlet").domain_tag, grad)
    kappa_inlet = op.project(dcoll, "vol", dd.trace("inlet").domain_tag, kappa)
    print(grad_u_inlet*kappa_inlet - (u_inlet - presc_value))

    if logmgr:
        logmgr.close()


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    casename = "advdiff"
    parser = argparse.ArgumentParser(description="Advection-Diffusion-Reaction Eq.")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    lazy = args.lazy

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True,
                                                    profiling=args.profiling,
                                                    numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_overintegration=args.overintegration,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
