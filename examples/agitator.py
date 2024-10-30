r"""
2D Advection Simulation with Spatially Varying Coefficients

This driver simulates the 2D advection of a Gaussian pulse with
spatially varying velocity coefficients. The advection equation
is solved using a high-order spectral element method optionally
with dealiasing techniques (overintegration). The velocity field
varies as a polynomial in space, which introduces aliasing errors
that are addressed by applying consistent integration as described
in the reference.

The test case follows the setup in Section 4.1 of Mengaldo et al. [1],
and the simulation results are consistent with Figure 7 from the paper.
This case serves as a benchmark for evaluating the effectiveness
of dealiasing strategies in high-order spectral element methods.

The governing equations for the advection problem are:

$$
\partial_t{u} + \partial_x{(a_x u)} + \partial_y{(a_y u)} = 0
$$

where $u$ is the scalar conserved quantity, and $a_x(x, y, t)$ and
$a_y(x, y, t)$ are the spatially varying advection velocities. The
initial conditions include a Gaussian pulse, and the solution evolves
periodically over time so that it recovers the initial state every
*period*.

References:
-----------
[1] G. Mengaldo, D. De Grazia, D. Moxey, P. E. Vincent, and S. J. Sherwin,
    "Dealiasing techniques for high-order spectral element methods on
     regular and irregular grids,"
    Journal of Computational Physics, vol. 299, pp. 56–81, 2015.
    DOI: 10.1016/j.jcp.2015.06.032
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
from functools import partial

from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DD_VOLUME_ALL
from meshmode.mesh import TensorProductElementGroup
from meshmode.mesh.generation import generate_regular_rect_mesh

from logpyle import IntervalTimer, set_dt

from mirgecom.mpi import mpi_entry_point
from mirgecom.discretization import create_discretization_collection
from mirgecom.simutil import (
    distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.operators import div_operator
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
)
from pytools.obj_array import make_obj_array
import grudge.op as op
import grudge.geometry as geo
from grudge.dof_desc import (
    DISCR_TAG_QUAD,
    DISCR_TAG_BASE,
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class _AgitatorCommTag:
    pass


def _advection_flux_interior(dcoll, state_tpair, velocity):
    r"""Compute the numerical flux for divergence of advective flux."""
    actx = state_tpair.int.array_context
    dd = state_tpair.dd
    normal = geo.normal(actx, dcoll, dd)
    v_n = np.dot(velocity, normal)
    # v2 = np.dot(velocity, velocity)
    # vmag = np.sqrt(v2)

    # Lax-Friedrichs type
    # return state_tpair.avg * v_dot_n \
    # + 0.5 * vmag * (state_tpair.int - state_tpair.ext)
    # Simple upwind flux
    # state_upwind = actx.np.where(v_n > 0, state_tpair.int, state_tpair.ext)
    # return state_upwind * v_n
    # Central flux
    return state_tpair.avg * v_n


@mpi_entry_point
def main(actx_class,
         use_overintegration=False, use_leap=False,
         casename=None, rst_filename=None,
         init_type=None, order=None, quad_order=None,
         tpe=None, p_adv=1.0):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"
    if init_type is None:
        init_type = "mengaldo"
    if order is None:
        order = 1
    if quad_order is None:
        quad_order = order if tpe else order + 3
    if tpe is None:
        tpe = False
    if use_overintegration is None:
        use_overintegration = False

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    # nsteps = 5000
    current_cfl = 1.0
    current_dt = 5e-3
    # t_final = nsteps * current_dt
    current_t = 0
    constant_cfl = False
    nsteps_period = 2000
    period = nsteps_period*current_dt
    t_final = 4*period

    # some i/o frequencies
    nstatus = 1
    nrestart = 10000
    nviz = 100
    nhealth = 1

    # Mengaldo test case setup stuff
    alpha = 41.
    xc = -.3
    yc = 0

    dim = 2
    nel_1d = 8

    def g(t, actx=None):
        if actx is None:
            return np.cos(np.pi*t/period)
        return actx.np.cos(np.pi*t/period)

    # Note that *r* here is used only to get the array_context
    # The actual velocity is returned at points on the discretization
    # that comes from the DD.
    def get_velocity(r, t=0, dd=None):
        if dd is None:
            dd = DD_VOLUME_ALL
        actx = r[0].array_context
        discr_local = dcoll.discr_from_dd(dd)
        r_local = actx.thaw(discr_local.nodes())
        x = r_local[0]
        y = r_local[1]
        vx = y**p_adv
        vy = -1*x**p_adv
        return np.pi * g(t, actx) * make_obj_array([vx, vy])

    def poly_vel_initializer(xyz_vec, t=0):
        x = xyz_vec[0]
        y = xyz_vec[1]
        actx = x.array_context
        return actx.np.exp(-alpha*((x-xc)**2 + (y-yc)**2))

    nel_axes = (nel_1d,)*dim
    box_ll = (-1,)*dim
    box_ur = (1,)*dim
    print(f"{box_ll=}, {box_ur=}, {nel_axes=}")

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == num_parts
    else:  # generate the grid from scratch
        grp_cls = TensorProductElementGroup if tpe else None
        mesh_type = None if tpe else "X"
        generate_mesh = partial(
            generate_regular_rect_mesh,
            a=box_ll, b=box_ur, nelements_per_axis=nel_axes,
            periodic=(True,)*dim, mesh_type=mesh_type,
            group_cls=grp_cls
        )

        local_mesh, global_nelements = distribute_mesh(comm, generate_mesh)
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order,
                                             quadrature_order=quad_order)

    nodes_base = actx.thaw(dcoll.nodes())
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else DISCR_TAG_BASE
    if use_overintegration:
        print(f"Using Overintegration: {quadrature_tag=}, {order=}, {quad_order=}")

    # transfer trace pairs to quad grid, update pair dd
    interp_to_surf_quad = partial(op.tracepair_with_discr_tag,
                                  dcoll, quadrature_tag)

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    exact_state_func = poly_vel_initializer
    velocity_func = get_velocity
    init_state_base = exact_state_func(nodes_base)

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_state = restart_data["state"]
    else:
        # Set the current state from time 0
        current_state = init_state_base

    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, current_step, current_t)

    visualizer = make_visualizer(dcoll)

    initname = init_type
    eosname = "none"
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    def my_write_viz(step, t, state):
        exact_state = exact_state_func(nodes_base, t)
        state_resid = state - exact_state
        viz_fields = [("state", state),
                      ("exact_state", exact_state),
                      ("state_resid", state_resid)]
        if velocity_func is not None:
            vel = velocity_func(nodes_base, t=t)
            viz_fields.append(("velocity", vel))

        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)
        norm_err = actx.to_numpy(op.norm(dcoll, state_resid, 2))
        print(f"L2 Error: {norm_err:.16g}")

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state):
        health_error = False
        from mirgecom.simutil import check_naninf_local
        if check_naninf_local(dcoll, "vol", state):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")
        return health_error

    def my_pre_step(step, t, dt, state):

        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(
                    my_health_check(state), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Scalar field failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        # Need a timestep calculator for scalar fields
        # dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl, t_final,
        #                      constant_cfl)
        time_left = max(0, t_final - t)
        dt = min(time_left, dt)
        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_advection_rhs(t, state, velocity,
                         state_interior_trace_pairs=None,
                         flux_only=False):

        if state_interior_trace_pairs is None:
            state_interior_trace_pairs = \
                op.interior_trace_pairs(dcoll, state, comm_tag=_AgitatorCommTag)
        v_quad = op.project(dcoll, dd_vol, dd_vol_quad, velocity)

        # This "flux" function returns the *numerical flux* that will
        # be used in the divergence operation given a trace pair,
        # i.e. the soln on the -/+ sides of the face.
        def flux(state_tpair):
            # why project to all_faces? to "size" the array correctly
            # for all faces rather than just the selected "tpair.dd"
            v_dd = op.project(dcoll, dd_vol, state_tpair.dd, velocity)
            return op.project(
                dcoll, state_tpair.dd, dd_allfaces_quad,
                _advection_flux_interior(
                    dcoll, state_tpair, v_dd))

        vol_flux = state * v_quad

        # sums up the fluxes for each element boundary
        surf_flux = sum(flux(tpair)
                        for tpair in state_interior_trace_pairs)

        if flux_only:
            return vol_flux, surf_flux

        return -div_operator(dcoll, dd_vol, dd_allfaces, vol_flux, surf_flux)

    def my_rhs(t, state):
        state_quad = op.project(dcoll, dd_vol, dd_vol_quad, state)
        velocity = get_velocity(nodes_base, t, dd_vol)

        state_interior_trace_pairs = [
            interp_to_surf_quad(tpair=tpair)
            for tpair in op.interior_trace_pairs(dcoll, state,
                                                 comm_tag=_AgitatorCommTag)
        ]
        vol_fluxes, surf_fluxes = \
            my_advection_rhs(t, state_quad, velocity,
                             state_interior_trace_pairs, flux_only=True)

        return -1.0*div_operator(dcoll, dd_vol_quad, dd_allfaces_quad,
                                 vol_fluxes, surf_fluxes)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_state = current_state

    my_write_viz(step=current_step, t=current_t, state=final_state)
    my_write_restart(step=current_step, t=current_t, state=final_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-12
    print(f"{current_t=}, {t_final=}, {(current_t - t_final)=}")
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "mengaldo"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("-o", "--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("-l", "--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("-t", "--tpe", action="store_true",
        help="use tensor product elements.")
    parser.add_argument("-y", "--order", type=int, default=1,
        help="use polynomials of degree=order for element basis")
    parser.add_argument("-q", "--quad-order", type=int,
        help="use quadrature exact to *quad-order*.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("-i", "--init", type=str, help="name of the init type",
                        default="mengaldo")
    parser.add_argument("-w", "--warp", type=float,
                        help="warp power for velocity field", default=1.0)
    parser.add_argument("-c", "--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class,
         use_overintegration=args.overintegration,
         use_leap=args.leap, init_type=args.init, p_adv=args.warp,
         order=args.order, quad_order=args.quad_order, tpe=args.tpe,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
