"""Demonstrate overintegration testing from Mengaldo paper."""

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

# from meshmode.mesh import BTAG_ALL
# from grudge.dof_desc import DISCR_TAG_QUAD
from arraycontext import get_container_context_recursively
from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DD_VOLUME_ALL

from logpyle import IntervalTimer, set_dt

from mirgecom.mpi import mpi_entry_point
from mirgecom.discretization import create_discretization_collection
from mirgecom.simutil import (
    distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.operators import (
    grad_operator,
    div_operator
)
from mirgecom.utils import force_evaluation
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


def _diffusion_flux_interior(dcoll, grad_tpair, state_tpair,
                             diffusivity, alpha=0):
    actx = state_tpair.int.array_context
    normal = actx.thaw(dcoll.normal(state_tpair.dd))
    # average of -J
    flux_avg = diffusivity * grad_tpair.avg
    # This is -J .dot. nhat
    flux_n = np.dot(flux_avg, normal)
    state_jump = state_tpair.ext - state_tpair.int
    dissipation = alpha * state_jump
    return flux_n - dissipation


def _gradient_flux_interior(dcoll, state_tpair):
    """Compute interior face flux for gradient operator."""
    actx = state_tpair.int.array_context
    dd_trace = state_tpair.dd
    dd_allfaces = dd_trace.with_boundary_tag(FACE_RESTR_ALL)
    normal = actx.thaw(dcoll.normal(state_tpair.dd))
    flux = state_tpair.avg * normal
    return op.project(dcoll, dd_trace, dd_allfaces, flux)


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
def main(actx_class, use_esdg=False,
         use_overintegration=False, use_leap=False,
         casename=None, rst_filename=None,
         init_type=None):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"
    if init_type is None:
        init_type = "mengaldo"

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

    nsteps = 2000
    current_cfl = 1.0
    current_dt = 1e-2
    t_final = nsteps * current_dt
    current_t = 0
    constant_cfl = False

    # some i/o frequencies
    nstatus = 1
    nrestart = 5
    nviz = 1
    nhealth = 1

    # Bunch of problem setup stuff
    dim = 3
    if init_type == "mengaldo":
        dim = 2

    nel_1d = 8
    order = 1

    advect = True
    diffuse = False
    geom_scale = 1e-3
    # wavelength = 2 * np.pi * geom_scale
    wavelength = geom_scale
    wavenumber = 2 * np.pi / wavelength
    wavenumber2 = wavenumber * wavenumber
    if init_type != "mengaldo":
        current_dt = wavelength / 5
    wave_hat = np.ones(shape=(dim,))
    # wave_hat[0] = 0.3
    k2 = np.dot(wave_hat, wave_hat)
    wave_hat = wave_hat / np.sqrt(k2)
    amplitude = wavelength/4
    alpha = 1e-1 / (wavelength * wavelength)
    diffusivity = 1e-3 / alpha
    print(f"{diffusivity=}")
    diffusion_flux_penalty = 0.
    velocity = np.ones(shape=(dim,))
    # velocity[0] = .1
    # velocity[1] = .1
    velocity = velocity * geom_scale
    p_adv = 1.
    period = 100*current_dt

    def g(t):
        return np.cos(np.pi*t/period)

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
        return np.pi * g(t) * make_obj_array([vx, vy])

    cos_axis = np.zeros(shape=(dim,))
    scal_fac = np.ones(shape=(dim,))
    box_ll = (-wavelength,)*dim
    box_ur = (wavelength,)*dim
    nel_axes = (nel_1d,)*dim
    boxl = list(box_ll)
    boxr = list(box_ur)
    nelax = list(nel_axes)
    for d in range(dim):
        axis = np.zeros(shape=(dim,))
        axis[d] = 1
        cos_axis[d] = np.abs(np.dot(wave_hat, axis))
        if cos_axis[d] < 1e-12:
            wave_hat[d] = 0
            # Reduce the domain size for 3D to make it cheaper
            if dim == 3:
                scal_fac[d] = 0.25
        else:
            scal_fac[d] = 1./cos_axis[d]
        boxl[d] = scal_fac[d]*boxl[d]
        boxr[d] = scal_fac[d]*boxr[d]
        nelax[d] = int(scal_fac[d]*nelax[d])

    box_ll = tuple(boxl)
    box_ur = tuple(boxr)
    nel_axes = tuple(nelax)
    print(f"{box_ll=}, {box_ur=}, {nel_axes=}")
    if init_type == "mengaldo":
        box_ll = (-1,)*dim
        box_ur = (1,)*dim

    # renormalize wave_vector after it potentially changed
    k2 = np.dot(wave_hat, wave_hat)
    wave_hat = wave_hat / np.sqrt(k2)

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
        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
            a=box_ll, b=box_ur, nelements_per_axis=nel_axes,
            periodic=(True,)*dim
        )

        local_mesh, global_nelements = distribute_mesh(comm, generate_mesh)
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else DISCR_TAG_BASE

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

    def my_scalar_gradient(state, state_interior_trace_pairs=None,
                           dd_vol=None, dd_allfaces=None):
        if dd_vol is None:
            dd_vol = DD_VOLUME_ALL
        if dd_allfaces is None:
            dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)
        if state_interior_trace_pairs is None:
            state_interior_trace_pairs = op.interior_trace_pairs(dcoll, state)

        get_interior_flux = partial(_gradient_flux_interior, dcoll)

        state_flux_bnd = sum(get_interior_flux(state_tpair) for state_tpair in
                             state_interior_trace_pairs)

        return grad_operator(
            dcoll, dd_vol, dd_allfaces, state, state_flux_bnd)

    # Mengaldo test case
    alpha = 41.
    xc = 0
    yc = 0

    def poly_vel_initializer(xyz_vec, t=0):
        x = xyz_vec[0]
        y = xyz_vec[1]
        actx = x.array_context
        return actx.np.exp(-alpha*((x+xc)**2 + (y+yc)**2))

    # These solution/gradient computers assume unit wave_hat
    def wave_initializer(xyz_vec, t=0):
        actx = get_container_context_recursively(xyz_vec)
        expterm = 1.0
        r_vec = 1.0*xyz_vec
        if advect:
            r_vec = xyz_vec - t*velocity
        if diffuse:
            expterm = np.exp(-diffusivity*wavenumber2*t)
        wave_x = wavenumber*np.dot(wave_hat, r_vec)
        trigterm = amplitude*actx.np.cos(wave_x)
        return expterm*trigterm

    def wave_gradient(xyz_vec, t=0):
        actx = get_container_context_recursively(xyz_vec)
        r_vec = 1.0*xyz_vec
        expterm = 1.0
        if advect:
            r_vec = xyz_vec - t*velocity
        if diffuse:
            expterm = np.exp(-diffusivity*wavenumber2*t)
        wave_x = wavenumber*np.dot(wave_hat, r_vec)
        trigterm = -1.0*amplitude*wavenumber*actx.np.sin(wave_x)
        return expterm*trigterm*wave_hat

    def gaussian_initializer(xyz_vec, t=0):
        actx = get_container_context_recursively(xyz_vec)
        expterm = 1.0
        r_vec = 1.0*xyz_vec
        if advect:
            r_vec = xyz_vec - t*velocity
        if diffuse:
            expterm = amplitude*np.exp(-diffusivity*alpha*alpha*t)
        r2 = np.dot(r_vec, r_vec)
        return expterm*actx.np.exp(-alpha*r2)

    def gaussian_gradient(xyz_vec, t=0):
        r_vec = 1.0*xyz_vec
        if advect:
            r_vec = xyz_vec - t*velocity
        dr2 = 2*r_vec
        return -alpha*dr2*gaussian_initializer(xyz_vec, t)

    if init_type == "gaussian":
        exact_state_func = gaussian_initializer
        exact_gradient_func = gaussian_gradient
    elif init_type == "wave":
        exact_state_func = wave_initializer
        exact_gradient_func = wave_gradient
    elif init_type == "mengaldo":
        exact_state_func = poly_vel_initializer
        exact_gradient_func = None
        velocity_func = get_velocity
    else:
        raise ValueError(f"Unexpected {init_type=}")

    init_state = exact_state_func(nodes)

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_state = restart_data["state"]
    else:
        # Set the current state from time 0
        current_state = init_state

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

    def my_write_viz(step, t, state, grad_state=None):
        if grad_state is None:
            grad_state = force_evaluation(actx, my_scalar_gradient(state))
        exact_state = exact_state_func(nodes, t)
        state_resid = state - exact_state
        expterm = np.exp(-diffusivity*wavenumber2*t)
        exact_amplitude = amplitude*expterm
        state_err = state_resid / exact_amplitude
        viz_fields = [("state", state),
                      ("exact_state", exact_state),
                      ("state_resid", state_resid),
                      ("state_relerr", state_err),
                      ("dstate", grad_state)]
        if exact_gradient_func is not None:
            exact_grad = exact_gradient_func(nodes, t)
            grad_resid = grad_state - exact_grad
            exact_grad_amplitude = wavenumber*exact_amplitude
            grad_err = grad_resid / exact_grad_amplitude
            grad_viz_fields = [("exact_dstate", exact_grad),
                               ("grad_resid", grad_resid),
                               ("grad_relerr", grad_err)]
            viz_fields.extend(grad_viz_fields)
        if velocity_func is not None:
            vel = velocity_func(nodes, t=t)
            viz_fields.append(("velocity", vel))

        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)

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

    def my_health_check(state, grad_state=None):
        if grad_state is None:
            grad_state = force_evaluation(actx, my_scalar_gradient(state))
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

            if do_viz or do_health:
                grad_state = force_evaluation(actx, my_scalar_gradient(state))

            if do_health:
                health_errors = global_reduce(my_health_check(state, grad_state),
                                              op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Scalar field failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state, grad_state=grad_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        # Need a timestep calculator for scalar fields
        # dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl, t_final,
        #                      constant_cfl)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_diffusion_rhs(t, state, dstate=None, state_interior_trace_pairs=None,
                         grad_state_interior_trace_pairs=None, flux_only=False):

        if state_interior_trace_pairs is None:
            state_interior_trace_pairs = op.interior_trace_pairs(dcoll, state)
        if dstate is None:
            dstate = my_scalar_gradient(state, state_interior_trace_pairs)
        if grad_state_interior_trace_pairs is None:
            grad_state_interior_trace_pairs = op.interior_trace_pairs(dcoll, dstate)

        all_trace_pairs = zip(grad_state_interior_trace_pairs,
                              state_interior_trace_pairs)

        vol_flux = diffusivity * dstate
        get_interior_flux = partial(
            _diffusion_flux_interior, dcoll, diffusivity=diffusivity,
            alpha=diffusion_flux_penalty)

        def flux(grad_tpair, state_tpair):
            return op.project(
                dcoll, grad_tpair.dd, "all_faces", get_interior_flux(grad_tpair,
                                                                     state_tpair))

        surf_flux = sum(flux(grad_tpair, state_tpair)
                        for grad_tpair, state_tpair in all_trace_pairs)

        if flux_only:
            return vol_flux, surf_flux

        return div_operator(dcoll, dd_vol, dd_allfaces, vol_flux, surf_flux)

    def my_advection_rhs(t, state, state_interior_trace_pairs=None,
                         flux_only=False):

        if state_interior_trace_pairs is None:
            state_interior_trace_pairs = op.interior_trace_pairs(dcoll, state)

        # This "flux" function returns the *numerical flux* that will
        # be used in the divergence operation given a trace pair,
        # i.e. the soln on the -/+ sides of the face.
        def flux(tpair):
            # why project to all_faces? to "size" the array correctly
            # for all faces rather than just the selected "tpair.dd"
            return op.project(
                dcoll, tpair.dd, "all_faces",
                _advection_flux_interior(dcoll, tpair,
                                         get_velocity(r=nodes, t=t, dd=tpair.dd)))

        vol_flux = state * get_velocity(r=nodes, t=t)

        # sums up the fluxes for each element boundary
        surf_flux = sum(flux(tpair) for tpair in state_interior_trace_pairs)

        if flux_only:
            return vol_flux, surf_flux

        return -div_operator(dcoll, dd_vol, dd_allfaces, vol_flux, surf_flux)

    def my_rhs(t, state):
        state_interior_trace_pairs = None
        vol_fluxes = 0
        surf_fluxes = 0

        if advect:
            if state_interior_trace_pairs is None:
                state_interior_trace_pairs = op.interior_trace_pairs(dcoll, state)
            vol_fluxes, surf_fluxes = my_advection_rhs(
                t, state, state_interior_trace_pairs, flux_only=True)
            vol_fluxes = -1.0*vol_fluxes
            surf_fluxes = -1.0*surf_fluxes

        if diffuse:
            if state_interior_trace_pairs is None:
                state_interior_trace_pairs = op.interior_trace_pairs(dcoll, state)
            grad_state = my_scalar_gradient(state, state_interior_trace_pairs)
            grad_state_interior_trace_pairs = \
                op.interior_trace_pairs(dcoll, grad_state)
            diff_vol_fluxes, diff_surf_fluxes = my_diffusion_rhs(
                t, state, dstate=grad_state,
                state_interior_trace_pairs=state_interior_trace_pairs,
                grad_state_interior_trace_pairs=grad_state_interior_trace_pairs,
                flux_only=True)
            vol_fluxes = diff_vol_fluxes + vol_fluxes
            surf_fluxes = diff_surf_fluxes + surf_fluxes

        if diffuse or advect:
            return div_operator(dcoll, dd_vol, dd_allfaces, vol_fluxes, surf_fluxes)

        return 0*state

    # Need to create a dt calculator for scalar field
    # current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
    #                              current_cfl, t_final, constant_cfl)

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
    casename = "scalar-transport"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--esdg", action="store_true",
        help="use entropy-stable dg for inviscid terms.")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--init", type=str, help="name of the init type",
                        default="mengaldo")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from warnings import warn
    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_esdg=args.esdg,
         use_overintegration=args.overintegration or args.esdg,
         use_leap=args.leap, init_type=args.init,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
