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
from pytools.obj_array import make_obj_array
from functools import partial
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag, DISCR_TAG_QUAD

from mirgecom.discretization import create_discretization_collection
from mirgecom.fluid import make_conserved
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    get_sim_timestep,
    get_box_mesh
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    AdiabaticNoslipWallBoundary,
    IsothermalWallBoundary
)
from mirgecom.transport import SimpleTransport
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state
from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    set_sim_state
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, use_esdg=False, use_overintegration=False,
         use_leap=False, casename=None, rst_filename=None):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    import pytato as pt
    pt.enable_traceback_tag()

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # timestepping control
    timestepper = rk4_step
    t_final = 1e-7
    current_cfl = 0.05
    current_dt = 1e-10
    current_t = 0
    constant_cfl = True
    current_step = 0

    # some i/o frequencies
    nstatus = 1
    nviz = 1
    nrestart = 10
    nhealth = 1

    # some geometry setup
    dim = 2
    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")
    left_boundary_location = 0
    right_boundary_location = 0.1
    ybottom = 0.
    ytop = .02
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
        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        n_refine = 2
        nels_x = 9 * n_refine
        nels_y = 5 * n_refine
        nels_axis = (nels_x, nels_y)
        box_ll = (left_boundary_location, ybottom)
        box_ur = (right_boundary_location, ytop)
        generate_mesh = partial(get_box_mesh, 2, a=box_ll, b=box_ur, n=nels_axis)
        from mirgecom.simutil import generate_and_distribute_mesh
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    # from meshmode.mesh.processing import rotate_mesh_around_axis
    # local_mesh = rotate_mesh_around_axis(local_mesh, theta=-np.pi/4)

    order = 2
    dcoll = create_discretization_collection(actx, local_mesh, order=order,
                                             quadrature_order=order+2)
    nodes = actx.thaw(dcoll.nodes())

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, dcoll, dim,
                             extract_vars_for_logging, units_for_logging)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K) = ({value:1.9e}, "),
            ("max_temperature",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    base_pressure = 100000.0
    pressure_ratio = 1.001
    mu = 1.0

    def poiseuille_2d(x_vec, eos, cv=None, **kwargs):
        y = x_vec[1]
        x = x_vec[0]
        x0 = left_boundary_location
        xmax = right_boundary_location
        xlen = xmax - x0
        p_low = base_pressure
        p_hi = pressure_ratio*base_pressure
        dp = p_hi - p_low
        dpdx = dp/xlen
        h = ytop - ybottom
        u_x = dpdx*y*(h - y)/(2*mu)
        p_x = p_hi - dpdx*x
        rho = 1.0
        mass = 0*x + rho
        u_y = 0*x
        velocity = make_obj_array([u_x, u_y])
        ke = .5*np.dot(velocity, velocity)*mass
        gamma = eos.gamma()
        # if cv is not None:
        #    mass = cv.mass
        #    vel = cv.velocity
        #    ke = .5*np.dot(vel, vel)*mass

        rho_e = p_x/(gamma-1) + ke
        return make_conserved(2, mass=mass, energy=rho_e,
                              momentum=mass*velocity)

    initializer = poiseuille_2d
    gas_model = GasModel(eos=IdealSingleGas(),
                         transport=SimpleTransport(viscosity=mu))

    from mirgecom.simutil import force_evaluation
    exact = force_evaluation(actx, initializer(x_vec=nodes, eos=gas_model.eos))

    def _boundary_solution(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        return make_fluid_state(initializer(x_vec=nodes, eos=gas_model.eos,
                                            cv=state_minus.cv, **kwargs), gas_model)

    use_adiabatic = False
    if use_adiabatic:
        wall_boundary = AdiabaticNoslipWallBoundary()
    else:
        wall_boundary = IsothermalWallBoundary(wall_temperature=300)

    boundaries = {
        BoundaryDomainTag("-1"):
            PrescribedFluidBoundary(
                boundary_state_func=_boundary_solution),
        BoundaryDomainTag("+1"):
            PrescribedFluidBoundary(
                boundary_state_func=_boundary_solution),
        BoundaryDomainTag("-2"): wall_boundary,
        BoundaryDomainTag("+2"): wall_boundary}

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_cv = exact

    def _make_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model)

    make_fluid_state_compiled = actx.compile(_make_fluid_state)

    current_state = make_fluid_state_compiled(current_cv)

    vis_timer = None

    visualizer = make_visualizer(dcoll, order)

    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_write_status(step, t, dt, state, component_errors):
        dv = state.dv
        from grudge.op import nodal_min, nodal_max
        p_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.pressure))
        p_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.pressure))
        t_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.temperature))
        t_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.temperature))
        if constant_cfl:
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            cfl = actx.to_numpy(nodal_max(dcoll, "vol",
                                          get_viscous_cfl(dcoll, dt, state)))
        if rank == 0:
            logger.info(f"Step: {step}, T: {t}, DT: {dt}, CFL: {cfl}\n"
                        f"----- Pressure({p_min}, {p_max})\n"
                        f"----- Temperature({t_min}, {t_max})\n"
                        "----- errors="
                        + ", ".join("%.3g" % en for en in component_errors))

    def my_write_viz(step, t, state, dv):
        resid = state - exact
        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("poiseuille", exact),
                      ("resid", resid)]

        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state, dv, component_errors):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(dcoll, "vol", dv.pressure, 9.999e4,
                                          1.00101e5), op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            p_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.pressure))
            p_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if check_naninf_local(dcoll, "vol", dv.temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/INFs in temperature data.")

        if global_reduce(check_range_local(dcoll, "vol", dv.temperature, 348, 350),
                        op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            t_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.temperature))
            t_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.temperature))
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        exittol = 0.1
        if max(component_errors) > exittol:
            health_error = True
            if rank == 0:
                logger.info("Solution diverged from exact soln.")

        return health_error

    def my_pre_step(step, t, dt, state):

        fluid_state = make_fluid_state_compiled(cv=state)
        dv = fluid_state.dv

        try:
            component_errors = None

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                from mirgecom.simutil import compare_fluid_solutions
                component_errors = compare_fluid_solutions(dcoll, state, exact)
                health_errors = global_reduce(
                    my_health_check(state, dv, component_errors), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state, dv=dv)

            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                  t_final, constant_cfl)

            if do_status:  # needed because logging fails to make output
                if component_errors is None:
                    from mirgecom.simutil import compare_fluid_solutions
                    component_errors = compare_fluid_solutions(dcoll, state, exact)
                my_write_status(step=step, t=t, dt=dt, state=fluid_state,
                                component_errors=component_errors)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state, dv=dv)
            my_write_restart(step=step, t=t, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, gas_model.eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(state, gas_model)
        return ns_operator(dcoll, gas_model=gas_model,
                            boundaries=boundaries,
                            state=fluid_state, time=t, use_esdg=use_esdg,
                            quadrature_tag=quadrature_tag)

    from mirgecom.simutil import force_evaluation
    current_state = force_evaluation(actx, current_state)

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state.cv, t=current_t, t_final=t_final)

    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    final_dv = current_state.dv
    final_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                current_cfl, t_final, constant_cfl)
    from mirgecom.simutil import compare_fluid_solutions
    component_errors = compare_fluid_solutions(dcoll, current_state.cv, exact)

    my_write_viz(step=current_step, t=current_t, state=current_state.cv, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state)
    my_write_status(step=current_step, t=current_t, dt=final_dt,
                    state=current_state, component_errors=component_errors)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "poiseuille"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--esdg", action="store_true",
        help="use flux-differencing/entropy stable DG for inviscid computations.")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
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

    main(actx_class, use_leap=args.leap, use_esdg=args.esdg,
         use_overintegration=args.overintegration or args.esdg,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
