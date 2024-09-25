"""Demonstrate simple scalar advection-diffusion."""

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
from pytools.obj_array import make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.discretization import create_discretization_collection
from grudge.shortcuts import make_visualizer


from mirgecom.transport import SimpleTransport
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    # get_sim_timestep,
    generate_and_distribute_mesh,
    compare_fluid_solutions
)
from mirgecom.limiter import bound_preserving_limiter
from mirgecom.fluid import make_conserved
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
# from mirgecom.boundary import PrescribedFluidBoundary
# from mirgecom.initializers import MulticomponentLump
from mirgecom.eos import IdealSingleGas

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, use_overintegration=False, use_esdg=False,
         use_leap=False, casename=None, rst_filename=None, use_tpe=True):
    """Drive example."""
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

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=True,
                           use_einsum_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    t_final = 2e-4
    current_cfl = 0.1
    current_dt = 1e-5
    current_t = 0
    constant_cfl = False

    # some i/o frequencies
    nstatus = 100
    nrestart = 100
    nviz = 10
    nhealth = 100

    dim = 2
    nel_1d = 8
    order = 3

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
        assert restart_data["num_parts"] == nparts
    else:  # generate the grid from scratch
        box_ll = -.5
        box_ur = .5
        periodic = (True, )*dim
        from meshmode.mesh.generation import generate_regular_rect_mesh
        from meshmode.mesh import TensorProductElementGroup
        grp_cls = TensorProductElementGroup if use_tpe else None
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim,
                                periodic=periodic, group_cls=grp_cls)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order,
                                             tensor_product_elements=use_tpe)
    nodes = actx.thaw(dcoll.nodes())

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    def _limit_fluid_cv(cv, temperature_seed=None, gas_model=None, dd=None):
        actx = cv.array_context

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], mmin=0.0,
                                     dd=dd)
            for i in range(nspecies)
        ])
        spec_lim = actx.np.where(actx.np.greater(spec_lim, 0.0), spec_lim, 0.0)

        # normalize to ensure sum_Yi = 1.0
        # aux = cv.mass*0.0
        # for i in range(0, nspecies):
        #     aux = aux + spec_lim[i]
        # spec_lim = spec_lim/aux

        # recompute density
        # mass_lim = eos.get_density(pressure=pressure,
        #    temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        # energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
        #    temperature, species_mass_fractions=spec_lim)
        #    + 0.5*np.dot(cv.velocity, cv.velocity)
        # )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=cv.mass, energy=cv.energy,
                              momentum=cv.momentum,
                              species_mass=cv.mass*spec_lim)
    use_limiter = False
    limiter_function = _limit_fluid_cv if use_limiter else None

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]

    from grudge.dt_utils import characteristic_lengthscales

    try:
        dx = characteristic_lengthscales(actx, dcoll)
    except NotImplementedError:
        from warnings import warn
        warn("This example requires https://github.com/inducer/grudge/pull/338 . "
             "Exiting.")
        return

    dx_min, dx_max = vol_min(dx), vol_max(dx)

    print(f"DX: ({dx_min}, {dx_max})")
    vis_timer = None

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, dcoll, dim,
                             extract_vars_for_logging, units_for_logging)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    # soln setup and init
    nspecies = 4
    centers = make_obj_array([np.zeros(shape=(dim,)) for i in range(nspecies)])
    velocity = np.zeros(shape=(dim,))
    velocity[0] = 300.
    wave_vector = np.zeros(shape=(dim,))
    wave_vector[0] = 1.
    wave_vector = wave_vector / np.sqrt(np.dot(wave_vector, wave_vector))

    spec_y0s = np.zeros(shape=(nspecies,))
    spec_amplitudes = np.ones(shape=(nspecies,))
    spec_omegas = 2. * np.pi * np.ones(shape=(nspecies,))

    kappa = 0.0
    mu = 1e-5
    spec_diff = 1e-1
    spec_diffusivities = np.array([spec_diff * 1./float(j+1)
                                   for j in range(nspecies)])
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivities)

    eos = IdealSingleGas()
    from mirgecom.gas_model import GasModel, make_fluid_state
    gas_model = GasModel(eos=eos, transport=transport_model)

    from mirgecom.initializers import MulticomponentTrig
    initializer = MulticomponentTrig(dim=dim, nspecies=nspecies,
                                     p0=101325, rho0=1.3,
                                     spec_centers=centers, velocity=velocity,
                                     spec_y0s=spec_y0s,
                                     spec_amplitudes=spec_amplitudes,
                                     spec_omegas=spec_omegas,
                                     spec_diffusivities=spec_diffusivities,
                                     wave_vector=wave_vector,
                                     trig_function=actx.np.sin)

    def boundary_solution(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        return make_fluid_state(initializer(x_vec=nodes, eos=gas_model.eos,
                                            **kwargs), gas_model,
                                limiter_func=limiter_function,
                                limiter_dd=dd_bdry)

    boundaries = {}

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_cv = initializer(nodes)

    current_state = make_fluid_state(current_cv, gas_model,
                                     limiter_func=limiter_function)
    convective_speed = np.sqrt(np.dot(velocity, velocity))
    c = current_state.speed_of_sound
    mach = vol_max(convective_speed / c)
    cell_peclet = c * dx / (2 * spec_diff)
    pe_min, pe_max = vol_min(cell_peclet), vol_max(cell_peclet)

    print(f"Mach: {mach}")
    print(f"Cell Peclet: ({pe_min, pe_max})")

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

    def my_write_status(component_errors):
        if rank == 0:
            logger.info(
                "------- errors="
                + ", ".join("%.3g" % en for en in component_errors))

    def my_write_viz(step, t, cv, dv, exact):
        resid = cv - exact
        viz_fields = [("cv", cv),
                       ("dv", dv),
                       ("exact", exact),
                       ("resid", resid)]
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def my_write_restart(step, t, cv):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure, component_errors):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        if check_range_local(dcoll, "vol", pressure, 101324.99, 101325.01):
            health_error = True
            logger.info(f"{rank=}: Pressure out of expected range.")

        exittol = .09
        if max(component_errors) > exittol:
            health_error = False
            if rank == 0:
                logger.info("Solution diverged from exact soln.")

        return health_error

    def my_pre_step(step, t, dt, state):
        cv = state

        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_viz or do_health or do_status:
                fluid_state = make_fluid_state(state, gas_model)
                dv = fluid_state.dv
                exact = initializer(x_vec=nodes, eos=eos, time=t)

                if do_health or do_status:
                    component_errors = \
                        compare_fluid_solutions(dcoll, cv, exact)

                    if do_health:
                        health_errors = global_reduce(
                            my_health_check(dv.pressure, component_errors), op="lor")
                        if health_errors:
                            if rank == 0:
                                logger.info("Fluid solution failed health check.")
                                raise MyRuntimeError(
                                    "Failed simulation health check.")

                    if do_status:
                        my_write_status(component_errors=component_errors)

                if do_viz:
                    my_write_viz(step=step, t=t, cv=cv, dv=dv, exact=exact)

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            raise

        # dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl, t_final,
        #                      constant_cfl)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(state, gas_model,
                                       limiter_func=limiter_function)
        return ns_operator(dcoll, state=fluid_state, time=t,
                           boundaries=boundaries, gas_model=gas_model,
                           quadrature_tag=quadrature_tag, use_esdg=use_esdg,
                           limiter_func=limiter_function)

    # current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
    #                              current_cfl, t_final, constant_cfl)

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step, dt=current_dt,
                      post_step_callback=my_post_step,
                      state=current_state.cv, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_state = make_fluid_state(current_cv, gas_model)
    final_dv = current_state.dv
    final_exact = initializer(x_vec=nodes, eos=eos, time=current_t)
    my_write_viz(step=current_step, t=current_t, cv=current_state.cv, dv=final_dv,
                 exact=final_exact)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    time_err = current_t - t_final
    if np.abs(time_err) > finish_tol:
        raise ValueError(f"Simulation did not finish at expected time {time_err=}.")


if __name__ == "__main__":
    import argparse
    casename = "scalar-advdiff"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--esdg", action="store_true",
        help="use entropy-stable rhs operator")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration")
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
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True,
                                                    profiling=args.profiling,
                                                    numpy=args.numpy)

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
