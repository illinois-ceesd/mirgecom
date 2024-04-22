"""Demonstrate combustive mixture with Pyrometheus."""

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
import sys
import logging
import numpy as np
from functools import partial

from pytools.obj_array import make_obj_array
from grudge.shortcuts import make_visualizer
import grudge.op as op
from logpyle import IntervalTimer, set_dt
from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    ApplicationOptionsError,
    check_step, check_naninf_local, check_range_local
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.initializers import Uniform
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.utils import force_evaluation
from mirgecom.limiter import bound_preserving_limiter
from mirgecom.fluid import make_conserved
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
)

import cantera


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception for fatal driver errors."""

    pass


@mpi_entry_point
def main(actx_class, use_leap=False, use_overintegration=False,
         casename=None, rst_filename=None, log_dependent=True,
         viscous_terms_on=False, use_esdg=False,
         use_tensor_product_els=False):
    """Drive example."""
    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    # if use_tensor_product_els:
    #    # For lazy:
    #    actx = initialize_actx(TensorProductMPIFusionContractorArrayContext,
    #                           comm)
    #    # For eager:
    #    # actx = initialize_actx(TensorProductMPIPyOpenCLArrayContext, comm)
    #    use_profiling = False
    #else:
    actx = initialize_actx(actx_class, comm)
    use_profiling = actx_class_is_profiling(actx_class)
    queue = getattr(actx, "queue", None)

    # Some discretization parameters
    dim = 2
    nel_1d = 8
    order = 1

    mech_file = "uiuc_7sp"

    # {{{ Time stepping control

    # This example runs only 3 steps by default (to keep CI ~short)
    # With the mixture defined below, equilibrium is achieved at ~40ms
    # To run to equilibrium, set t_final >= 40ms.

    # Time stepper selection
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    # Time loop control parameters
    current_step = 0
    t_final = 1e-6
    current_cfl = 1.0
    current_dt = 1e-7
    current_t = 0
    constant_cfl = False

    # i.o frequencies
    nstatus = 1
    nviz = 100
    nhealth = 1
    nrestart = 5

    # }}}  Time stepping control

    debug = False

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
        assert restart_data["num_parts"] == nproc
        rst_time = restart_data["t"]
        rst_step = restart_data["step"]
        rst_order = restart_data["order"]
    else:  # generate the grid from scratch
        from meshmode.mesh.generation import generate_regular_rect_mesh
        box_ll = -0.005
        box_ur = 0.005
        from meshmode.mesh import TensorProductElementGroup
        grp_cls = TensorProductElementGroup if use_tensor_product_els else None
        generate_mesh = partial(
            generate_regular_rect_mesh, a=(box_ll,)*dim,
            b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim,
            group_cls=grp_cls)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(
        actx, local_mesh, order=order,
        tensor_product_elements=use_tensor_product_els)
    nodes = actx.thaw(dcoll.nodes())
    ones = dcoll.zeros(actx) + 1.0

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

        if log_dependent:
            logmgr_add_many_discretization_quantities(logmgr, dcoll, dim,
                                                      extract_vars_for_logging,
                                                      units_for_logging)
            logmgr.add_watches([
                ("min_pressure", "\n------- P (min, max) (Pa) = ({value:1.9e}, "),
                ("max_pressure",    "{value:1.9e})\n"),
                ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
                ("max_temperature",    "{value:7g})\n")])

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up the input data for the thermochemistry mechanism
    # --- Note: Users may add their own mechanism input file by dropping it into
    # ---       mirgecom/mechanisms alongside the other mech input files.
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mech_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temperature_seed = 1200.0  # Initial temperature hot enough to burn

    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    # which directly sets the species fractions inside cantera
    cantera_soln.set_equivalence_ratio(phi=1.0, fuel="C2H4:1",
                                       oxidizer={"O2": 1.0, "N2": 3.76})
    x = cantera_soln.X

    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm  # pylint: disable=no-member
    # one_atm = 101325.0

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temperature_seed}, {one_atm}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TP = temperature_seed, one_atm
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    can_t, can_rho, can_y = cantera_soln.TDY
    can_p = cantera_soln.P
    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Create a Pyrometheus EOS with the Cantera soln. Pyrometheus uses Cantera and
    # generates a set of methods to calculate chemothermomechanical properties and
    # states for this particular mechanism.
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyro_mechanism = \
        get_pyrometheus_wrapper_class_from_cantera(cantera_soln)(actx.np)
    eos = PyrometheusMixture(pyro_mechanism, temperature_guess=temperature_seed)

    # print out the mechanism for inspection
    # import pyrometheus as pyro
    # with open(f"mechanism.py", "w") as mech_file:
    #     code = pyro.codegen.python.gen_thermochem_code(cantera_soln)
    #     print(code, file=mech_file)

    # {{{ Initialize gas model

    gas_model = GasModel(eos=eos)

    # }}}

    # {{{ MIRGE-Com state initialization

    # Initialize the fluid/gas state with Cantera-consistent data:
    # (density, pressure, temperature, mass_fractions)
    print(f"Cantera state (rho,T,P,Y) = ({can_rho}, {can_t}, {can_p}, {can_y}")
    velocity = np.zeros(shape=(dim,))
    initializer = Uniform(dim=dim, pressure=can_p, temperature=can_t,
                          species_mass_fractions=can_y, velocity=velocity)

    from mirgecom.viscous import get_viscous_timestep

    def get_dt(state):
        return get_viscous_timestep(dcoll, state=state)

    compute_dt = actx.compile(get_dt)

    from mirgecom.viscous import get_viscous_cfl

    def get_cfl(state, dt):
        return get_viscous_cfl(dcoll, dt=dt, state=state)

    compute_cfl = actx.compile(get_cfl)

    # Evaluate species production rate
    def get_production_rates(cv, temperature):
        return eos.get_production_rates(cv, temperature)

    compute_production_rates = actx.compile(get_production_rates)

    # Evaluate energy release rate due to chemistry
    def get_heat_release_rate(state):
        return pyro_mechanism.get_heat_release_rate(state)

    compute_heat_release_rate = actx.compile(get_heat_release_rate)

    def my_get_timestep(t, dt, state):
        if use_tensor_product_els:
            return 0*state.mass_density, 0, current_dt

        #  richer interface to calculate {dt,cfl} returns node-local estimates
        t_remaining = max(0, t_final - t)

        if constant_cfl:
            ts_field = current_cfl * compute_dt(state)
            from grudge.op import nodal_min_loc
            dt = global_reduce(actx.to_numpy(nodal_min_loc(dcoll, "vol", ts_field)),
                               op="min")
            cfl = current_cfl
        else:
            ts_field = compute_cfl(state, current_dt)
            from grudge.op import nodal_max_loc
            cfl = global_reduce(actx.to_numpy(nodal_max_loc(dcoll, "vol", ts_field)),
                                op="max")
        return ts_field, cfl, min(t_remaining, dt)

    def _limit_fluid_cv(cv, temperature_seed, gas_model, dd=None):

        temperature = gas_model.eos.temperature(
            cv=cv, temperature_seed=temperature_seed)
        pressure = gas_model.eos.pressure(
            cv=cv, temperature=temperature)

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], mmin=0.0,
                                     dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                              momentum=mass_lim*cv.velocity,
                              species_mass=mass_lim*spec_lim)

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = gas_model.eos.internal_energy(cv) / cv.mass
        return pyro_mechanism.get_temperature_update_energy(e, temperature, y)

    def get_fluid_state(cv, tseed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=tseed,
                                limiter_func=_limit_fluid_cv)

    compute_temperature_update = actx.compile(get_temperature_update)
    construct_fluid_state = actx.compile(get_fluid_state)

    if rst_filename:
        current_step = rst_step
        current_t = rst_time
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
        if order == rst_order:
            current_cv = restart_data["cv"]
            temperature_seed = restart_data["temperature_seed"]
        else:
            rst_cv = restart_data["cv"]
            old_dcoll = \
                create_discretization_collection(actx, local_mesh, order=rst_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(actx, dcoll.discr_from_dd("vol"),
                                                   old_dcoll.discr_from_dd("vol"))
            current_cv = connection(rst_cv)
            temperature_seed = connection(restart_data["temperature_seed"])
    else:
        # Set the current state from time 0
        current_cv = initializer(eos=gas_model.eos, x_vec=nodes)
        temperature_seed = temperature_seed * ones

    current_cv = force_evaluation(actx, current_cv)
    temperature_seed = force_evaluation(actx, temperature_seed)

    # The temperature_seed going into this function is:
    # - At time 0: the initial temperature input data (maybe from Cantera)
    # - On restart: the restarted temperature seed from restart file (saving
    #               the *seed* allows restarts to be deterministic
    current_fluid_state = construct_fluid_state(current_cv, temperature_seed)
    current_dv = current_fluid_state.dv
    temperature_seed = current_dv.temperature

    # Inspection at physics debugging time
    if debug:
        print("Initial MIRGE-Com state:")
        print(f"Initial DV pressure: {current_fluid_state.pressure}")
        print(f"Initial DV temperature: {current_fluid_state.temperature}")

    # }}}

    visualizer = make_visualizer(dcoll)
    initname = initializer.__class__.__name__
    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)

    # Cantera equilibrate calculates the expected end state @ chemical equilibrium
    # i.e. the expected state after all reactions
    cantera_soln.equilibrate("UV")
    eq_temperature, eq_density, eq_mass_fractions = cantera_soln.TDY
    eq_pressure = cantera_soln.P

    # Report the expected final state to the user
    if rank == 0:
        logger.info(init_message)
        logger.info(f"Expected equilibrium state:"
                    f" {eq_pressure=}, {eq_temperature=},"
                    f" {eq_density=}, {eq_mass_fractions=}")

    def my_write_status(t, dt, cfl, dv=None):
        status_msg = f"------ {dt=}" if constant_cfl else f"----- {cfl=}"
        if ((dv is not None) and (not log_dependent)):

            temp = dv.temperature
            press = dv.pressure

            from grudge.op import nodal_min_loc, nodal_max_loc
            tmin = global_reduce(actx.to_numpy(nodal_min_loc(dcoll, "vol", temp)),
                                 op="min")
            tmax = global_reduce(actx.to_numpy(nodal_max_loc(dcoll, "vol", temp)),
                                 op="max")
            pmin = global_reduce(actx.to_numpy(nodal_min_loc(dcoll, "vol", press)),
                                 op="min")
            pmax = global_reduce(actx.to_numpy(nodal_max_loc(dcoll, "vol", press)),
                                 op="max")
            dv_status_msg = f"\n{t}, P({pmin}, {pmax}), T({tmin}, {tmax})"
            status_msg = status_msg + dv_status_msg

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, dt, state, ts_field, dv, production_rates,
                     heat_release_rate, cfl):
        viz_fields = [("cv", state), ("dv", dv),
                      ("production_rates", production_rates),
                      ("heat_release_rate", heat_release_rate),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)

    def my_write_restart(step, t, state, temperature_seed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname == rst_filename:
            if rank == 0:
                logger.info("Skipping overwrite of restart file.")
        else:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": state.cv,
                "temperature_seed": temperature_seed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nproc
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(cv, dv):
        health_error = False

        pressure = dv.pressure
        temperature = dv.temperature

        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        if check_range_local(dcoll, "vol", pressure, 1e5, 2.6e5):
            health_error = True
            logger.info(f"{rank=}: Pressure range violation.")

        if check_naninf_local(dcoll, "vol", temperature):
            health_error = True
            logger.info(f"{rank=}: Invalid temperature data found.")

        if check_range_local(dcoll, "vol", temperature, 1.198e3, 1.3e3):
            health_error = True
            logger.info(f"{rank=}: Temperature range violation.")

        # This check is the temperature convergence check
        # The current *temperature* is what Pyrometheus gets
        # after a fixed number of Newton iterations, *n_iter*.
        # Calling `compute_temperature` here with *temperature*
        # input as the guess returns the calculated gas temperature after
        # yet another *n_iter*.
        # The difference between those two temperatures is the
        # temperature residual, which can be used as an indicator of
        # convergence in Pyrometheus `get_temperature`.
        # Note: The local max jig below works around a very long compile
        # in lazy mode.
        temp_resid = compute_temperature_update(cv, temperature) / temperature
        temp_err = (actx.to_numpy(op.nodal_max_loc(dcoll, "vol", temp_resid)))
        if temp_err > 1e-8:
            health_error = True
            logger.info(f"{rank=}: Temperature is not converged {temp_resid=}.")

        return health_error

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed = state

        # update temperature value
        fluid_state = construct_fluid_state(cv, tseed)
        cv = fluid_state.cv
        dv = fluid_state.dv

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            ts_field, cfl, dt = my_get_timestep(t=t, dt=dt, state=fluid_state)

            if do_status:
                my_write_status(t=t, dt=dt, cfl=cfl, dv=dv)

            if do_restart:
                my_write_restart(step=step, t=t, state=fluid_state,
                                 temperature_seed=tseed)

            if do_viz:
                production_rates = compute_production_rates(fluid_state.cv,
                                                            fluid_state.temperature)
                heat_release_rate = compute_heat_release_rate(fluid_state)
                my_write_viz(step=step, t=t, dt=dt, state=cv, dv=dv,
                             production_rates=production_rates,
                             heat_release_rate=heat_release_rate,
                             ts_field=ts_field, cfl=cfl)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            # my_write_viz(step=step, t=t, dt=dt, state=cv)
            # my_write_restart(step=step, t=t, state=fluid_state)
            raise

        return make_obj_array([cv, dv.temperature]), dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

        return state, dt

    def my_rhs(t, state):
        cv, tseed = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed,
                                       limiter_func=_limit_fluid_cv)

        chem_rhs = eos.get_species_source_terms(cv, fluid_state.temperature)
        return make_obj_array([chem_rhs, fluid_state.temperature*0.0])

    current_dt = get_sim_timestep(dcoll, current_fluid_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=make_obj_array([current_cv, temperature_seed]),
                      t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    final_cv, tseed = current_state
    final_fluid_state = construct_fluid_state(final_cv, tseed)
    final_dv = final_fluid_state.dv
    final_dm = compute_production_rates(final_cv, final_dv.temperature)
    final_heat_rls = compute_heat_release_rate(final_fluid_state)
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=final_fluid_state)
    my_write_viz(step=current_step, t=current_t, dt=dt, state=final_cv,
                 dv=final_dv, production_rates=final_dm,
                 heat_release_rate=final_heat_rls, ts_field=ts_field, cfl=cfl)
    my_write_status(t=current_t, dt=dt, cfl=cfl, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=final_fluid_state,
                     temperature_seed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "autoignition"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--navierstokes", action="store_true",
                        help="turns on compressible Navier-Stokes RHS")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--esdg", action="store_true",
        help="use flux-differencing/entropy stable DG for inviscid computations.")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    parser.add_argument("--use_tensor_product_elements", action="store_true")
    args = parser.parse_args()
    from warnings import warn
    warn("Automatically turning off DV logging. MIRGE-Com Issue(578)")

    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    log_dependent = False
    viscous_terms_on = args.navierstokes

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy,
        tensor_product_elements=args.use_tensor_product_elements)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_leap=args.leap,
         use_overintegration=args.overintegration or args.esdg,
         casename=casename, rst_filename=rst_filename, use_esdg=args.esdg,
         log_dependent=log_dependent, viscous_terms_on=args.navierstokes,
         use_tensor_product_els=args.use_tensor_product_elements)

# vim: foldmethod=marker
