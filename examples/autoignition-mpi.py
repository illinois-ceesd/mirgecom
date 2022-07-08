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
import logging
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.discretization import create_discretization_collection
from grudge.dof_desc import DISCR_TAG_QUAD
from grudge.shortcuts import make_visualizer

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel
from mirgecom.utils import force_evaluation

from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    set_sim_state
)

import cantera

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception for fatal driver errors."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_overintegration=False, use_profiling=False,
         casename=None, lazy=False, rst_filename=None, log_dependent=True,
         viscous_terms_on=False):
    """Drive example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # Some discretization parameters
    dim = 2
    nel_1d = 8
    order = 1

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
    t_final = 1e-8
    current_cfl = 1.0
    current_dt = 1e-9
    current_t = 0
    constant_cfl = False

    # i.o frequencies
    nstatus = 1
    nviz = 5
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
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    discr = create_discretization_collection(actx, local_mesh, order=order,
                                             mpi_communicator=comm)
    nodes = actx.thaw(discr.nodes())
    ones = discr.zeros(actx) + 1.0

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    ones = discr.zeros(actx) + 1.0

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
            logmgr_add_many_discretization_quantities(logmgr, discr, dim,
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
    mech_input = get_mechanism_input("uiuc")

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temperature_seed = 1500.0  # Initial temperature hot enough to burn
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 3.0
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index("C2H4")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    x = np.zeros(nspecies)
    # Set the species mole fractions according to our desired fuel/air mixture
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm  # pylint: disable=no-member
    # one_atm = 101325.0

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temperature_seed}, {one_atm}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temperature_seed, one_atm, x
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
    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    pyro_mechanism = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)
    eos = PyrometheusMixture(pyro_mechanism, temperature_guess=temperature_seed)

    # {{{ Initialize simple transport model
    from mirgecom.transport import SimpleTransport
    transport_model = None
    if viscous_terms_on:
        kappa = 1e-5
        spec_diffusivity = 1e-5 * np.ones(nspecies)
        sigma = 1e-5
        transport_model = SimpleTransport(viscosity=sigma,
                                          thermal_conductivity=kappa,
                                          species_diffusivity=spec_diffusivity)
    # }}}

    gas_model = GasModel(eos=eos, transport=transport_model)
    from pytools.obj_array import make_obj_array

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = gas_model.eos.internal_energy(cv) / cv.mass
        return pyro_mechanism.get_temperature_update_energy(e, temperature, y)

    from mirgecom.gas_model import make_fluid_state

    def get_fluid_state(cv, tseed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=tseed)

    compute_temperature_update = actx.compile(get_temperature_update)
    construct_fluid_state = actx.compile(get_fluid_state)

    # }}}

    # {{{ MIRGE-Com state initialization

    # Initialize the fluid/gas state with Cantera-consistent data:
    # (density, pressure, temperature, mass_fractions)
    print(f"Cantera state (rho,T,P,Y) = ({can_rho}, {can_t}, {can_p}, {can_y}")
    velocity = np.zeros(shape=(dim,))
    initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                     pressure=can_p, temperature=can_t,
                                     massfractions=can_y, velocity=velocity)

    my_boundary = AdiabaticSlipBoundary()
    boundaries = {BTAG_ALL: my_boundary}

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
            old_discr = \
                create_discretization_collection(actx, local_mesh, order=rst_order,
                                                 mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(actx, discr.discr_from_dd("vol"),
                                                   old_discr.discr_from_dd("vol"))
            current_cv = connection(rst_cv)
            temperature_seed = connection(restart_data["temperature_seed"])
    else:
        # Set the current state from time 0
        current_cv = initializer(eos=gas_model.eos, x_vec=nodes)
        temperature_seed = temperature_seed * ones

    current_cv = force_evaluation(actx, current_cv)

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

    visualizer = make_visualizer(discr)
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

    def my_write_status(dt, cfl, dv=None):
        status_msg = f"------ {dt=}" if constant_cfl else f"----- {cfl=}"
        if ((dv is not None) and (not log_dependent)):

            temp = dv.temperature
            press = dv.pressure

            from grudge.op import nodal_min_loc, nodal_max_loc
            tmin = global_reduce(actx.to_numpy(nodal_min_loc(discr, "vol", temp)),
                                 op="min")
            tmax = global_reduce(actx.to_numpy(nodal_max_loc(discr, "vol", temp)),
                                 op="max")
            pmin = global_reduce(actx.to_numpy(nodal_min_loc(discr, "vol", press)),
                                 op="min")
            pmax = global_reduce(actx.to_numpy(nodal_max_loc(discr, "vol", press)),
                                 op="max")
            dv_status_msg = f"\nP({pmin}, {pmax}), T({tmin}, {tmax})"
            status_msg = status_msg + dv_status_msg

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, dt, state, ts_field, dv, production_rates, cfl):
        viz_fields = [("cv", state), ("dv", dv),
                      ("production_rates", production_rates),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
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
        import grudge.op as op
        health_error = False

        pressure = dv.pressure
        temperature = dv.temperature

        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        if check_range_local(discr, "vol", pressure, 1e5, 2.6e5):
            health_error = True
            logger.info(f"{rank=}: Pressure range violation.")

        if check_naninf_local(discr, "vol", temperature):
            health_error = True
            logger.info(f"{rank=}: Invalid temperature data found.")
        if check_range_local(discr, "vol", temperature, 1.498e3, 1.6e3):
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
        temp_err = (actx.to_numpy(op.nodal_max_loc(discr, "vol", temp_resid)))
        if temp_err > 1e-8:
            health_error = True
            logger.info(f"{rank=}: Temperature is not converged {temp_resid=}.")

        return health_error

    from mirgecom.viscous import get_viscous_timestep

    def get_dt(state):
        return get_viscous_timestep(discr, state=state)

    compute_dt = actx.compile(get_dt)

    from mirgecom.viscous import get_viscous_cfl

    def get_cfl(state, dt):
        return get_viscous_cfl(discr, dt=dt, state=state)

    compute_cfl = actx.compile(get_cfl)

    def get_production_rates(cv, temperature):
        return eos.get_production_rates(cv, temperature)

    compute_production_rates = actx.compile(get_production_rates)

    def my_get_timestep(t, dt, state):
        #  richer interface to calculate {dt,cfl} returns node-local estimates
        t_remaining = max(0, t_final - t)

        if constant_cfl:
            ts_field = current_cfl * compute_dt(state)
            from grudge.op import nodal_min_loc
            dt = global_reduce(actx.to_numpy(nodal_min_loc(discr, "vol", ts_field)),
                               op="min")
            cfl = current_cfl
        else:
            ts_field = compute_cfl(state, current_dt)
            from grudge.op import nodal_max_loc
            cfl = global_reduce(actx.to_numpy(nodal_max_loc(discr, "vol", ts_field)),
                                op="max")
        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = construct_fluid_state(cv, tseed)
        dv = fluid_state.dv

        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
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
                my_write_status(dt=dt, cfl=cfl, dv=dv)

            if do_restart:
                my_write_restart(step=step, t=t, state=fluid_state,
                                 temperature_seed=tseed)

            if do_viz:
                production_rates = compute_production_rates(fluid_state.cv,
                                                            fluid_state.temperature)
                my_write_viz(step=step, t=t, dt=dt, state=cv, dv=dv,
                             production_rates=production_rates,
                             ts_field=ts_field, cfl=cfl)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            # my_write_viz(step=step, t=t, dt=dt, state=cv)
            # my_write_restart(step=step, t=t, state=fluid_state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = construct_fluid_state(cv, tseed)

        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, cv, gas_model.eos)
            logmgr.tick_after()

        return make_obj_array([cv, fluid_state.temperature]), dt

    from mirgecom.inviscid import inviscid_facial_flux_rusanov as inv_num_flux_func
    from mirgecom.gas_model import make_operator_fluid_states
    from mirgecom.navierstokes import ns_operator

    fluid_operator = euler_operator
    if viscous_terms_on:
        fluid_operator = ns_operator

    def my_rhs(t, state):
        cv, tseed = state
        from mirgecom.gas_model import make_fluid_state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        fluid_operator_states = make_operator_fluid_states(
            discr, fluid_state, gas_model, boundaries=boundaries,
            quadrature_tag=quadrature_tag)
        fluid_rhs = fluid_operator(
            discr, state=fluid_state, gas_model=gas_model, time=t,
            boundaries=boundaries, operator_states_quad=fluid_operator_states,
            quadrature_tag=quadrature_tag,
            inviscid_numerical_flux_func=inv_num_flux_func)
        chem_rhs = eos.get_species_source_terms(cv, fluid_state.temperature)
        tseed_rhs = fluid_state.temperature - tseed
        cv_rhs = fluid_rhs + chem_rhs
        return make_obj_array([cv_rhs, tseed_rhs])

    current_dt = get_sim_timestep(discr, current_fluid_state, current_t, current_dt,
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
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=final_fluid_state)
    my_write_viz(step=current_step, t=current_t, dt=dt, state=final_cv,
                 dv=final_dv, production_rates=final_dm, ts_field=ts_field, cfl=cfl)
    my_write_status(dt=dt, cfl=cfl, dv=final_dv)
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
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    from warnings import warn
    warn("Automatically turning off DV logging. MIRGE-Com Issue(578)")
    log_dependent = False
    lazy = args.lazy
    viscous_terms_on = args.navierstokes
    if args.profiling:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_logmgr=args.log, use_leap=args.leap,
         use_overintegration=args.overintegration, use_profiling=args.profiling,
         lazy=lazy, casename=casename, rst_filename=rst_filename,
         log_dependent=log_dependent, viscous_terms_on=args.navierstokes)

# vim: foldmethod=marker
