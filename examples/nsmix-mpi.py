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
from pytools.obj_array import make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_QUAD

from mirgecom.discretization import create_discretization_collection
from mirgecom.transport import SimpleTransport
from mirgecom.simutil import get_sim_timestep
from mirgecom.navierstokes import ns_operator

from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (  # noqa
    AdiabaticSlipBoundary,
    IsothermalNoSlipBoundary,
)
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
import cantera

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
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=None, lazy=False,
         log_dependent=True, use_overintegration=False):
    """Drive example."""
    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # Timestepping control
    # This example runs only 3 steps by default (to keep CI ~short)
    t_final = 3e-9
    current_cfl = .0009
    current_dt = 1e-9
    current_t = 0
    constant_cfl = True
    current_step = 0
    timestepper = rk4_step
    debug = False

    # Some i/o frequencies
    nstatus = 1
    nviz = 5
    nrestart = 5
    nhealth = 1

    dim = 2
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
        nel_1d = 8
        box_ll = -0.005
        box_ur = 0.005
        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        from mirgecom.simutil import generate_and_distribute_mesh
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    order = 1
    discr = create_discretization_collection(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = actx.thaw(discr.nodes())

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    ones = discr.zeros(actx) + 1.0

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

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    init_temperature = 1500.0  # Initial temperature hot enough to burn
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
    print(f"Input state (T,P,X) = ({init_temperature}, {one_atm}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = init_temperature, one_atm, x
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    can_t, can_rho, can_y = cantera_soln.TDY
    can_p = cantera_soln.P
    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # {{{ Initialize simple transport model
    kappa = 1e-5
    spec_diffusivity = 1e-5 * np.ones(nspecies)
    sigma = 1e-5
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)
    # }}}

    # Create a Pyrometheus EOS with the Cantera soln. Pyrometheus uses Cantera and
    # generates a set of methods to calculate chemothermomechanical properties and
    # states for this particular mechanism.
    from mirgecom.thermochemistry import get_thermochemistry_class_by_mechanism_name
    pyrometheus_mechanism = \
        get_thermochemistry_class_by_mechanism_name("uiuc")(actx.np)

    pyro_eos = PyrometheusMixture(pyrometheus_mechanism,
                                  temperature_guess=init_temperature)
    gas_model = GasModel(eos=pyro_eos, transport=transport_model)

    # }}}

    # {{{ MIRGE-Com state initialization
    velocity = np.zeros(shape=(dim,))

    # Initialize the fluid/gas state with Cantera-consistent data:
    # (density, pressure, temperature, mass_fractions)
    print(f"Cantera state (rho,T,P,Y) = ({can_rho}, {can_t}, {can_p}, {can_y}")
    initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                     pressure=can_p, temperature=can_t,
                                     massfractions=can_y, velocity=velocity)

    #    my_boundary = AdiabaticSlipBoundary()
    my_boundary = IsothermalNoSlipBoundary(wall_temperature=can_t)
    visc_bnds = {BTAG_ALL: my_boundary}

    def _get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = gas_model.eos.internal_energy(cv) / cv.mass
        return actx.np.abs(
            pyrometheus_mechanism.get_temperature_update_energy(e, temperature, y))

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temp_seed)

    get_temperature_update = actx.compile(_get_temperature_update)
    get_fluid_state = actx.compile(_get_fluid_state)

    tseed = can_t
    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        tseed = restart_data["temperature_seed"]

        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_cv = initializer(x_vec=nodes, eos=gas_model.eos)
        tseed = tseed * ones

    current_state = get_fluid_state(current_cv, tseed)

    # Inspection at physics debugging time
    if debug:
        print("Initial MIRGE-Com state:")
        print(f"{current_state.mass_density=}")
        print(f"{current_state.energy_density=}")
        print(f"{current_state.momentum_density=}")
        print(f"{current_state.species_mass_density=}")
        print(f"Initial Y: {current_state.species_mass_fractions=}")
        print(f"Initial DV pressure: {current_state.temperature=}")
        print(f"Initial DV temperature: {current_state.pressure=}")

    # }}}

    visualizer = make_visualizer(discr, order + 3 if dim == 2 else order)
    initname = initializer.__class__.__name__
    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nviz=nviz, cfl=current_cfl, nstatus=1,
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

    def my_write_status(step, t, dt, dv, state):
        if constant_cfl:
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            cfl_field = get_viscous_cfl(discr, dt, state=state)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, "vol", cfl_field))
        status_msg = f"Step: {step}, T: {t}, DT: {dt}, CFL: {cfl}"

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

    def my_write_viz(step, t, cv, dv, ns_rhs=None, chem_rhs=None,
                     grad_cv=None, grad_t=None, grad_v=None):
        viz_fields = [("cv", cv),
                      ("dv", dv)]
        if ns_rhs is not None:
            viz_ext = [("nsrhs", ns_rhs),
                       ("chemrhs", chem_rhs),
                       ("grad_rho", grad_cv.mass),
                       ("grad_e", grad_cv.energy),
                       ("grad_mom_x", grad_cv.momentum[0]),
                       ("grad_mom_y", grad_cv.momentum[1]),
                       ("grad_y_1", grad_cv.species_mass[0]),
                       ("grad_v_x", grad_v[0]),
                       ("grad_v_y", grad_v[1]),
                       ("grad_temperature", grad_t)]
            viz_fields.extend(viz_ext)
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": state,
                "temperature_seed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(cv, dv):
        # Note: This health check is tuned to expected results
        #       which effectively makes this example a CI test that
        #       the case gets the expected solution.  If dt,t_final or
        #       other run parameters are changed, this check should
        #       be changed accordingly.
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(discr, "vol", dv.pressure, 9.9e4, 1.06e5),
                         op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            p_min = actx.to_numpy(nodal_min(discr, "vol", dv.pressure))
            p_max = actx.to_numpy(nodal_max(discr, "vol", dv.pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if check_naninf_local(discr, "vol", dv.temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/INFs in temperature data.")

        if global_reduce(check_range_local(discr, "vol", dv.temperature, 1450, 1570),
                         op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            t_min = actx.to_numpy(nodal_min(discr, "vol", dv.temperature))
            t_max = actx.to_numpy(nodal_max(discr, "vol", dv.temperature))
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        # This check is the temperature convergence check
        # Note: The local max jig below works around a very long compile
        # in lazy mode.
        from grudge import op
        temp_resid = get_temperature_update(cv, dv.temperature) / dv.temperature
        temp_err = (actx.to_numpy(op.nodal_max_loc(discr, "vol", temp_resid)))
        if temp_err > 1e-8:
            health_error = True
            logger.info(f"{rank=}: Temperature is not converged {temp_resid=}.")

        return health_error

    def my_pre_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = get_fluid_state(cv, tseed)
        dv = fluid_state.dv
        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step, interval=nstatus)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=cv, tseed=tseed)

            if do_viz:
                from mirgecom.fluid import velocity_gradient
                ns_rhs, grad_cv, grad_t = \
                    ns_operator(discr, state=fluid_state, time=t,
                                boundaries=visc_bnds, gas_model=gas_model,
                                return_gradients=True, quadrature_tag=quadrature_tag)
                grad_v = velocity_gradient(cv, grad_cv)
                chem_rhs = \
                    pyro_eos.get_species_source_terms(cv,
                                                      fluid_state.temperature)
                my_write_viz(step=step, t=t, cv=cv, dv=dv, ns_rhs=ns_rhs,
                             chem_rhs=chem_rhs, grad_cv=grad_cv, grad_t=grad_t,
                             grad_v=grad_v)

            dt = get_sim_timestep(discr, fluid_state, t, dt, current_cfl,
                                  t_final, constant_cfl)
            if do_status:
                my_write_status(step, t, dt, dv=dv, state=fluid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, cv=cv, dv=dv)
            my_write_restart(step=step, t=t, state=cv, tseed=tseed)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = get_fluid_state(cv, tseed)

        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, cv, gas_model.eos)
            logmgr.tick_after()

        return make_obj_array([cv, fluid_state.temperature]), dt

    flux_beta = .25

    from mirgecom.viscous import viscous_flux
    from mirgecom.flux import num_flux_central

    def _num_flux_dissipative(u_minus, u_plus, beta):
        return num_flux_central(u_minus, u_plus) + beta*(u_plus - u_minus)/2

    def _viscous_facial_flux_dissipative(discr, state_pair, grad_cv_pair,
                                         grad_t_pair, beta=0., gas_model=None):
        actx = state_pair.int.array_context
        normal = actx.thaw(discr.normal(state_pair.dd))

        f_int = viscous_flux(state_pair.int, grad_cv_pair.int,
                             grad_t_pair.int)
        f_ext = viscous_flux(state_pair.ext, grad_cv_pair.ext,
                             grad_t_pair.ext)

        return _num_flux_dissipative(f_int, f_ext, beta=beta)@normal

    grad_num_flux_func = partial(_num_flux_dissipative, beta=flux_beta)
    viscous_num_flux_func = partial(_viscous_facial_flux_dissipative,
                                    beta=-flux_beta)

    def my_rhs(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        ns_rhs = ns_operator(discr, state=fluid_state, time=t,
                             boundaries=visc_bnds, gas_model=gas_model,
                             gradient_numerical_flux_func=grad_num_flux_func,
                             viscous_numerical_flux_func=viscous_num_flux_func,
                             quadrature_tag=quadrature_tag)
        cv_rhs = ns_rhs + pyro_eos.get_species_source_terms(cv,
                                                            fluid_state.temperature)
        return make_obj_array([cv_rhs, 0*tseed])

    current_dt = get_sim_timestep(discr, current_state, current_t,
                                  current_dt, current_cfl, t_final, constant_cfl)

    current_step, current_t, current_stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=make_obj_array([current_state.cv,
                                            current_state.temperature]),
                      t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_cv, tseed = current_stepper_state
    current_state = get_fluid_state(current_cv, tseed)
    final_dv = current_state.dv
    final_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                current_cfl, t_final, constant_cfl)
    from mirgecom.fluid import velocity_gradient
    ns_rhs, grad_cv, grad_t = \
        ns_operator(discr, state=current_state, time=current_t,
                    boundaries=visc_bnds, gas_model=gas_model,
                    return_gradients=True)
    grad_v = velocity_gradient(current_state.cv, grad_cv)
    chem_rhs = \
        pyro_eos.get_species_source_terms(current_state.cv,
                                          current_state.temperature)
    my_write_viz(step=current_step, t=current_t, cv=current_state.cv, dv=final_dv,
                             chem_rhs=chem_rhs, grad_cv=grad_cv, grad_t=grad_t,
                             grad_v=grad_v)
    my_write_restart(step=current_step, t=current_t, state=current_state.cv,
                     tseed=tseed)
    my_write_status(current_step, current_t, final_dt, state=current_state,
                    dv=final_dv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "nsmix"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    lazy = args.lazy

    from warnings import warn
    warn("Automatically turning off DV logging. MIRGE-Com Issue(578)")
    log_dependent = False

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

    main(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class,
         log_dependent=log_dependent, lazy=lazy,
         use_overintegration=args.overintegration)

# vim: foldmethod=marker
