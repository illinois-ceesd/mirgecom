"""Demonstrate a generic gas example."""

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
import argparse
import numpy as np
from functools import partial

from meshmode.mesh import BTAG_ALL
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_QUAD, BoundaryDomainTag
import grudge.op as op

from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array
from mirgecom.mpi import mpi_entry_point
from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import euler_operator
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    get_sim_timestep,
    distribute_mesh
)
from mirgecom.utils import force_evaluation
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step, euler_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    AdiabaticSlipBoundary,
    IsothermalWallBoundary
)
from mirgecom.initializers import (
    Uniform,
    AcousticPulse
)
from mirgecom.eos import (
    IdealSingleGas,
    PyrometheusMixture
)
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.transport import (
    SimpleTransport,
    MixtureAveragedTransport,
    PowerLawTransport,
    ArtificialViscosityTransportDiv,
    ArtificialViscosityTransportDiv2,
    ArtificialViscosityTransportDiv3
)
from mirgecom.limiter import bound_preserving_limiter
from mirgecom.fluid import make_conserved
from mirgecom.logging_quantities import (
    initialize_logmgr,
    # logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage
)
import cantera

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, use_esdg=False, use_tpe=False,
         use_overintegration=False, use_leap=False,
         casename=None, rst_filename=None, dim=None,
         periodic_mesh=False, multiple_boundaries=False,
         use_navierstokes=False, use_mixture=False,
         use_reactions=False, newton_iters=3,
         mech_name="uiuc_7sp", transport_type=0,
         use_av=0, use_limiter=False, order=1,
         nscale=1, npassive_species=0, map_mesh=False,
         rotation_angle=0, add_pulse=False, nsteps=20,
         mesh_filename=None, euler_timestepping=False):
    """Drive the example."""
    if casename is None:
        casename = "gas-in-box"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=use_tpe,
                           use_einsum_inference_fallback=use_tpe)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = euler_step if euler_timestepping else rk4_step

    current_cfl = 1.0
    current_dt = 1e-6
    t_final = current_dt * nsteps
    current_t = 0
    constant_cfl = False
    temperature_tolerance = 1e-2

    # some i/o frequencies
    nstatus = 1
    nrestart = 100
    nviz = 10
    nhealth = 1

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
        if mesh_filename is not None:
            from meshmode.mesh.io import read_gmsh
            mesh_construction_kwargs = {
                "force_positive_orientation": True,
                "skip_tests": False
            }
            if dim is None or (dim == 3):
                generate_mesh = partial(
                    read_gmsh, filename=mesh_filename,
                    mesh_construction_kwargs=mesh_construction_kwargs
                )
            else:
                generate_mesh = partial(
                    read_gmsh, filename=mesh_filename,
                    mesh_construction_kwargs=mesh_construction_kwargs,
                    force_ambient_dim=dim
                )
        else:
            if dim is None:
                dim = 2
            nscale = max(nscale, 1)
            scale_fac = pow(float(nscale), 1.0/dim)
            nel_1d = int(scale_fac*24/dim)
            from mirgecom.simutil import get_box_mesh
            box_ll = -1
            box_ur = 1
            generate_mesh = partial(
                get_box_mesh, dim=dim, a=(box_ll,)*dim, b=(box_ur,)*dim,
                n=(nel_1d,)*dim, periodic=(periodic_mesh,)*dim,
                tensor_product_elements=use_tpe)

        local_mesh, global_nelements = distribute_mesh(comm, generate_mesh)
        local_nelements = local_mesh.nelements

        if dim is None:
            dim = local_mesh.ambient_dim

        def add_wonk(x: np.ndarray) -> np.ndarray:
            wonk_field = np.empty_like(x)
            if len(x) >= 2:
                wonk_field[0] = (
                    1.5*x[0] + np.cos(x[0])
                    + 0.1*np.sin(10*x[1]))
                wonk_field[1] = (
                    0.05*np.cos(10*x[0])
                    + 1.3*x[1] + np.sin(x[1]))
            else:
                wonk_field[0] = 1.5*x[0] + np.cos(x[0])

            if len(x) >= 3:
                wonk_field[2] = x[2] + np.sin(x[0] / 2) / 2
            return wonk_field

        if map_mesh:
            from meshmode.mesh.processing import map_mesh
            local_mesh = map_mesh(local_mesh, add_wonk)

        if abs(rotation_angle) > 0:
            from meshmode.mesh.processing import rotate_mesh_around_axis
            theta = rotation_angle/180.0 * np.pi
            local_mesh = rotate_mesh_around_axis(local_mesh, theta=theta)

    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    ones = dcoll.zeros(actx) + 1.

    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

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

    velocity = np.zeros(shape=(dim,))

    species_diffusivity = None
    speedup_factor = 1.0
    pyro_mechanism = None
    if use_mixture:
        # {{{  Set up initial state using Cantera

        # Use Cantera for initialization
        # -- Pick up the input data for the thermochemistry mechanism
        # --- Note: Users may add their own mechanism input file by dropping it into
        # ---       mirgecom/mechanisms alongside the other mech input files.
        from mirgecom.mechanisms import get_mechanism_input
        mech_input = get_mechanism_input(mech_name)

        cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
        nspecies = cantera_soln.n_species

        species_diffusivity = 1e-5 * np.ones(nspecies)
        # Initial temperature, pressure, and mixutre mole fractions are needed to
        # set up the initial state in Cantera.
        temperature_seed = 1200.0  # Initial temperature hot enough to burn

        # Parameters for calculating the amounts of fuel, oxidizer, and inert species
        # which directly sets the species fractions inside cantera
        cantera_soln.set_equivalence_ratio(phi=1.0, fuel="C2H4:1",
                                           oxidizer={"O2": 1.0, "N2": 3.76})
        x = cantera_soln.X

        one_atm = cantera.one_atm  # pylint: disable=no-member

        # Let the user know about how Cantera is being initilized
        print(f"Input state (T,P,X) = ({temperature_seed}, {one_atm}, {x}")
        # Set Cantera internal gas temperature, pressure, and mole fractios
        cantera_soln.TP = temperature_seed, one_atm
        # Pull temperature, total density, mass fractions, and pressure
        # from Cantera. We need total density, and mass fractions to initialize
        # the fluid/gas state.
        can_t, can_rho, can_y = cantera_soln.TDY
        can_p = cantera_soln.P
        # *can_t*, *can_p* should not differ (significantly) from user's
        # initial data, but we want to ensure that we use exactly the same
        # starting point as Cantera, so we use Cantera's version of these data.

        # }}}

        # {{{ Create Pyrometheus thermochemistry object & EOS

        # Create a Pyrometheus EOS with the Cantera soln. Pyrometheus uses
        # Cantera and generates a set of methods to calculate chemothermomechanical
        # properties and states for this particular mechanism.
        from mirgecom.thermochemistry import \
            get_pyrometheus_wrapper_class_from_cantera
        pyro_mechanism = \
            get_pyrometheus_wrapper_class_from_cantera(
                cantera_soln, temperature_niter=newton_iters)(actx.np)
        eos = PyrometheusMixture(pyro_mechanism, temperature_guess=temperature_seed)
        initializer = Uniform(dim=dim, pressure=can_p, temperature=can_t,
                              species_mass_fractions=can_y, velocity=velocity)
        init_t = can_t
    else:
        use_reactions = False
        eos = IdealSingleGas(gamma=1.4)
        species_y = None
        if npassive_species > 0:
            print(f"Initializing with {npassive_species} passive species.")
            nspecies = npassive_species
            spec_diff = 1e-4
            species_y = np.array([1./float(nspecies) for _ in range(nspecies)])
            species_diffusivity = np.array([spec_diff * 1./float(j+1)
                                            for j in range(nspecies)])

        initializer = Uniform(velocity=velocity, pressure=101325,
                              rho=1.2039086127319172,
                              species_mass_fractions=species_y)
        init_t = 293.15

    temperature_seed = init_t * ones
    temperature_seed = force_evaluation(actx, temperature_seed)

    wall_bc = IsothermalWallBoundary(wall_temperature=init_t) \
        if use_navierstokes else AdiabaticSlipBoundary()

    # initialize parameters for transport model
    transport = None
    thermal_conductivity = 1e-5
    viscosity = 1.0e-5
    transport_alpha = 0.6
    transport_beta = 4.093e-7
    transport_sigma = 2.0
    transport_n = 0.666

    av2_mu0 = 0.1
    av2_beta0 = 6.0
    av2_kappa0 = 1.0
    av2_d0 = 0.1
    av2_prandtl0 = 0.9
    # av2_mu_s0 = 0.
    # av2_kappa_s0 = 0.
    # av2_beta_s0 = .01
    # av2_d_s0 = 0.

    # use_av=1 specific parameters
    # flow stagnation temperature
    static_temp = 2076.43
    # steepness of the smoothed function
    theta_sc = 100
    # cutoff, smoothness below this value is ignored
    beta_sc = 0.01
    gamma_sc = 1.5
    alpha_sc = 0.3
    kappa_sc = 0.5
    s0_sc = np.log10(1.0e-4 / np.power(order, 4))

    smoothness_alpha = 0.1
    smoothness_tau = .01
    physical_transport_model = None

    if use_navierstokes:
        if transport_type == 2:
            if not use_mixture:
                error_message = "Invalid transport_type "\
                    "{} for single gas.".format(transport_type)
                raise RuntimeError(error_message)
            if rank == 0:
                print("Pyrometheus transport model:")
                print("\t temperature/mass fraction dependence")
            physical_transport_model = \
                MixtureAveragedTransport(pyro_mechanism,
                                         factor=speedup_factor)
        elif transport_type == 0:
            if rank == 0:
                print("Simple transport model:")
                print("\tconstant viscosity, species diffusivity")
                print(f"\tmu = {viscosity}")
                print(f"\tkappa = {thermal_conductivity}")
                print(f"\tspecies diffusivity = {species_diffusivity}")
            physical_transport_model = SimpleTransport(
                viscosity=viscosity, thermal_conductivity=thermal_conductivity,
                species_diffusivity=species_diffusivity)
        elif transport_type == 1:
            if rank == 0:
                print("Power law transport model:")
                print("\ttemperature dependent viscosity, species diffusivity")
                print(f"\ttransport_alpha = {transport_alpha}")
                print(f"\ttransport_beta = {transport_beta}")
                print(f"\ttransport_sigma = {transport_sigma}")
                print(f"\ttransport_n = {transport_n}")
                print(f"\tspecies diffusivity = {species_diffusivity}")
            physical_transport_model = PowerLawTransport(
                alpha=transport_alpha, beta=transport_beta,
                sigma=transport_sigma, n=transport_n,
                species_diffusivity=species_diffusivity)
        else:
            error_message = "Unknown transport_type {}".format(transport_type)
            raise RuntimeError(error_message)

    transport = physical_transport_model
    if use_av == 1:
        transport = ArtificialViscosityTransportDiv(
            physical_transport=physical_transport_model,
            av_mu=alpha_sc, av_prandtl=0.75)
    elif use_av == 2:
        transport = ArtificialViscosityTransportDiv2(
            physical_transport=physical_transport_model,
            av_mu=av2_mu0, av_beta=av2_beta0, av_kappa=av2_kappa0,
            av_prandtl=av2_prandtl0)
    elif use_av == 3:
        transport = ArtificialViscosityTransportDiv3(
            physical_transport=physical_transport_model,
            av_mu=av2_mu0, av_beta=av2_beta0,
            av_kappa=av2_kappa0, av_d=av2_d0,
            av_prandtl=av2_prandtl0)

    if rank == 0 and use_navierstokes and use_av > 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")
        print(f"Artificial viscosity {smoothness_alpha=}")
        print(f"Artificial viscosity {smoothness_tau=}")

        if use_av == 1:
            print("Artificial viscosity using modified physical viscosity")
            print("Using velocity divergence indicator")
            print(f"Shock capturing parameters: alpha {alpha_sc}, "
                  f"gamma_sc {gamma_sc}"
                  f"theta_sc {theta_sc}, beta_sc {beta_sc}, Pr 0.75, "
                  f"stagnation temperature {static_temp}")
        elif use_av == 2:
            print("Artificial viscosity using modified transport properties")
            print("\t mu, beta, kappa")
            # MJA update this
            print(f"Shock capturing parameters:"
                  f"\n\tav_mu {av2_mu0}"
                  f"\n\tav_beta {av2_beta0}"
                  f"\n\tav_kappa {av2_kappa0}"
                  f"\n\tav_prantdl {av2_prandtl0}"
                  f"\nstagnation temperature {static_temp}")
        elif use_av == 3:
            print("Artificial viscosity using modified transport properties")
            print("\t mu, beta, kappa, D")
            print(f"Shock capturing parameters:"
                  f"\tav_mu {av2_mu0}"
                  f"\tav_beta {av2_beta0}"
                  f"\tav_kappa {av2_kappa0}"
                  f"\tav_d {av2_d0}"
                  f"\tav_prantdl {av2_prandtl0}"
                  f"stagnation temperature {static_temp}")
        else:
            error_message = "Unknown artifical viscosity model {}".format(use_av)
            raise RuntimeError(error_message)

    gas_model = GasModel(eos=eos, transport=transport)
    fluid_operator = ns_operator if use_navierstokes else euler_operator
    orig = np.zeros(shape=(dim,))
    uniform_cv = initializer(nodes, eos=eos)

    def mixture_mass_fraction_limiter(cv, temperature_seed, gas_model, dd=None):

        temperature = gas_model.eos.temperature(
            cv=cv, temperature_seed=temperature_seed)
        pressure = gas_model.eos.pressure(
            cv=cv, temperature=temperature)

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                                     mmin=0.0, mmax=1.0, modify_average=True,
                                     dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = gas_model.eos.get_density(pressure=pressure,
                                             temperature=temperature,
                                             species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        cv_lim = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                                momentum=mass_lim*cv.velocity,
                                species_mass=mass_lim*spec_lim)

        # return make_obj_array([cv_lim, pressure, temperature])
        return cv_lim

    limiter_func = mixture_mass_fraction_limiter if use_limiter else None

    def my_limiter(cv, tseed):
        if limiter_func is not None:
            return limiter_func(cv, tseed, gas_model=gas_model)
        return cv

    limiter_compiled = actx.compile(my_limiter)

    def stepper_state_to_gas_state(stepper_state):
        if use_mixture:
            cv, tseed = stepper_state
            return make_fluid_state(cv=cv, gas_model=gas_model,
                                    limiter_func=limiter_func,
                                    temperature_seed=tseed)
        else:
            return make_fluid_state(cv=stepper_state, gas_model=gas_model)

    def gas_rhs_to_stepper_rhs(gas_rhs, gas_temperature):
        if use_mixture:
            return make_obj_array([gas_rhs, 0.*gas_temperature])
        else:
            return gas_rhs

    def gas_state_to_stepper_state(gas_state):
        if use_mixture:
            return make_obj_array([gas_state.cv, gas_state.temperature])
        else:
            return gas_state.cv

    boundaries = {}
    if not periodic_mesh:
        if multiple_boundaries:
            for idir in range(dim):
                boundaries[BoundaryDomainTag(f"+{idir+1}")] = wall_bc
                boundaries[BoundaryDomainTag(f"-{idir+1}")] = wall_bc
        else:
            boundaries = {BTAG_ALL: wall_bc}

    def mfs(cv, tseed):
        return make_fluid_state(cv, gas_model, limiter_func=limiter_func,
                                temperature_seed=tseed)

    mfs_compiled = actx.compile(mfs)

    def get_temperature_update(cv, temperature):
        if pyro_mechanism is not None:
            y = cv.species_mass_fractions
            e = gas_model.eos.internal_energy(cv) / cv.mass
            return pyro_mechanism.get_temperature_update_energy(e, temperature, y)
        else:
            return 0*temperature

    gtu_compiled = actx.compile(get_temperature_update)

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        rst_tseed = restart_data["temperature_seed"]
        current_cv = force_evaluation(actx, current_cv)
        current_gas_state = mfs_compiled(current_cv, rst_tseed)
    else:
        # Set the current state from time 0
        if add_pulse:
            acoustic_pulse = AcousticPulse(dim=dim, amplitude=100., width=.1,
                                           center=orig)
            current_cv = acoustic_pulse(x_vec=nodes, cv=uniform_cv, eos=eos,
                                        tseed=temperature_seed)
        else:
            current_cv = uniform_cv
        current_cv = force_evaluation(actx, current_cv)
        # Force to use/compile limiter so we can evaluate DAG
        if limiter_func is not None:
            current_cv = limiter_compiled(current_cv, temperature_seed)

        current_gas_state = mfs_compiled(current_cv, temperature_seed)

    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, current_step, current_t)

    visualizer = make_visualizer(dcoll)

    initname = "gas-in-box"
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

    def my_write_viz(step, t, gas_state):
        dv = gas_state.dv
        cv = gas_state.cv
        viz_fields = [("cv", cv),
                      ("dv", dv)]
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)

    def my_write_restart(step, t, gas_state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": gas_state.cv,
                "temperature_seed": gas_state.temperature,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(gas_state):
        pressure = gas_state.pressure
        temperature = gas_state.temperature

        health_error = False
        from mirgecom.simutil import check_naninf_local
        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")
        if check_naninf_local(dcoll, "vol", temperature):
            health_error = True
            logger.info(f"{rank=}: Invalid temperature data found.")

        if gas_state.is_mixture:
            temper_update = gtu_compiled(gas_state.cv, gas_state.temperature)
            temp_relup = temper_update / gas_state.temperature
            max_temp_relup = (actx.to_numpy(op.nodal_max_loc(dcoll, "vol",
                                                             temp_relup)))
            if max_temp_relup > temperature_tolerance:
                health_error = True
                logger.info(f"{rank=}: Temperature is not "
                            f"converged {max_temp_relup=}.")

        return health_error

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        stepper_state = state
        gas_state = stepper_state_to_gas_state(stepper_state)
        gas_state = force_evaluation(actx, gas_state)

        try:

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(my_health_check(gas_state), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, gas_state=gas_state)

            if do_viz:
                my_write_viz(step=step, t=t, gas_state=gas_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, gas_state=gas_state)
            my_write_restart(step=step, t=t, gas_state=gas_state)
            raise

        dt = get_sim_timestep(dcoll, gas_state.cv, t, dt, current_cfl, t_final,
                              constant_cfl)
        return gas_state_to_stepper_state(gas_state), dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, stepper_state):
        gas_state = stepper_state_to_gas_state(stepper_state)
        gas_rhs = fluid_operator(dcoll, state=gas_state, time=t,
                                   boundaries=boundaries,
                                   gas_model=gas_model, use_esdg=use_esdg,
                                   quadrature_tag=quadrature_tag)
        if use_reactions:
            gas_rhs = \
                gas_rhs + eos.get_species_source_terms(gas_state.cv,
                                                       gas_state.temperature)
        return gas_rhs_to_stepper_rhs(gas_rhs, gas_state.temperature)

    current_dt = get_sim_timestep(dcoll, current_gas_state.cv, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_stepper_state = gas_state_to_stepper_state(current_gas_state)
    current_step, current_t, current_stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_stepper_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_gas_state = stepper_state_to_gas_state(current_stepper_state)

    my_write_viz(step=current_step, t=current_t, gas_state=final_gas_state)
    my_write_restart(step=current_step, t=current_t, gas_state=final_gas_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    example_name = "gas-in-box"
    parser = argparse.ArgumentParser(
        description=f"MIRGE-Com Example: {example_name}")
    parser.add_argument("-o", "--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("-l", "--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--esdg", action="store_true",
        help="use entropy-stable dg for inviscid terms.")
    parser.add_argument("--euler-timestepping", action="store_true",
                        help="use euler timestepping")
    parser.add_argument("--nsteps", type=int, default=20,
                        help="number of timesteps to take")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("-d", "--dimension", type=int, choices=[1, 2, 3],
                        help="spatial dimension of simulation")
    parser.add_argument("-i", "--iters", type=int, default=1,
                        help="number of Newton iterations for mixture temperature")
    parser.add_argument("-r", "--restart_file", help="root name of restart file")
    parser.add_argument("-c", "--casename", help="casename to use for i/o")
    parser.add_argument("-p", "--periodic", action="store_true",
                        help="use periodic boundaries")
    parser.add_argument("-n", "--navierstokes", action="store_true",
                        help="use Navier-Stokes operator")
    parser.add_argument("-a", "--artificial-viscosity", type=int,
                        choices=[0, 1, 2, 3],
                        default=0, help="use artificial viscosity")
    parser.add_argument("-b", "--boundaries", action="store_true",
                        help="use multiple (2*ndim) boundaries")
    parser.add_argument("-k", "--wonky", action="store_true", default=False,
                        help="make a wonky mesh")
    parser.add_argument("-m", "--mixture", action="store_true",
                        help="use gas mixture EOS")
    parser.add_argument("-f", "--flame", action="store_true",
                        help="use combustion chemistry")
    parser.add_argument("-g", "--rotate", type=float, default=0,
                        help="rotate mesh by angle (degrees)")
    parser.add_argument("-x", "--transport", type=int, choices=[0, 1, 2], default=0,
                        help=("transport model specification\n"
                              + "(0)Simple\n(1)PowerLaw\n(2)Mix"))
    parser.add_argument("-e", "--limiter", action="store_true",
                        help="use limiter to limit fluid state")
    parser.add_argument("-s", "--species", type=int, default=0,
                        help="number of passive species")
    parser.add_argument("-t", "--tpe", action="store_true",
                        help="use tensor-product elements (quads/hexes)")
    parser.add_argument("-u", "--pulse", action="store_true", default=False,
                        help="add an acoustic pulse at the origin")
    parser.add_argument("-y", "--polynomial-order", type=int, default=1,
                        help="polynomal order for the discretization")
    parser.add_argument("-w", "--weak-scale", type=int, default=1,
                        help="factor by which to scale the number of elements")
    parser.add_argument("-z", "--mechanism-name", type=str, default="uiuc_7sp",
                        help="name of thermochemical mechanism yaml file")
    parser.add_argument("--meshfile", type=str,
                        help="name of gmsh input file")
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

    main(actx_class, use_esdg=args.esdg, dim=args.dimension,
         use_overintegration=args.overintegration or args.esdg,
         use_leap=args.leap, use_tpe=args.tpe, nsteps=args.nsteps,
         casename=args.casename, rst_filename=rst_filename,
         periodic_mesh=args.periodic, use_mixture=args.mixture,
         multiple_boundaries=args.boundaries,
         transport_type=args.transport, order=args.polynomial_order,
         use_limiter=args.limiter, use_av=args.artificial_viscosity,
         use_reactions=args.flame, newton_iters=args.iters,
         use_navierstokes=args.navierstokes, npassive_species=args.species,
         nscale=args.weak_scale, mech_name=args.mechanism_name,
         map_mesh=args.wonky, rotation_angle=args.rotate, add_pulse=args.pulse,
         mesh_filename=args.meshfile, euler_timestepping=args.euler_timestepping)

# vim: foldmethod=marker
