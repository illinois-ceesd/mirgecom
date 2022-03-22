"""Define the :class:`mirgecom.simutil.SimulationApplication` for NSMix."""

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
from abc import abstractmethod
from pytools.obj_array import make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer

from mirgecom.transport import SimpleTransport
from mirgecom.simutil import (
    get_sim_timestep,
    SimulationApplication as SimulationApplicationBase,
    configurate
)
from mirgecom.navierstokes import ns_operator
from mirgecom.euler import euler_operator

from mirgecom.boundary import (  # noqa
    AdiabaticSlipBoundary,
    IsothermalNoSlipBoundary,
)
from mirgecom.eos import (
    IdealSingleGas,
    PyrometheusMixture
)
from mirgecom.initializers import (
    MixtureInitializer,
)

from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
import cantera

from logpyle import set_dt

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class FluidSimulation(SimulationApplicationBase):
    """Fluid simulation prototype."""

    def __init__(self, **kwargs):
        """Initialize the dummy simulation."""
        super().__init__(**kwargs)
        self._name = "FluidSimulation"
        self._order = self._configit("element_order", 1)
        self._dim = self._configit("spatial_dimensions", 2)
        self._debug = False

        self._rst_path = self._configit("restart_path", "restart_data/")
        self._rst_pattern = self._rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"

        self._sim_cfl = self._configit("CFL", .3)
        self._sim_dt = self._configit("timestep_dt", 1e-6)
        self._t_final = self._configit("final_time", 1e-4)
        self._max_steps = self._configit("max_steps", 1000000)
        self._constant_cfl = self._configit("constant_cfl", False)

        self._current_step = 0
        self._current_time = 0.

        if self._restart_data is not None:
            self._current_time = self._restart_data["time"]
            self._current_step = self._restart_data["step"]

        self._nstatus = self._configit("nstep_status", 1)
        self._nhealth = self._configit("nstep_health", 1)
        self._nviz = self._configit("nstep_viz", 10)
        self._nrestart = self._configit("nstep_restart", 100)

        self._nel_per_axis = self._configit("nelements_per_axis", (8,)*self._dim)
        self._box_ll = self._configit("box_ll", (-0.005,)*self._dim)
        self._box_ur = self._configit("box_ur", (0.005,)*self._dim)

        self._init_velocity = self._configit("initial_velocity",
                                                np.zeros(shape=(self._dim,)))
        self._init_pressure = self._configit("initial_pressure", 10325.)
        self._init_temperature = self._configit("initial_temperature", 300.)
        self._gas_const = self._configit("gas_constant", 287.1)
        self._gamma = self._configit("gamma", 1.4)
        self._eos = IdealSingleGas(gas_const=self._gas_const,
                                   gamma=self._gamma)
        self._gas_model = GasModel(eos=self._eos)

        # user limits
        self._max_pressure = self._configit("maximum_pressure", 1000000.0)
        self._min_pressure = self._configit("minimum_pressure", 1e-12)
        self._min_temperature = self._configit("minimum_temperature", 0)
        self._max_temperature = self._configit("maximum_temperature", 1000000.)

        # Pick up any configuration customizations
        self.configure_simulation()

        # Create the grid and discretization infrastructure pieces
        self._local_mesh, self._global_nelements = self._generate_mesh()
        from mirgecom.discretization import create_dg_discretization
        self._discr = create_dg_discretization(self._actx, self._local_mesh,
                                               self._order)
        self._visualizer = make_visualizer(self._discr, self._order)

    def _generate_mesh(self):
        if self._restart_data is not None:  # read the grid from restart data
            return (
                self._restart_data["local_mesh"],
                self._restart_data["global_nelements"]
            )

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                                a=self._box_ll, b=self._box_ur,
                                nelements_per_axis=self._nel_per_axis)
        from mirgecom.simutil import generate_and_distribute_mesh
        return generate_and_distribute_mesh(self._comm, generate_mesh)

    def configure_simulation(self):
        """Customize the simulation configuration."""
        pass

    @abstractmethod
    def get_initial_fluid_state(self):
        """Return the initial fluid state at time = 0."""
        pass

    def get_initial_advancer_state(self):
        """Initialize and return the starting state for the stepper."""
        if self._restart_data is not None:
            init_cv = self._restart_data["cv"]
        else:
            # Set the current state from time 0
            initial_fluid_state = self.get_initial_fluid_state()
            init_cv = initial_fluid_state.cv

        # Inspection at physics debugging time
        if self._debug:
            logger.info("Initial MIRGE-Com state:")
            logger.info(f"{init_cv.mass=}")
            logger.info(f"{init_cv.energy=}")
            logger.info(f"{init_cv.momentum=}")
            logger.info(f"{init_cv.species_mass=}")

        return init_cv

    def get_initial_stepper_position(self):
        """Return initial stepper position."""
        return self._current_step, self._current_time

    def get_final_stepper_position(self):
        """Return final stepper position."""
        return self._max_steps, self._t_final

    def get_timestep_dt(self, stepper_state=None, stepper_time=0, stepper_dt=0):
        """Return desired DT."""
        return 1.

    def get_timestepper_method(self):
        """Return the desired timestepper."""
        from mirgecom.integrators import rk4_step
        return configurate("timestepper", self._config, rk4_step)

    def get_status_message(self, step, t, dt, cfl, state):
        """Return a status message for reporting to stdout during stepping."""
        status_msg = f"\nStep: {step}, Time: {t}, DT: {dt}, CFL: {cfl}"

        temp = state.temperature
        press = state.pressure

        from grudge.op import nodal_min_loc, nodal_max_loc
        tmin = self._allreduce(
            local_values=self._actx.to_numpy(nodal_min_loc(self._discr, "vol",
                                                           temp)), op="min")
        tmax = self._allreduce(
            local_values=self._actx.to_numpy(nodal_max_loc(self._discr, "vol",
                                                           temp)), op="max")
        pmin = self._allreduce(
            local_values=self._actx.to_numpy(nodal_min_loc(self._discr, "vol",
                                                           press)), op="min")
        pmax = self._allreduce(
            local_values=self._actx.to_numpy(nodal_max_loc(self._discr, "vol",
                                                           press)), op="max")

        dv_status_msg = (f"\n-------- Pressure({pmin}, {pmax})"
                         f"\n-------- Temperature({tmin}, {tmax})")
        status_msg = status_msg + dv_status_msg
        return status_msg

    def _write_status(self, step, t, dt, cfl, state):
        status_msg = self.get_status_message(step=step, t=t, dt=dt, cfl=cfl,
                                             state=state)
        if self._rank == 0:
            logger.info(status_msg)

    def get_viz_fields(self, step, t, state):
        """Return a dictionary with the fields to visualize."""
        return [("cv", state.cv),
                ("dv", state.dv)]

    def _write_viz(self, step, t, state):
        from mirgecom.simutil import write_visfile
        write_visfile(self._discr, self.get_viz_fields(step, t, state),
                      self._visualizer, vizname=self._casename, step=step,
                      t=t, overwrite=True)

    def make_restart_data(self, step, t, cv):
        """Return a dictionary with the fields to add to the restart file."""
        return {
            "local_mesh": self._local_mesh,
            "cv": cv,
            "t": t,
            "step": step,
            "order": self._order,
            "global_nelements": self._global_nelements,
            "num_parts": self._nparts
        }

    def _write_restart(self, step, t, restart_data):
        rst_fname = self._rst_pattern.format(cname=self._casename, step=step,
                                             rank=self._rank)
        if rst_fname != self._rst_filename:
            from mirgecom.restart import write_restart_file
            write_restart_file(self._actx, restart_data, rst_fname, self._comm)

    def _health_check(self, state):
        # Note: This health check is tuned to expected results
        #       which effectively makes this example a CI test that
        #       the case gets the expected solution.  If dt,t_final or
        #       other run parameters are changed, this check should
        #       be changed accordingly.
        health_error = False
        actx = self._actx
        discr = self._discr
        global_reduce = self._allreduce
        rank = self._rank
        dv = state.dv

        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(
                local_values=check_range_local(discr, "vol", dv.pressure,
                                               self._min_pressure,
                                               self._max_pressure), op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            p_min = actx.to_numpy(nodal_min(discr, "vol", dv.pressure))
            p_max = actx.to_numpy(nodal_max(discr, "vol", dv.pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if check_naninf_local(discr, "vol", dv.temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/INFs in temperature data.")

        if global_reduce(
                local_values=check_range_local(discr, "vol", dv.temperature,
                                               self._min_temperature,
                                               self._max_temperature), op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            t_min = actx.to_numpy(nodal_min(discr, "vol", dv.temperature))
            t_max = actx.to_numpy(nodal_max(discr, "vol", dv.temperature))
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        return health_error

    def rhs(self, time, cv):
        """Return the RHS."""
        fluid_state = make_fluid_state(cv=cv, gas_model=self._gas_model)
        return euler_operator(
            self._discr, state=fluid_state, time=time,
            boundaries=self._boundaries, gas_model=self._gas_model
        )

    def pre_step_callback(self, step, t, dt, state):
        """Perform pre-step activities."""
        cv = state

        from mirgecom.simutil import check_step
        do_viz = check_step(step=step, interval=self._nviz)
        do_restart = check_step(step=step, interval=self._nrestart)
        do_health = check_step(step=step, interval=self._nhealth)
        do_status = check_step(step, interval=self._nstatus)

        if any([self._constant_cfl, do_viz, do_health, do_status]):
            fluid_state = make_fluid_state(cv=cv, gas_model=self._gas_model)

        try:

            if self._logmgr:
                self._logmgr.tick_before()

            if do_health:
                health_errors = self._allreduce(
                    local_values=self._health_check(fluid_state), op="lor")

                if health_errors:
                    if self._rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                restart_data = self.make_restart_data(step, t, cv)
                self._write_restart(step=step, t=t, restart_data=restart_data)

            if do_viz:
                self._write_viz(step=step, t=t, state=fluid_state)

            dt = self._sim_dt
            if self._constant_cfl or do_status:
                min_dt = get_sim_timestep(self._discr, fluid_state, t, dt,
                                          1, self._t_final, True)
            if self._constant_cfl:
                dt = min_dt * self._sim_cfl

            dt = min(dt, self._t_final - t)

            if do_status:
                self._write_status(step, t, dt, dt/min_dt, state=fluid_state)

        except MyRuntimeError:
            if self._rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            # my_write_viz(step=step, t=t, state=cv, dv=dv)
            # my_write_restart(step=step, t=t, state=cv, tseed=tseed)
            raise

        return state, dt

    def post_step_callback(self, step, t, dt, state):
        """Perform post-step activities."""
        if self._logmgr:
            set_dt(self._logmgr, dt)
            self._logmgr.tick_after()

        return state, dt

    def finalize(self, step, t, state):
        """Perform finalization."""
        final_cv = state
        final_state = make_fluid_state(cv=final_cv, gas_model=self._gas_model)

        self._write_viz(step=step, t=t, state=final_state)
        restart_data = self.make_restart_data(step, t, final_cv)
        self._write_restart(step=step, t=t, restart_data=restart_data)
        self._write_status(step, t, 0, 0, state=final_state)

        finish_tol = 1e-16
        assert np.abs(self._t_final-t) < finish_tol

    @abstractmethod
    def get_boundaries(self):
        """Return a valid boundaries dictionary with any boundary conditions."""
        pass


class ViscousFluid(FluidSimulation):
    """Viscous fluid simulation prototype."""

    def __init__(self, **kwargs):
        """Initialize the dummy simulation."""
        super().__init__(**kwargs)
        self._name = "ViscousFluid"

        # Viscous fluid settings
        self._kappa = self._configit("thermal_conductivity", 1e-5)
        self._mu = self._configit("viscosity", 1e-5)

        transport_model = SimpleTransport(viscosity=self._mu,
                                          thermal_conductivity=self._kappa)
        self._gas_model = GasModel(eos=self._eos, transport=transport_model)

    def rhs(self, time, cv):
        """Return the RHS."""
        fluid_state = make_fluid_state(cv=cv, gas_model=self._gas_model)
        return ns_operator(
            self._discr, state=fluid_state, time=time,
            boundaries=self._boundaries, gas_model=self._gas_model
        )

    @abstractmethod
    def get_boundaries(self):
        """Return a valid boundaries dictionary with any boundary conditions."""
        pass


class ViscousReactiveMixture(ViscousFluid):
    """Reactive mixture with CNS operator."""

    def __init__(self, **kwargs):
        """Initialize the dummy simulation."""
        super().__init__(**kwargs)
        self._name = "ViscousReactiveMixture"

        # {{{  Set up Cantera

        # Use Cantera for initialization
        # -- Pick up a CTI for the thermochemistry config
        # --- Note: Users may add their own CTI file by dropping it into
        # ---       mirgecom/mechanisms alongside the other CTI files.
        from mirgecom.mechanisms import get_mechanism_cti
        self._mechanism_name = self._configit("mechanism_name", "uiuc")
        mech_cti = get_mechanism_cti(self._mechanism_name)

        cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
        self._nspecies = cantera_soln.n_species

        # Initial temperature, pressure, and mixutre mole fractions are needed to
        # set up the initial state in Cantera.
        # Default initial temperature hot enough to initiate combustion
        # Parameters for calculating the amounts of fuel, oxidizer, and inert species
        equiv_ratio = self._configit("equivalence_ratio", 1.0)
        ox_di_ratio = self._configit("oxidizer_ratio", 0.21)
        stoich_ratio = self._configit("stoichiometric_ratio", 3.0)
        # Grab the array indices for the specific species, fuel, oxygen,
        # and nitrogen
        self._fuel_name = self._configit("fuel_species_name", "C2H4")
        i_fu = cantera_soln.species_index(self._fuel_name)
        i_ox = cantera_soln.species_index("O2")
        i_di = cantera_soln.species_index("N2")
        x = np.zeros(self._nspecies)
        # Set the species mole fractions according to our desired fuel/air mixture
        x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
        x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
        x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
        # Uncomment next line to make pylint fail when it can't find cantera.one_atm
        one_atm = cantera.one_atm  # pylint: disable=no-member
        self._species_names = cantera_soln.species_names
        # Let the user know about how Cantera is being initilized
        if self._rank == 0:
            logger.info(f"Input state (T,P,X) = ({self._init_temperature},"
                        f" {one_atm}, {x}")

        # Set Cantera internal gas temperature, pressure, and mole fractios
        cantera_soln.TPX = self._init_temperature, one_atm, x
        # Pull temperature, total density, mass fractions, and pressure from Cantera
        # We need tot density, and mass fractions to initialize the fluid/gas state.
        can_t, can_rho, can_y = cantera_soln.TDY
        can_p = cantera_soln.P
        # *can_t*, *can_p* should not differ (much) from user's initial data,
        # but to nsure that we use exactly the same starting point as Cantera,
        # we use Cantera's version of these data.

        # Cantera equilibrate calculates expected end state @ chemical equilibrium
        # i.e. the expected state after all reactions
        cantera_soln.equilibrate("UV")
        eq_temperature, eq_density, eq_mass_fractions = cantera_soln.TDY
        eq_pressure = cantera_soln.P
        self._tseed = can_t

        # Report the expected final state to the user
        if self._rank == 0:
            logger.info(f"Expected equilibrium state:"
                        f" {eq_pressure=}, {eq_temperature=},"
                        f" {eq_density=}, {eq_mass_fractions=}")
        # }}}

        # {{{ Create Pyrometheus thermochemistry object, EOS, and transport model

        self._spec_diffusivity = self._configit("species_diffusivities",
                                                1e-5 * np.ones(self._nspecies))
        transport_model = SimpleTransport(viscosity=self._mu,
                                          thermal_conductivity=self._kappa,
                                          species_diffusivity=self._spec_diffusivity)

        # Create a Pyrometheus object with cantera, then EOS with Pyro object.
        from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
        pyrometheus_mechanism = \
            make_pyrometheus_mechanism_class(cantera_soln)(self._actx.np)
        self._eos = PyrometheusMixture(pyrometheus_mechanism,
                                       temperature_guess=self._init_temperature)
        self._gas_model = GasModel(eos=self._eos, transport=transport_model)

        # }}}

        # Initialize the fluid/gas state with Cantera-consistent data:
        # (density, pressure, temperature, mass_fractions)
        if self._rank == 0:
            logger.info(f"Cantera state (rho,T,P,Y) = ({can_rho}, {can_t},"
                        f" {can_p}, {can_y}")

        self._initializer = \
            MixtureInitializer(dim=self._dim, nspecies=self._nspecies,
                               pressure=can_p, temperature=can_t,
                               massfractions=can_y, velocity=self._init_velocity)

        def _get_temperature_update(cv, temperature):
            y = cv.species_mass_fractions
            e = self._gas_model.eos.internal_energy(cv) / cv.mass
            return pyrometheus_mechanism.get_temperature_update_energy(
                e, temperature, y
            )

        def _get_fluid_state(cv, temp_seed):
            return make_fluid_state(cv=cv, gas_model=self._gas_model,
                                    temperature_seed=temp_seed)

        self._compute_temperature_update = \
            self._actx.compile(_get_temperature_update)
        self._construct_fluid_state = \
            self._actx.compile(_get_fluid_state)

        # }}} - case specific setup

    @abstractmethod
    def get_boundaries(self):
        """Return a valid boundaries dictionary with any boundary conditions."""
        pass

    @abstractmethod
    def get_initial_fluid_state(self):
        """Return the initial fluid state at time = 0."""
        pass

    def _make_stepper_state(self, cv, tseed):
        return make_obj_array([cv, tseed])

    def get_initial_advancer_state(self):
        """Initialize and return the starting state for the stepper."""
        if self._restart_data is not None:
            init_cv = self._restart_data["cv"]
            init_tseed = self._restart_data["temperature_seed"]
        else:
            # Set the current state from time 0
            initial_fluid_state = self.get_initial_fluid_state()
            init_cv = initial_fluid_state.cv
            init_tseed = initial_fluid_state.temperature

        # Inspection at physics debugging time
        if self._debug:
            logger.info("Initial MIRGE-Com state:")
            logger.info(f"{init_cv.mass=}")
            logger.info(f"{init_cv.energy=}")
            logger.info(f"{init_cv.momentum=}")
            logger.info(f"{init_cv.species_mass=}")
            logger.info(f"Initial Y: {init_cv.species_mass_fractions=}")

        return self._make_stepper_state(cv=init_cv, tseed=init_tseed)

    def get_status_message(self, step, t, dt, cfl, state):
        """Return a status message for reporting to stdout during stepping."""
        status_msg = super().get_status_message(step=step, t=t, dt=dt,
                                                cfl=cfl, state=state)

        def vol_min(x):
            from grudge.op import nodal_min
            return self._actx.to_numpy(nodal_min(self._discr, "vol", x))[()]

        def vol_max(x):
            from grudge.op import nodal_max
            return self._actx.to_numpy(nodal_max(self._discr, "vol", x))[()]

        from pytools.obj_array import obj_array_vectorize
        y_min = obj_array_vectorize(lambda x: vol_min(x),
                                      state.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(x),
                                      state.species_mass_fractions)
        for i in range(self._nspecies):
            status_msg += (
                f"\n-------- y_{self._species_names[i]} (min, max) = "
                f"({y_min[i]:1.3e}, {y_max[i]:1.3e})")

        return status_msg

    def get_viz_fields(self, step, t, state):
        """Return a dictionary with the fields to visualize."""
        viz_fields = super().get_viz_fields(step, t, state)
        viz_fields.extend(
            ("Y_"+self._species_names[i], state.species_mass_fractions[i])
            for i in range(self._nspecies))
        return viz_fields

    def make_restart_data(self, step, t, cv, tseed):
        """Return a dictionary with the fields to add to the restart file."""
        restart_data = super().make_restart_data(step, t, cv)
        restart_data["temperature_seed"] = tseed
        return restart_data

    def _health_check(self, state):
        health_error = super()._health_check(state)

        actx = self._actx
        discr = self._discr
        cv = state.cv
        dv = state.dv

        compute_temperature_update = self._compute_temperature_update

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
        from grudge import op
        temp_resid = compute_temperature_update(cv, dv.temperature) / dv.temperature
        temp_err = (actx.to_numpy(op.nodal_max_loc(discr, "vol", temp_resid)))
        if temp_err > 1e-8:
            health_error = True
            logger.info(f"{self._rank=}: Temperature is not converged"
                        f" {temp_resid=}.")

        return health_error

    def rhs(self, time, stepper_state):
        """Return the RHS."""
        from pytools.obj_array import make_obj_array
        cv, tseed = stepper_state
        fluid_state = make_fluid_state(cv=cv, gas_model=self._gas_model,
                                       temperature_seed=tseed)
        ns_rhs = ns_operator(
            self._discr, state=fluid_state, time=time,
            boundaries=self._boundaries, gas_model=self._gas_model
        )

        cv_rhs = (
            ns_rhs
            + self._eos.get_species_source_terms(
                cv, fluid_state.temperature)
        )

        return make_obj_array([cv_rhs, fluid_state.temperature-tseed])

    def pre_step_callback(self, step, t, dt, state):
        """Perform pre-step activities."""
        # print(f"{state=}")
        cv, tseed = state

        from mirgecom.simutil import check_step
        do_viz = check_step(step=step, interval=self._nviz)
        do_restart = check_step(step=step, interval=self._nrestart)
        do_health = check_step(step=step, interval=self._nhealth)
        do_status = check_step(step, interval=self._nstatus)

        if any([self._constant_cfl, do_viz, do_health, do_status]):
            fluid_state = self._construct_fluid_state(cv, tseed)

        try:

            if self._logmgr:
                self._logmgr.tick_before()

            if do_health:
                health_errors = self._allreduce(
                    local_values=self._health_check(fluid_state), op="lor")

                if health_errors:
                    if self._rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                restart_data = self.make_restart_data(step, t, cv, tseed)
                self._write_restart(step=step, t=t, restart_data=restart_data)

            if do_viz:
                self._write_viz(step=step, t=t, state=fluid_state)

            dt = self._sim_dt
            if self._constant_cfl or do_status:
                min_dt = get_sim_timestep(self._discr, fluid_state, t, dt,
                                          1, self._t_final, True)
            if self._constant_cfl:
                dt = min_dt * self._sim_cfl

            dt = min(dt, self._t_final - t)

            if do_status:
                self._write_status(step, t, dt, dt/min_dt, state=fluid_state)

        except MyRuntimeError:
            if self._rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            # my_write_viz(step=step, t=t, state=cv, dv=dv)
            # my_write_restart(step=step, t=t, state=cv, tseed=tseed)
            raise

        return state, dt

    def post_step_callback(self, step, t, dt, state):
        """Perform post-step activities."""
        cv, tseed = state

        if self._logmgr:
            set_dt(self._logmgr, dt)
            self._logmgr.tick_after()

        return state, dt

    def finalize(self, step, t, state):
        """Perform finalization."""
        final_cv, tseed = state
        final_state = self._construct_fluid_state(final_cv, tseed)

        self._write_viz(step=step, t=t, state=final_state)
        restart_data = self.make_restart_data(step, t, final_cv, tseed)
        self._write_restart(step=step, t=t, restart_data=restart_data)
        self._write_status(step, t, 0, 0, state=final_state)

        finish_tol = 1e-16
        assert np.abs(self._t_final-t) < finish_tol

# vim: foldmethod=marker
