"""
:mod:`mirgecom.eos` provides implementations of gas equations of state.

Equations of State
^^^^^^^^^^^^^^^^^^
This module is designed provide Equation of State objects used to compute and
manage the relationships between and among state and thermodynamic variables.

.. autoclass:: GasDependentVars
.. autoclass:: MixtureDependentVars
.. autoclass:: GasEOS
.. autoclass:: MixtureEOS
.. autoclass:: IdealSingleGas
.. autoclass:: PyrometheusMixture

Exceptions
^^^^^^^^^^
.. autoexception:: TemperatureSeedMissingError
.. autoexception:: MixtureEOSNeededError
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
from typing import Union, Optional
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from arraycontext import dataclass_array_container
import numpy as np
from meshmode.dof_array import DOFArray
from mirgecom.fluid import ConservedVars, make_conserved


class TemperatureSeedMissingError(Exception):
    """Indicate that EOS is inappropriately called without seeding temperature."""

    pass


class MixtureEOSNeededError(Exception):
    """Indicate that a mixture EOS is required for model evaluation."""

    pass


@dataclass_array_container
@dataclass(frozen=True, eq=False)
class GasDependentVars:
    """State-dependent quantities for :class:`GasEOS`.

    Prefer individual methods for model use, use this
    structure for visualization or probing.

    .. attribute:: temperature
    .. attribute:: pressure
    .. attribute:: speed_of_sound
    .. attribute:: smoothness_mu
    .. attribute:: smoothness_kappa
    .. attribute:: smoothness_beta
    .. attribute:: smoothness_d
    """

    temperature: DOFArray
    pressure: DOFArray
    speed_of_sound: DOFArray
    smoothness_mu: DOFArray
    smoothness_kappa: DOFArray
    smoothness_d: DOFArray
    smoothness_beta: DOFArray


@dataclass_array_container
@dataclass(frozen=True, eq=False)
class MixtureDependentVars(GasDependentVars):
    """Mixture state-dependent quantities for :class:`MixtureEOS`.

    ..attribute:: species_enthalpies
    """

    species_enthalpies: DOFArray


class GasEOS(metaclass=ABCMeta):
    r"""Abstract interface to equation of state class.

    Equation of state (EOS) classes are responsible for
    computing relations between fluid or gas state variables.

    Each interface call takes an :class:`~mirgecom.fluid.ConservedVars` object
    array representing the simulation state quantities. Each EOS class
    implementation should document its own state data requirements.

    .. automethod:: pressure
    .. automethod:: temperature
    .. automethod:: sound_speed
    .. automethod:: internal_energy
    .. automethod:: gas_const
    .. automethod:: dependent_vars
    .. automethod:: total_energy
    .. automethod:: kinetic_energy
    .. automethod:: gamma
    .. automethod:: get_internal_energy
    .. automethod:: get_density
    """

    @abstractmethod
    def pressure(self, cv: ConservedVars, temperature: DOFArray):
        """Get the gas pressure."""

    @abstractmethod
    def temperature(self, cv: ConservedVars,
                    temperature_seed: Optional[DOFArray] = None) -> DOFArray:
        """Get the gas temperature."""

    @abstractmethod
    def sound_speed(self, cv: ConservedVars, temperature: DOFArray):
        """Get the gas sound speed."""

    @abstractmethod
    def gas_const(self, cv: Optional[ConservedVars] = None,
                  temperature: Optional[DOFArray] = None,
                  species_mass_fractions: Optional[np.ndarray] = None) -> DOFArray:
        r"""Get the specific gas constant ($R_s$)."""

    @abstractmethod
    def heat_capacity_cp(self, cv: ConservedVars, temperature: DOFArray):
        r"""Get the specific heat capacity at constant pressure ($C_p$)."""

    @abstractmethod
    def heat_capacity_cv(self, cv: ConservedVars, temperature: DOFArray):
        r"""Get the specific heat capacity at constant volume ($C_v$)."""

    @abstractmethod
    def internal_energy(self, cv: ConservedVars):
        """Get the thermal energy of the gas."""

    @abstractmethod
    def total_energy(self, cv: ConservedVars, pressure: DOFArray,
                     temperature: DOFArray):
        """Get the total (thermal + kinetic) energy for the gas."""

    @abstractmethod
    def kinetic_energy(self, cv: ConservedVars):
        """Get the kinetic energy for the gas."""

    @abstractmethod
    def gamma(self, cv: Optional[ConservedVars] = None,
              temperature: Optional[DOFArray] = None):
        """Get the ratio of gas specific heats Cp/Cv."""

    @abstractmethod
    def get_density(self, pressure, temperature,
            species_mass_fractions: Optional[DOFArray] = None):
        """Get the density from pressure, and temperature."""

    @abstractmethod
    def get_internal_energy(self, temperature: DOFArray,
            species_mass_fractions: Optional[np.ndarray] = None) -> DOFArray:
        """Get the fluid internal energy from temperature."""

    def dependent_vars(
            self, cv: ConservedVars,
            temperature_seed: Optional[DOFArray] = None,
            smoothness_mu: Optional[DOFArray] = None,
            smoothness_kappa: Optional[DOFArray] = None,
            smoothness_d: Optional[DOFArray] = None,
            smoothness_beta: Optional[DOFArray] = None) -> GasDependentVars:
        """Get an agglomerated array of the dependent variables.

        Certain implementations of :class:`GasEOS` (e.g. :class:`MixtureEOS`)
        may raise :exc:`TemperatureSeedMissingError` if *temperature_seed* is not
        given.
        """
        temperature = self.temperature(cv, temperature_seed)
        # MJA, it doesn't appear that we can have a None field embedded inside DV,
        # make a dummy smoothness in this case
        zeros = cv.array_context.np.zeros_like(cv.mass)
        if smoothness_mu is None:
            smoothness_mu = zeros
        if smoothness_kappa is None:
            smoothness_kappa = zeros
        if smoothness_d is None:
            smoothness_d = zeros
        if smoothness_beta is None:
            smoothness_beta = zeros

        return GasDependentVars(
            temperature=temperature,
            pressure=self.pressure(cv, temperature),
            speed_of_sound=self.sound_speed(cv, temperature),
            smoothness_mu=smoothness_mu,
            smoothness_kappa=smoothness_kappa,
            smoothness_d=smoothness_d,
            smoothness_beta=smoothness_beta
        )


class MixtureEOS(GasEOS):
    r"""Abstract interface to gas mixture equation of state class.

    This EOS base class extends the GasEOS base class to include the
    necessary interface for dealing with gas mixtures.

    .. automethod:: get_density
    .. automethod:: get_species_molecular_weights
    .. automethod:: get_production_rates
    .. automethod:: species_enthalpies
    .. automethod:: get_species_source_terms
    .. automethod:: get_temperature_seed
    """

    @abstractmethod
    def get_temperature_seed(
            self, ary: Optional[DOFArray] = None,
            temperature_seed: Optional[Union[float, DOFArray]] = None) -> DOFArray:
        r"""Get a constant and uniform guess for the gas temperature.

        This function returns an appropriately sized `DOFArray` for the
        temperature field that will be used as a starting point for the
        solve to find the actual temperature field of the gas.
        """

    @abstractmethod
    def get_density(self, pressure: DOFArray,  # type: ignore[override]
            temperature: DOFArray, species_mass_fractions: np.ndarray):
        """Get the density from pressure, temperature, and species fractions (Y)."""

    @abstractmethod
    def get_species_molecular_weights(self):
        """Get the species molecular weights."""

    @abstractmethod
    def species_enthalpies(self, cv: ConservedVars, temperature: DOFArray):
        """Get the species specific enthalpies."""

    @abstractmethod
    def get_production_rates(self, cv: ConservedVars, temperature: DOFArray):
        """Get the production rate for each species."""

    @abstractmethod
    def get_species_source_terms(self, cv: ConservedVars, temperature: DOFArray):
        r"""Get the species mass source terms to be used on the RHS for chemistry."""

    def dependent_vars(
            self, cv: ConservedVars,
            temperature_seed: Optional[DOFArray] = None,
            smoothness_mu: Optional[DOFArray] = None,
            smoothness_kappa: Optional[DOFArray] = None,
            smoothness_d: Optional[DOFArray] = None,
            smoothness_beta: Optional[DOFArray] = None) -> MixtureDependentVars:
        """Get an agglomerated array of the dependent variables.

        Certain implementations of :class:`GasEOS` (e.g. :class:`MixtureEOS`)
        may raise :exc:`TemperatureSeedMissingError` if *temperature_seed* is not
        given.
        """
        temperature = self.temperature(cv, temperature_seed)
        # MJA, it doesn't appear that we can have a None field embedded inside DV,
        # make a dummy smoothness in this case
        zeros = cv.array_context.np.zeros_like(cv.mass)
        if smoothness_mu is None:
            smoothness_mu = zeros
        if smoothness_kappa is None:
            smoothness_kappa = zeros
        if smoothness_d is None:
            smoothness_d = zeros
        if smoothness_beta is None:
            smoothness_beta = zeros

        return MixtureDependentVars(
            temperature=temperature,
            pressure=self.pressure(cv, temperature),
            speed_of_sound=self.sound_speed(cv, temperature),
            species_enthalpies=self.species_enthalpies(cv, temperature),
            smoothness_mu=smoothness_mu,
            smoothness_kappa=smoothness_kappa,
            smoothness_d=smoothness_d,
            smoothness_beta=smoothness_beta
        )


class IdealSingleGas(GasEOS):
    r"""Ideal gas law single-component gas ($p = \rho{R}{T}$).

    The specific gas constant, R, defaults to the air-like 287.1 J/(kg*K),
    but can be set according to simulation units and materials.

    Each interface call expects that the :class:`~mirgecom.fluid.ConservedVars`
    object representing the simulation conserved quantities contains at least
    the canonical conserved quantities mass ($\rho$), energy ($\rho{E}$), and
    momentum ($\rho\vec{V}$).

    .. automethod:: __init__
    .. automethod:: gamma
    .. automethod:: heat_capacity_cp
    .. automethod:: heat_capacity_cv
    .. automethod:: gas_const
    .. automethod:: get_density
    .. automethod:: kinetic_energy
    .. automethod:: internal_energy
    .. automethod:: pressure
    .. automethod:: sound_speed
    .. automethod:: temperature
    .. automethod:: total_energy
    .. automethod:: get_internal_energy
    .. automethod:: dependent_vars
    """

    def __init__(self, gamma=1.4, gas_const=287.1):
        """Initialize Ideal Gas EOS parameters."""
        self._gamma = gamma
        self._gas_const = gas_const

    def gamma(self, cv: Optional[ConservedVars] = None,
            temperature: Optional[DOFArray] = None) -> DOFArray:
        """Get specific heat ratio Cp/Cv.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Unused for this EOS
        temperature: :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS
        """
        return self._gamma

    def heat_capacity_cp(self, cv: Optional[ConservedVars] = None,
            temperature: Optional[DOFArray] = None) -> DOFArray:
        r"""Get specific heat capacity at constant pressure ($C_p$).

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Unused for this EOS
        temperature: :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS
        """
        return self._gas_const * self._gamma / (self._gamma - 1)

    def heat_capacity_cv(self, cv: Optional[ConservedVars] = None,
            temperature: Optional[DOFArray] = None) -> DOFArray:
        r"""Get specific heat capacity at constant volume ($C_v$).

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Unused for this EOS
        temperature: :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS
        """
        return self._gas_const / (self._gamma - 1)

    def gas_const(self, cv: Optional[ConservedVars] = None,
                  temperature: Optional[DOFArray] = None,
                  species_mass_fractions: Optional[np.ndarray] = None) -> DOFArray:
        """Get specific gas constant R.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Unused for this EOS
        temperature: :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS
        species_mass_fractions: numpy.ndarray
            Unused for this EOS
        """
        return self._gas_const

    def get_density(self, pressure: DOFArray, temperature: DOFArray,
            species_mass_fractions: Optional[DOFArray] = None) -> DOFArray:
        r"""Get gas density from pressure and temperature.

        The gas density is calculated as:

        .. math::

            \rho = \frac{p}{R_s T}

        Parameters
        ----------
        species_mass_fractions: numpy.ndarray
            Unused for this EOS
        """
        return pressure / (self._gas_const * temperature)

    def kinetic_energy(self, cv: ConservedVars) -> DOFArray:
        r"""Get kinetic energy of gas.

        The kinetic energy is calculated as:

        .. math::

            k = \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mom = cv.momentum
        return (0.5 * np.dot(mom, mom) / cv.mass)

    def internal_energy(self, cv: ConservedVars) -> DOFArray:
        r"""Get internal thermal energy of gas.

        The internal energy (e) is calculated as:

        .. math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        return (cv.energy - self.kinetic_energy(cv))

    def pressure(self, cv: ConservedVars,
            temperature: Optional[DOFArray] = None) -> DOFArray:
        r"""Get thermodynamic pressure of the gas.

        Gas pressure (p) is calculated from the internal energy (e) as:

        .. math::

            p = (\gamma - 1)e

        Parameters
        ----------
        temperature: :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS
        """
        return self.internal_energy(cv) * (self._gamma - 1.0)

    def sound_speed(self, cv: ConservedVars,
            temperature: Optional[DOFArray] = None) -> DOFArray:
        r"""Get the speed of sound in the gas.

        The speed of sound (c) is calculated as:

        .. math::

            c = \sqrt{\frac{\gamma{p}}{\rho}}

        Parameters
        ----------
        temperature: :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS
        """
        actx = cv.array_context
        return actx.np.sqrt(self._gamma / cv.mass * self.pressure(cv))

    def temperature(self, cv: ConservedVars,
            temperature_seed: Optional[DOFArray] = None) -> DOFArray:
        r"""Get the thermodynamic temperature of the gas.

        The thermodynamic temperature (T) is calculated from the internal
        energy (e) and specific gas constant (R) as:

        .. math::

            T = \frac{(\gamma - 1)e}{R\rho}

        Parameters
        ----------
        temperature_seed: float or :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS.
        """
        return (
            (((self._gamma - 1.0) / self._gas_const)
             * self.internal_energy(cv) / cv.mass)
        )

    def total_energy(self, cv: ConservedVars, pressure: DOFArray,
            temperature: Optional[DOFArray] = None) -> DOFArray:
        r"""
        Get gas total energy from mass, pressure, and momentum.

        The total energy density (rhoE) is calculated from
        the mass density (rho) , pressure (p) , and momentum (rhoV) as:

        .. math::

            \rho{E} = \frac{p}{(\gamma - 1)} +
            \frac{1}{2}\rho(\vec{v} \cdot \vec{v})

        .. note::

            The total_energy function computes cv.energy from pressure,
            mass, and momentum in this case. In general in the EOS we need
            DV = EOS(CV), and inversions CV = EOS(DV). This is one of those
            inversion interfaces.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            The conserved variables
        pressure: :class:`~meshmode.dof_array.DOFArray`
            The fluid pressure
        temperature: :class:`~meshmode.dof_array.DOFArray`
            Unused for this EOS

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The total energy of the fluid (i.e. internal + kinetic)
        """
        return (pressure / (self._gamma - 1.0)
                + self.kinetic_energy(cv))

    def get_internal_energy(self, temperature: DOFArray,
            species_mass_fractions: Optional[DOFArray] = None) -> DOFArray:
        r"""Get the gas thermal energy from temperature.

        The gas internal energy $e$ is calculated from:

        .. math::

            e = \frac{R_s T}{\left(\gamma - 1\right)}

        Parameters
        ----------
        species_mass_fractions:
            Unused for this EOS
        """
        return self._gas_const * temperature / (self._gamma - 1)


class PyrometheusMixture(MixtureEOS):
    r"""Ideal gas mixture ($p = \rho{R}_\mathtt{mix}{T}$).

    This is the :mod:`pyrometheus`-based EOS. Please refer to the :any:`documentation
    of Pyrometheus <pyrometheus>` for underlying implementation details.

    Each interface call expects that the :class:`mirgecom.fluid.ConservedVars` object
    representing the simulation conserved quantities contains at least the
    canonical conserved quantities mass ($\rho$), energy ($\rho{E}$), and
    momentum ($\rho\vec{V}$), and the vector of species masses, ($\rho{Y}_\alpha$).

    .. important::
        When using this EOS, users should take care to match solution initialization
        with the appropriate units that are used in the user-provided `Cantera`
        mechanism input files.

    .. automethod:: __init__
    .. automethod:: pressure
    .. automethod:: temperature
    .. automethod:: sound_speed
    .. automethod:: internal_energy
    .. automethod:: gas_const
    .. automethod:: dependent_vars
    .. automethod:: total_energy
    .. automethod:: kinetic_energy
    .. automethod:: heat_capacity_cv
    .. automethod:: heat_capacity_cp
    .. automethod:: gamma
    .. automethod:: get_internal_energy
    .. automethod:: get_density
    .. automethod:: get_species_molecular_weights
    .. automethod:: get_production_rates
    .. automethod:: species_enthalpies
    .. automethod:: get_species_source_terms
    .. automethod:: get_temperature_seed
    """

    def __init__(self, pyrometheus_mech, temperature_guess=300.0):
        """Initialize Pyrometheus-based EOS with mechanism class.

        Parameters
        ----------
        pyrometheus_mech: :class:`~pyrometheus.thermochem_example.Thermochemistry`
            The :mod:`pyrometheus` mechanism
            :class:`~pyrometheus.thermochem_example.Thermochemistry`
            object that is generated by the user with a call to
            *pyrometheus.get_thermochem_class*. To create the mechanism
            object, users need to provide a mechanism input file. Several example
            mechanisms are provided in `mirgecom/mechanisms/` and can be used through
            the :meth:`mirgecom.mechanisms.get_mechanism_input`.

        tguess: float
            This provides a constant starting temperature for the Newton iterations
            used to find the mixture temperature. It defaults to 300.0 Kelvin. This
            parameter is important for the performance and proper function of the
            code. Users should set a tguess that is close to the average temperature
            of the simulated domain.
        """
        self._pyrometheus_mech = pyrometheus_mech
        self._tguess = temperature_guess

    def get_temperature_seed(self, ary: Optional[DOFArray] = None,
            temperature_seed: Optional[DOFArray] = None) -> DOFArray:
        """Get a *cv*-shaped array with which to seed temperature calcuation.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` used to conjure the required shape
            for the returned temperature guess.
        temperature_seed: float or :class:`~meshmode.dof_array.DOFArray`
            Optional data from which to seed temperature calculation.

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The temperature with which to seed the Newton solver in
            :module:thermochemistry.
        """
        tseed = self._tguess
        if temperature_seed is not None:
            tseed = temperature_seed
        if isinstance(tseed, DOFArray):
            return tseed
        else:
            if ary is None:
                raise ValueError("Requires *ary* for shaping temperature seed.")
        return tseed * (0*ary + 1.0)

    def heat_capacity_cp(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Get mixture-averaged specific heat capacity at constant pressure."""
        y = cv.species_mass_fractions
        return \
            self._pyrometheus_mech.get_mixture_specific_heat_cp_mass(temperature, y)

    def heat_capacity_cv(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Get mixture-averaged specific heat capacity at constant volume."""
        y = cv.species_mass_fractions
        cp = self._pyrometheus_mech.get_mixture_specific_heat_cp_mass(temperature, y)
        rspec = self.gas_const(species_mass_fractions=y)
        return cp - rspec

    def gamma(self, cv: ConservedVars,  # type: ignore[override]
            temperature: DOFArray) -> DOFArray:
        r"""Get mixture-averaged heat capacity ratio, $\frac{C_p}{C_p - R_s}$."""
        y = cv.species_mass_fractions
        cp = self._pyrometheus_mech.get_mixture_specific_heat_cp_mass(temperature, y)
        rspec = self.gas_const(species_mass_fractions=y)
        return cp / (cp - rspec)

    def gas_const(self, cv: Optional[ConservedVars] = None,
                  temperature: Optional[DOFArray] = None,
                  species_mass_fractions: Optional[np.ndarray] = None) -> DOFArray:
        r"""Get specific gas constant $R_s$.

        The mixture specific gas constant is calculated as

        .. math::

            R_s = \frac{R}{\sum{\frac{{Y}_\alpha}{{M}_\alpha}}}

        by the :mod:`pyrometheus` mechanism provided by the user. In this
        equation, ${M}_\alpha$ are the species molar masses and R is the
        universal gas constant.
        """
        if cv is not None:
            from warnings import warn
            warn("Passing CV to eos.gas_const will be deprecated in Q1 2024.",
                 stacklevel=2)
        y = species_mass_fractions if cv is None else cv.species_mass_fractions
        return self._pyrometheus_mech.get_specific_gas_constant(y)

    def kinetic_energy(self, cv: ConservedVars) -> DOFArray:
        r"""Get kinetic (i.e. not internal) energy of gas.

        The kinetic energy is calculated as:

        .. math::

            k = \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mom = cv.momentum
        return (0.5 * np.dot(mom, mom) / cv.mass)

    def internal_energy(self, cv: ConservedVars) -> DOFArray:
        r"""Get internal thermal energy of gas.

        The internal energy ($e$) is calculated as:

        .. math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        return (cv.energy - self.kinetic_energy(cv))

    def get_density(self, pressure: DOFArray,  # type: ignore[override]
            temperature: DOFArray, species_mass_fractions: np.ndarray) -> DOFArray:
        r"""Get the density from pressure, temperature, and species fractions (Y)."""
        return self._pyrometheus_mech.get_density(pressure, temperature,
                                                  species_mass_fractions)

    def get_internal_energy(self, temperature: DOFArray,  # type: ignore[override]
            species_mass_fractions: np.ndarray) -> DOFArray:
        r"""Get the gas thermal energy from temperature, and species fractions (Y).

        The gas internal energy $e$ is calculated from:

        .. math::

            e = R_s T \sum{Y_\alpha e_\alpha}
        """
        return self._pyrometheus_mech.get_mixture_internal_energy_mass(
            temperature, species_mass_fractions)

    def get_enthalpy(self, temperature: DOFArray,
            species_mass_fractions: np.ndarray) -> DOFArray:
        r"""Get the gas enthalpy from temperature, and species fractions (Y).

        The enthalpy of the gas mixture $h$ is calculated from:

        .. math::

            h = \sum{Y_\alpha h_\alpha}
        """
        return self._pyrometheus_mech.get_mixture_enthalpy_mass(
            temperature, species_mass_fractions)

    def get_species_molecular_weights(self):
        """Get the species molecular weights."""
        return self._pyrometheus_mech.wts
        # return self._pyrometheus_mech.molecular_weights

    def species_enthalpies(self, cv: ConservedVars,
            temperature: DOFArray) -> DOFArray:
        """Get the species specific enthalpies."""
        spec_r = self._pyrometheus_mech.gas_constant/self._pyrometheus_mech.wts
        # spec_r = (self._pyrometheus_mech.gas_constant
        #           / self._pyrometheus_mech.molecular_weights)
        return (spec_r * temperature
                * self._pyrometheus_mech.get_species_enthalpies_rt(temperature))

    def get_production_rates(self, cv: ConservedVars,
            temperature: DOFArray) -> np.ndarray:
        r"""Get the chemical production rates for each species."""
        y = cv.species_mass_fractions
        return self._pyrometheus_mech.get_net_production_rates(
            cv.mass, temperature, y)

    def pressure(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Get thermodynamic pressure of the gas.

        Gas pressure ($p$) is calculated using ideal gas law:

        .. math::

            p = \rho R_{mix} T
        """
        y = cv.species_mass_fractions
        return self._pyrometheus_mech.get_pressure(cv.mass, temperature, y)

    def sound_speed(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Get the speed of sound in the gas.

        The speed of sound ($c$) is calculated as:

        .. math::

            c = \sqrt{\frac{\gamma_{\mathtt{mix}}{p}}{\rho}}
        """
        actx = cv.array_context
        return actx.np.sqrt((self.gamma(cv, temperature)
                             * self.pressure(cv, temperature))
                            / cv.mass)

    def temperature(self, cv: ConservedVars,
            temperature_seed: Optional[DOFArray] = None) -> DOFArray:
        r"""Get the thermodynamic temperature of the gas.

        The thermodynamic temperature ($T$) is calculated iteratively with
        Newton-Raphson method from the mixture internal energy ($e$) as:

        .. math::

            e(T) = \sum_i e_i(T) Y_i

        Parameters
        ----------
        temperature_seed: float or :class:`~meshmode.dof_array.DOFArray`
            Data from which to seed temperature calculation.
        """
        # For mixtures, the temperature calculation *must* be seeded. This
        # check catches any actual temperature calculation that did not
        # provide a seed.
        if temperature_seed is None:
            raise TemperatureSeedMissingError("MixtureEOS.get_temperature"
                                              "requires a *temperature_seed*.")
        tseed = self.get_temperature_seed(cv.mass, temperature_seed)

        y = cv.species_mass_fractions
        e = self.internal_energy(cv) / cv.mass
        return self._pyrometheus_mech.get_temperature(e, tseed, y)

    def temperature_from_enthalpy(self, enthalpy: DOFArray,
            temperature_seed: Optional[DOFArray] = None,
            species_mass_fractions: Optional[np.ndarray] = None) -> DOFArray:
        r"""Get the thermodynamic temperature of the gas.

        The thermodynamic temperature ($T$) is calculated iteratively with
        Newton-Raphson method from the mixture specific enthalpy ($h$) as:

        .. math::

            h(T) = \sum_i h_i(T) Y_i

        Parameters
        ----------
        temperature_seed: float or :class:`~meshmode.dof_array.DOFArray`
            Data from which to seed temperature calculation.
        """
        # For mixtures, the temperature calculation *must* be seeded. This
        # check catches any actual temperature calculation that did not
        # provide a seed.
        if temperature_seed is None:
            raise TemperatureSeedMissingError("MixtureEOS.get_temperature"
                                              "requires a *temperature_seed*.")
        tseed = self.get_temperature_seed(enthalpy, temperature_seed)

        return self._pyrometheus_mech.get_temperature(
            enthalpy, tseed, species_mass_fractions, use_energy=False)

    def total_energy(self, cv: ConservedVars, pressure: DOFArray,
            temperature: DOFArray) -> DOFArray:
        r"""
        Get gas total energy from mass, pressure, and momentum.

        The total energy density ($\rho E$) is calculated from
        the mass density ($\rho$) , pressure ($p$) , and
        momentum ($\rho\vec{V}$) as:

        .. math::

            \rho E = \frac{p}{(\gamma_{\mathtt{mix}} - 1)} +
            \frac{1}{2}\rho(\vec{v} \cdot \vec{v})

        .. note::

            The total_energy function computes cv.energy from pressure,
            mass, and momentum in this case. In general in the EOS we need
            DV = EOS(CV), and inversions CV = EOS(DV). This is one of those
            inversion interfaces.
        """
        y = cv.species_mass_fractions
        return (cv.mass * self.get_internal_energy(temperature, y)
                + self.kinetic_energy(cv))

    def get_species_source_terms(self, cv: ConservedVars, temperature: DOFArray):
        r"""Get the species mass source terms to be used on the RHS for chemistry.

        Returns
        -------
        :class:`~mirgecom.fluid.ConservedVars`
            Chemistry source terms
        """
        omega = self.get_production_rates(cv, temperature)
        w = self.get_species_molecular_weights()
        dim = cv.dim
        species_sources = w * omega
        rho_source = 0 * cv.mass
        mom_source = 0 * cv.momentum
        energy_source = 0 * cv.energy

        return make_conserved(dim, rho_source, energy_source, mom_source,
                              species_sources)

    def get_mole_fractions(self, species_mass_fractions):
        """Get mole fractions from mass fractions."""
        mix_mol_weight = \
            self._pyrometheus_mech.get_mix_molecular_weight(species_mass_fractions)
        return self._pyrometheus_mech.get_mole_fractions(mix_mol_weight,
                                                         species_mass_fractions)

    def get_mass_fractions(self, species_mole_fractions):
        """Get mass fractions from mole fractions."""
        mol_weights = self.get_species_molecular_weights()
        mmw = sum(species_mole_fractions*mol_weights)
        return species_mole_fractions*mol_weights/mmw
