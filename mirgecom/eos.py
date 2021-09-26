"""
:mod:`mirgecom.eos` provides implementations of gas equations of state.

Equations of State
^^^^^^^^^^^^^^^^^^
This module is designed provide Equation of State objects used to compute and
manage the relationships between and among state and thermodynamic variables.

.. autoclass:: EOSDependentVars
.. autoclass:: GasEOS
.. autoclass:: MixtureEOS
.. autoclass:: IdealSingleGas
.. autoclass:: PyrometheusMixture
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

from dataclasses import dataclass
import numpy as np
from pytools import memoize_in
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import ConservedVars, make_conserved
from abc import ABCMeta, abstractmethod


@dataclass
class EOSDependentVars:
    """State-dependent quantities for :class:`GasEOS`.

    Prefer individual methods for model use, use this
    structure for visualization or probing.

    .. attribute:: temperature
    .. attribute:: pressure
    """

    temperature: np.ndarray
    pressure: np.ndarray


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
    .. automethod:: transport_model
    .. automethod:: get_internal_energy
    """

    @abstractmethod
    def pressure(self, cv: ConservedVars):
        """Get the gas pressure."""

    @abstractmethod
    def temperature(self, cv: ConservedVars):
        """Get the gas temperature."""

    @abstractmethod
    def sound_speed(self, cv: ConservedVars):
        """Get the gas sound speed."""

    @abstractmethod
    def gas_const(self, cv: ConservedVars):
        r"""Get the specific gas constant ($R_s$)."""

    @abstractmethod
    def heat_capacity_cp(self, cv: ConservedVars):
        r"""Get the specific heat capacity at constant pressure ($C_p$)."""

    @abstractmethod
    def heat_capacity_cv(self, cv: ConservedVars):
        r"""Get the specific heat capacity at constant volume ($C_v$)."""

    @abstractmethod
    def internal_energy(self, cv: ConservedVars):
        """Get the thermal energy of the gas."""

    @abstractmethod
    def total_energy(self, cv: ConservedVars, pressure: np.ndarray):
        """Get the total (thermal + kinetic) energy for the gas."""

    @abstractmethod
    def kinetic_energy(self, cv: ConservedVars):
        """Get the kinetic energy for the gas."""

    @abstractmethod
    def gamma(self, cv: ConservedVars):
        """Get the ratio of gas specific heats Cp/Cv."""

    @abstractmethod
    def transport_model(self):
        """Get the transport model if it exists."""

    @abstractmethod
    def get_internal_energy(self, temperature, *, mass, species_mass_fractions):
        """Get the fluid internal energy from temperature and mass."""

    def dependent_vars(self, cv: ConservedVars) -> EOSDependentVars:
        """Get an agglomerated array of the dependent variables."""
        return EOSDependentVars(
            pressure=self.pressure(cv),
            temperature=self.temperature(cv),
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
    """

    @abstractmethod
    def get_density(self, pressure, temperature, species_mass_fractions):
        """Get the density from pressure, temperature, and species fractions (Y)."""

    @abstractmethod
    def get_species_molecular_weights(self):
        """Get the species molecular weights."""

    @abstractmethod
    def species_enthalpies(self, cv: ConservedVars):
        """Get the species specific enthalpies."""

    @abstractmethod
    def get_production_rates(self, cv: ConservedVars):
        """Get the production rate for each species."""

    @abstractmethod
    def get_species_source_terms(self, cv: ConservedVars):
        r"""Get the species mass source terms to be used on the RHS for chemistry."""


class IdealSingleGas(GasEOS):
    r"""Ideal gas law single-component gas ($p = \rho{R}{T}$).

    The specific gas constant, R, defaults to the air-like 287.1 J/(kg*K),
    but can be set according to simulation units and materials.

    Each interface call expects that the :class:`~mirgecom.fluid.ConservedVars`
    object representing the simulation conserved quantities contains at least
    the canonical conserved quantities mass ($\rho$), energy ($\rho{E}$), and
    momentum ($\rho\vec{V}$).

    .. automethod:: __init__
    .. automethod:: pressure
    .. automethod:: temperature
    .. automethod:: sound_speed
    .. automethod:: internal_energy
    .. automethod:: gas_const
    .. automethod:: dependent_vars
    .. automethod:: total_energy
    .. automethod:: kinetic_energy
    .. automethod:: gamma
    .. automethod:: transport_model
    .. automethod:: get_internal_energy
    """

    def __init__(self, gamma=1.4, gas_const=287.1, transport_model=None):
        """Initialize Ideal Gas EOS parameters."""
        self._gamma = gamma
        self._gas_const = gas_const
        self._transport_model = transport_model

    def transport_model(self):
        """Get the transport model object for this EOS."""
        return self._transport_model

    def gamma(self, cv: ConservedVars = None):
        """Get specific heat ratio Cp/Cv."""
        return self._gamma

    def heat_capacity_cp(self, cv: ConservedVars = None):
        r"""Get specific heat capacity at constant pressure.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector of
            species masses, ($\rho{Y}_\alpha$).
        """
        return self._gas_const * self._gamma / (self._gamma - 1)

    def heat_capacity_cv(self, cv: ConservedVars = None):
        r"""Get specific heat capacity at constant volume.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector of
            species masses, ($\rho{Y}_\alpha$).
        """
        return self._gas_const / (self._gamma - 1)

    def gas_const(self, cv: ConservedVars = None):
        """Get specific gas constant R."""
        return self._gas_const

    def kinetic_energy(self, cv: ConservedVars):
        r"""Get kinetic (i.e. not internal) energy of gas.

        The kinetic energy is calculated as:

        .. math::

            k = \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The kinetic energy of the fluid flow
        """
        mom = cv.momentum
        return (0.5 * np.dot(mom, mom) / cv.mass)

    def internal_energy(self, cv: ConservedVars):
        r"""Get internal thermal energy of gas.

        The internal energy (e) is calculated as:

        .. math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The internal energy of the fluid material
        """
        return (cv.energy - self.kinetic_energy(cv))

    def pressure(self, cv: ConservedVars):
        r"""Get thermodynamic pressure of the gas.

        Gas pressure (p) is calculated from the internal energy (e) as:

        .. math::

            p = (\gamma - 1)e

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The fluid pressure
        """
        return self.internal_energy(cv) * (self._gamma - 1.0)

    def sound_speed(self, cv: ConservedVars):
        r"""Get the speed of sound in the gas.

        The speed of sound (c) is calculated as:

        .. math::

            c = \sqrt{\frac{\gamma{p}}{\rho}}

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The speed of sound in the fluid
        """
        actx = cv.array_context
        return actx.np.sqrt(self._gamma / cv.mass * self.pressure(cv))

    def temperature(self, cv: ConservedVars):
        r"""Get the thermodynamic temperature of the gas.

        The thermodynamic temperature (T) is calculated from
        the internal energy (e) and specific gas constant (R)
        as:

        .. math::

            T = \frac{(\gamma - 1)e}{R\rho}

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The fluid temperature
        """
        return (
            (((self._gamma - 1.0) / self._gas_const)
             * self.internal_energy(cv) / cv.mass)
        )

    def total_energy(self, cv, pressure):
        r"""
        Get gas total energy from mass, pressure, and momentum.

        The total energy density (rhoE) is calculated from
        the mass density (rho) , pressure (p) , and
        momentum (rhoV) as:

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
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$).

        pressure: :class:`~meshmode.dof_array.DOFArray`
            The fluid pressure

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The total energy of the fluid (i.e. internal + kinetic)
        """
        return (pressure / (self._gamma - 1.0)
                + self.kinetic_energy(cv))

    def get_internal_energy(self, temperature, mass, species_mass_fractions=None):
        r"""Get the gas thermal energy from temperature, and fluid density.

        The gas internal energy $e$ is calculated from:

        .. math::

            e = R_s T \frac{\rho}{\left(\gamma - 1\right)}

        Parameters
        ----------
        temperature: :class:`~meshmode.dof_array.DOFArray`
            The fluid temperature
        mass: float or :class:`~meshmode.dof_array.DOFArray`
            The fluid mass density
        species_mass_fractions:
            Unused
        """
        return self._gas_const * mass * temperature / (self._gamma - 1)


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
    .. automethod:: gamma
    .. automethod:: transport_model
    .. automethod:: get_internal_energy
    .. automethod:: get_density
    .. automethod:: get_species_molecular_weights
    .. automethod:: get_production_rates
    .. automethod:: species_enthalpies
    .. automethod:: get_species_source_terms
    """

    def __init__(self, pyrometheus_mech, temperature_guess=300.0,
                 transport_model=None):
        """Initialize Pyrometheus-based EOS with mechanism class.

        Parameters
        ----------
        pyrometheus_mech: :class:`pyrometheus.Thermochemistry`
            The :mod:`pyrometheus`  mechanism :class:`~pyrometheus.Thermochemistry`
            object that is generated by the user with a call to
            *pyrometheus.get_thermochem_class*. To create the mechanism
            object, users need to provide a mechanism input file. Several built-in
            mechanisms are provided in `mirgecom/mechanisms/` and can be used through
            the :meth:`mirgecom.mechanisms.get_mechanism_cti`.

        tguess: float
            This provides a constant starting temperature for the Newton iterations
            used to find the mixture temperature. It defaults to 300.0 Kelvin. This
            parameter is important for the performance and proper function of the
            code. Users should set a tguess that is close to the average temperature
            of the simulated domain.  Ideally, we would use the last computed
            temperature for the guess, but doing so requires restart infrastructure
            that is TBD.
        """
        self._pyrometheus_mech = pyrometheus_mech
        self._tguess = temperature_guess
        self._transport_model = transport_model

    def transport_model(self):
        """Get the transport model object for this EOS."""
        return self._transport_model

    def heat_capacity_cp(self, cv: ConservedVars):
        r"""Get mixture-averaged specific heat capacity at constant pressure.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector of
            species masses, ($\rho{Y}_\alpha$).
        """
        temp = self.temperature(cv)
        y = cv.species_mass_fractions
        return self._pyrometheus_mech.get_mixture_specific_heat_cp_mass(temp, y)

    def heat_capacity_cv(self, cv: ConservedVars):
        r"""Get mixture-averaged specific heat capacity at constant volume.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector of
            species masses, ($\rho{Y}_\alpha$).
        """
        temp = self.temperature(cv)
        y = cv.species_mass_fractions
        return (
            self._pyrometheus_mech.get_mixture_specific_heat_cp_mass(temp, y)
            / self.gamma(cv)
        )

    def gamma(self, cv: ConservedVars):
        r"""Get mixture-averaged specific heat ratio for mixture $\frac{C_p}{C_p - R_s}$.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector of
            species masses, ($\rho{Y}_\alpha$).
        """
        temperature = self.temperature(cv)
        y = cv.species_mass_fractions
        cp = self._pyrometheus_mech.get_mixture_specific_heat_cp_mass(temperature, y)
        rspec = self.gas_const(cv)
        return cp / (cp - rspec)

    def gas_const(self, cv: ConservedVars):
        r"""Get specific gas constant $R_s$.

        The mixture specific gas constant is calculated
        as $R_s = \frac{R}{\sum{\frac{{Y}_\alpha}{{M}_\alpha}}}$ by the
        :mod:`pyrometheus` mechanism provided by the user. ${M}_\alpha$ are the
        species molar masses.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).
        """
        y = cv.species_mass_fractions
        return self._pyrometheus_mech.get_specific_gas_constant(y)

    def kinetic_energy(self, cv: ConservedVars):
        r"""Get kinetic (i.e. not internal) energy of gas.

        The kinetic energy is calculated as:

        .. math::

            k = \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).
        """
        mom = cv.momentum
        return (0.5 * np.dot(mom, mom) / cv.mass)

    def internal_energy(self, cv: ConservedVars):
        r"""Get internal thermal energy of gas.

        The internal energy ($e$) is calculated as:

        .. math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            Internal energy of the fluid
        """
        return (cv.energy - self.kinetic_energy(cv))

    def get_density(self, pressure, temperature, species_mass_fractions):
        r"""Get the density from pressure, temperature, and species fractions (Y).

        The gas density $\rho$ is calculated from pressure, temperature and $R$ as:

        .. math::

            \rho = \frac{p}{R_s T}

        Parameters
        ----------
        pressure: :class:`~meshmode.dof_array.DOFArray`
            The fluid pressure
        temperature: :class:`~meshmode.dof_array.DOFArray`
            The fluid temperature
        species_mass_fractions: numpy.ndarray
            Object array of species mass fractions

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The total fluid mass density
        """
        return self._pyrometheus_mech.get_density(pressure, temperature,
                                                  species_mass_fractions)

    def get_internal_energy(self, temperature, species_mass_fractions,
                            mass=None):
        r"""Get the gas thermal energy from temperature, and species fractions (Y).

        The gas internal energy $e$ is calculated from:

        .. math::

            e = R_s T \sum{Y_\alpha h_\alpha}

        Parameters
        ----------
        temperature: :class:`~meshmode.dof_array.DOFArray`
            The fluid temperature
        species_mass_fractions: numpy.ndarray
            Object array of species mass fractions
        mass:
            Unused
        """
        return self._pyrometheus_mech.get_mixture_internal_energy_mass(
            temperature, species_mass_fractions)

    def get_species_molecular_weights(self):
        """Get the species molecular weights."""
        return self._pyrometheus_mech.wts

    def species_enthalpies(self, cv: ConservedVars):
        """Get the species specific enthalpies."""
        return self._pyrometheus_mech.get_species_enthalpies_rt(self.temperature(cv))

    def get_production_rates(self, cv: ConservedVars):
        r"""Get the production rate for each species.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).

        Returns
        -------
        numpy.ndarray
            The chemical production rates for each species
        """
        temperature = self.temperature(cv)
        y = cv.species_mass_fractions
        return self._pyrometheus_mech.get_net_production_rates(
            cv.mass, temperature, y)

    def pressure(self, cv: ConservedVars):
        r"""Get thermodynamic pressure of the gas.

        Gas pressure ($p$) is calculated from the internal energy ($e$) as:

        .. math::

            p = (\gamma_{\mathtt{mix}} - 1)e

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The pressure of the fluid.
        """
        @memoize_in(cv, (PyrometheusMixture.pressure,
                         type(self._pyrometheus_mech)))
        def get_pressure():
            temperature = self.temperature(cv)
            y = cv.species_mass_fractions
            return self._pyrometheus_mech.get_pressure(cv.mass, temperature, y)
        return get_pressure()

    def sound_speed(self, cv: ConservedVars):
        r"""Get the speed of sound in the gas.

        The speed of sound ($c$) is calculated as:

        .. math::

            c = \sqrt{\frac{\gamma_{\mathtt{mix}}{p}}{\rho}}

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The speed of sound in the fluid.
        """
        @memoize_in(cv, (PyrometheusMixture.sound_speed,
                         type(self._pyrometheus_mech)))
        def get_sos():
            actx = cv.array_context
            return actx.np.sqrt((self.gamma(cv) * self.pressure(cv)) / cv.mass)
        return get_sos()

    def temperature(self, cv: ConservedVars):
        r"""Get the thermodynamic temperature of the gas.

        The thermodynamic temperature ($T$) is calculated from
        the internal energy ($e$) and specific gas constant ($R_s$)
        as:

        .. math::

            T = \frac{(\gamma_{\mathtt{mix}} - 1)e}{R_s \rho}

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The temperature of the fluid.
        """
        @memoize_in(cv, (PyrometheusMixture.temperature,
                         type(self._pyrometheus_mech)))
        def get_temp():
            y = cv.species_mass_fractions
            e = self.internal_energy(cv) / cv.mass
            return self._pyrometheus_mech.get_temperature(e, self._tguess,
                                                          y, True)
        return get_temp()

    def total_energy(self, cv, pressure):
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

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).
        pressure: :class:`~meshmode.dof_array.DOFArray`
            The fluid pressure

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The total energy fo the fluid (i.e. internal + kinetic)
        """
        return (pressure / (self.gamma(cv) - 1.0)
                + self.kinetic_energy(cv))

    def get_species_source_terms(self, cv: ConservedVars):
        r"""Get the species mass source terms to be used on the RHS for chemistry.

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            :class:`~mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).

        Returns
        -------
        :class:`~mirgecom.fluid.ConservedVars`
            Chemistry source terms
        """
        omega = self.get_production_rates(cv)
        w = self.get_species_molecular_weights()
        dim = cv.dim
        species_sources = w * omega
        rho_source = 0 * cv.mass
        mom_source = 0 * cv.momentum
        energy_source = 0 * cv.energy

        return make_conserved(dim, rho_source, energy_source, mom_source,
                              species_sources)
