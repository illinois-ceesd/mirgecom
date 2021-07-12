"""
:mod:`mirgecom.eos` provides implementations of gas equations of state.

Equations of State
^^^^^^^^^^^^^^^^^^
This module is designed provide Equation of State objects used to compute and
manage the relationships between and among state and thermodynamic variables.

.. autoclass:: EOSDependentVars
.. autoclass:: GasEOS
.. autoclass:: IdealSingleGas
.. autoclass:: PyrometheusMixture
"""

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

from dataclasses import dataclass
import numpy as np
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import ConservedVars


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


class GasEOS:
    r"""Abstract interface to equation of state class.

    Equation of state (EOS) classes are responsible for
    computing relations between fluid or gas state variables.

    Each interface call takes an :class:`mirgecom.fluid.ConservedVars` object
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
    """

    def pressure(self, cv: ConservedVars):
        """Get the gas pressure."""
        raise NotImplementedError()

    def temperature(self, cv: ConservedVars):
        """Get the gas temperature."""
        raise NotImplementedError()

    def sound_speed(self, cv: ConservedVars):
        """Get the gas sound speed."""
        raise NotImplementedError()

    def gas_const(self, cv: ConservedVars = None):
        r"""Get the specific gas constant ($R_s$)."""
        raise NotImplementedError()

    def internal_energy(self, cv: ConservedVars):
        """Get the thermal energy of the gas."""
        raise NotImplementedError()

    def total_energy(self, cv: ConservedVars, pressure: np.ndarray):
        """Get the total (thermal + kinetic) energy for the gas."""
        raise NotImplementedError()

    def kinetic_energy(self, cv: ConservedVars):
        """Get the kinetic energy for the gas."""
        raise NotImplementedError()

    def gamma(self, cv: ConservedVars = None):
        """Get the ratio of gas specific heats Cp/Cv."""
        raise NotImplementedError()

    def dependent_vars(self, cv: ConservedVars) -> EOSDependentVars:
        """Get an agglomerated array of the depedent variables."""
        return EOSDependentVars(
            pressure=self.pressure(cv),
            temperature=self.temperature(cv),
            )


class IdealSingleGas(GasEOS):
    r"""Ideal gas law single-component gas ($p = \rho{R}{T}$).

    The specific gas constant, R, defaults to the air-like 287.1 J/(kg*K),
    but can be set according to simulation units and materials.

    Each interface call expects that the :class:`mirgecom.fluid.ConservedVars` object
    representing the simulation conserved quantities contains at least the canonical
    conserved quantities mass ($\rho$), energy ($\rho{E}$), and
    momentum ($\rho\vec{V}$).

    .. automethod:: __init__

    Inherits from (and implements) :class:`GasEOS`.
    """

    def __init__(self, gamma=1.4, gas_const=287.1):
        """Initialize Ideal Gas EOS parameters."""
        self._gamma = gamma
        self._gas_const = gas_const

    def gamma(self, cv: ConservedVars = None):
        """Get specific heat ratio Cp/Cv."""
        return self._gamma

    def gas_const(self, cv: ConservedVars = None):
        """Get specific gas constant R."""
        return self._gas_const

    def kinetic_energy(self, cv: ConservedVars):
        r"""Get kinetic (i.e. not internal) energy of gas.

        The kinetic energy is calculated as:
        .. :math::

            k = \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mom = cv.momentum
        return (0.5 * np.dot(mom, mom) / cv.mass)

    def internal_energy(self, cv: ConservedVars):
        r"""Get internal thermal energy of gas.

        The internal energy (e) is calculated as:
        .. :math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        return (cv.energy - self.kinetic_energy(cv))

    def pressure(self, cv: ConservedVars):
        r"""Get thermodynamic pressure of the gas.

        Gas pressure (p) is calculated from the internal energy (e) as:

        .. :math::

            p = (\gamma - 1)e
        """
        return self.internal_energy(cv) * (self._gamma - 1.0)

    def sound_speed(self, cv: ConservedVars):
        r"""Get the speed of sound in the gas.

        The speed of sound (c) is calculated as:

        .. :math::

            c = \sqrt{\frac{\gamma{p}}{\rho}}
        """
        actx = cv.mass.array_context

        p = self.pressure(cv)
        c2 = self._gamma / cv.mass * p
        return actx.np.sqrt(c2)

    def temperature(self, cv: ConservedVars):
        r"""Get the thermodynamic temperature of the gas.

        The thermodynamic temperature (T) is calculated from
        the internal energy (e) and specific gas constant (R)
        as:

        .. :math::

            T = \frac{(\gamma - 1)e}{R\rho}
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

        .. :math::

            \rhoE = \frac{p}{(\gamma - 1)} +
            \frac{1}{2}\rho(\vec{v} \cdot \vec{v})

        .. note::

            The total_energy function computes cv.energy from pressure,
            mass, and momentum in this case. In general in the EOS we need
            DV = EOS(CV), and inversions CV = EOS(DV). This is one of those
            inversion interfaces.
        """
        return (pressure / (self._gamma - 1.0)
                + self.kinetic_energy(cv))


class PyrometheusMixture(GasEOS):
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
    .. automethod:: get_density
    .. automethod:: get_species_molecular_weights
    .. automethod:: get_production_rates
    .. automethod:: get_species_source_terms
    .. automethod:: get_internal_energy
    .. automethod:: species_fractions
    .. automethod:: total_energy
    .. automethod:: gamma
    .. automethod:: gas_const

    Inherits from (and implements) :class:`GasEOS`.
    """

    def __init__(self, pyrometheus_mech, temperature_guess=300.0):
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

    def gamma(self, cv: ConservedVars = None):
        r"""Get mixture-averaged specific heat ratio for mixture $\frac{C_p}{C_p - R_s}$.

        Parameters
        ----------
        cv: :class:`mirgecom.fluid.ConservedVars`
            :class:`mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector of
            species masses, ($\rho{Y}_\alpha$).
        """
        if cv is None:
            raise ValueError("EOS.gamma requires ConservedVars (cv) argument.")
        temperature = self.temperature(cv)
        y = self.species_fractions(cv)
        cp = self._pyrometheus_mech.get_mixture_specific_heat_cp_mass(temperature, y)
        rspec = self.gas_const(cv)
        return cp / (cp - rspec)

    def gas_const(self, cv: ConservedVars = None):
        r"""Get specific gas constant $R_s$.

        The mixture specific gas constant is calculated
        as $R_s = \frac{R}{\sum{\frac{{Y}_\alpha}{{M}_\alpha}}}$ by the
        :mod:`pyrometheus` mechanism provided by the user. ${M}_\alpha$ are the
        species molar masses.

        Parameters
        ----------
        cv: :class:`mirgecom.fluid.ConservedVars`
            :class:`mirgecom.fluid.ConservedVars` containing at least the mass
            ($\rho$), energy ($\rho{E}$), momentum ($\rho\vec{V}$), and the vector
            of species masses, ($\rho{Y}_\alpha$).
        """
        if cv is None:
            raise ValueError("EOS.gas_const requires ConservedVars (cv) argument.")
        y = self.species_fractions(cv)
        return self._pyrometheus_mech.get_specific_gas_constant(y)

    def kinetic_energy(self, cv: ConservedVars):
        r"""Get kinetic (i.e. not internal) energy of gas.

        The kinetic energy is calculated as:

        .. math::

            k = \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mom = cv.momentum
        return (0.5 * np.dot(mom, mom) / cv.mass)

    def internal_energy(self, cv: ConservedVars):
        r"""Get internal thermal energy of gas.

        The internal energy ($e$) is calculated as:

        .. math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        return (cv.energy - self.kinetic_energy(cv))

    def get_density(self, pressure, temperature, species_fractions):
        r"""Get the density from pressure, temperature, and species fractions (Y).

        The gas density $\rho$ is calculated from pressure, temperature and $R$ as:

        .. math::

            \rho = \frac{p}{R_s T}
        """
        return self._pyrometheus_mech.get_density(pressure, temperature,
                                                  species_fractions)

    def get_internal_energy(self, temperature, species_fractions):
        r"""Get the gas thermal energy from temperature, and species fractions (Y).

        The gas internal energy $e$ is calculated from:

        .. math::

            e = R_s T \sum{Y_\alpha h_\alpha}
        """
        return self._pyrometheus_mech.get_mixture_internal_energy_mass(
            temperature, species_fractions)

    def get_species_molecular_weights(self):
        """Get the species molecular weights."""
        return self._pyrometheus_mech.wts

    def get_production_rates(self, cv: ConservedVars):
        """Get the production rate for each species."""
        temperature = self.temperature(cv)
        y = self.species_fractions(cv)
        return self._pyrometheus_mech.get_net_production_rates(
            cv.mass, temperature, y)

    def species_fractions(self, cv: ConservedVars):
        r"""Get species fractions $Y_\alpha$ from species mass density."""
        return cv.species_mass / cv.mass

    def pressure(self, cv: ConservedVars):
        r"""Get thermodynamic pressure of the gas.

        Gas pressure ($p$) is calculated from the internal energy ($e$) as:

        .. math::

            p = (\gamma_{\mathtt{mix}} - 1)e
        """
        temperature = self.temperature(cv)
        y = self.species_fractions(cv)
        return self._pyrometheus_mech.get_pressure(cv.mass, temperature, y)

    def sound_speed(self, cv: ConservedVars):
        r"""Get the speed of sound in the gas.

        The speed of sound ($c$) is calculated as:

        .. math::

            c = \sqrt{\frac{\gamma_{\mathtt{mix}}{p}}{\rho}}
        """
        actx = cv.mass.array_context
        c2 = (self.gamma(cv) * self.pressure(cv)) / cv.mass
        return actx.np.sqrt(c2)

    def temperature(self, cv: ConservedVars):
        r"""Get the thermodynamic temperature of the gas.

        The thermodynamic temperature ($T$) is calculated from
        the internal energy ($e$) and specific gas constant ($R_s$)
        as:

        .. math::

            T = \frac{(\gamma_{\mathtt{mix}} - 1)e}{R_s \rho}
        """
        y = self.species_fractions(cv)
        e = self.internal_energy(cv) / cv.mass
        return self._pyrometheus_mech.get_temperature(e, self._tguess, y, True)

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
        """
        return (pressure / (self.gamma(cv) - 1.0)
                + self.kinetic_energy(cv))

    def get_species_source_terms(self, cv: ConservedVars):
        """Get the species mass source terms to be used on the RHS for chemistry."""
        omega = self.get_production_rates(cv)
        w = self.get_species_molecular_weights()
        species_sources = w * omega
        rho_source = 0 * cv.mass
        mom_source = 0 * cv.momentum
        energy_source = 0 * cv.energy

        return ConservedVars(rho_source, energy_source, mom_source, species_sources)
