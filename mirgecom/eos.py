"""
:mod:`mirgecom.eos` provides implementations of gas equations of state.

Equations of State
^^^^^^^^^^^^^^^^^^
This module is designed provide Equation of State objects used to compute and
manage the relationships between and among state and thermodynamic variables.

.. autoclass:: EOSDependentVars
.. autoclass:: GasEOS
.. autoclass:: IdealSingleGas
.. autoclass:: PrometheusMixture
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
from pytools.obj_array import make_obj_array

import numpy as np
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.euler import ConservedVars


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

    Each interface call expects that the agglomerated
    object array representing the state vector ($q$),
    contains the relevant simulation state quantities. Each
    EOS class should document its own state data requirements.

    .. automethod:: pressure
    .. automethod:: temperature
    .. automethod:: sound_speed
    .. automethod:: internal_energy
    .. automethod:: gas_const
    .. automethod:: dependent_vars
    .. automethod:: total_energy
    .. automethod:: kinetic_energy
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

    def gas_const(self, cv: ConservedVars):
        r"""Get the specific gas constant ($R$)."""
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

    def dependent_vars(self, q: ConservedVars) -> EOSDependentVars:
        """Get an agglomerated array of the depedent variables."""
        return EOSDependentVars(
            pressure=self.pressure(q),
            temperature=self.temperature(q),
            )


class IdealSingleGas(GasEOS):
    r"""Ideal gas law single-component gas ($p = \rho{R}{T}$).

    The specific gas constant, R, defaults to the air-like 287.1 J/(kg*K),
    but can be set according to simulation units and materials.

    Each interface call expects that the agglomerated
    object array representing the state vector ($q$),
    contains at least the canonical conserved quantities
    mass ($\rho$), energy ($\rho{E}$), and
    momentum ($\rho\vec{V}$).

    .. automethod:: __init__
    .. automethod:: gamma

    Inherits from (and implements) :class:`GasEOS`.
    """

    def __init__(self, gamma=1.4, gas_const=287.1):
        """Initialize Ideal Gas EOS parameters."""
        self._gamma = gamma
        self._gas_const = gas_const

    def gamma(self):
        """Get specific heat ratio Cp/Cv."""
        return self._gamma

    def gas_const(self):
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


class PrometheusMixture(GasEOS):
    r"""Ideal gas mixture (:math:`p = \bar{\rho}{R}_\mathtt{mix}{T}`).

    The mixture gas constant, :math:`R_\mathtt{mix}`, is calculated
    as :math:`R_\mathtt{mix} = \sum{Y_\alpha R_\alpha}` by the _Prometheus_
    mechanism provided by the user.

    Each interface call expects that the agglomerated
    object array representing the state vector (:math:`q`),
    contains at least the canonical conserved quantities
    mass (:math:`\rho`), energy (:math:`\rho{E}`), and
    momentum (:math:`\rho\vec{V}`) and the vector of
    mass fractions for each species (:math:`Y_\alpha`).

    .. automethod:: __init__
    .. automethod:: gamma

    Inherits from (and implements) :class:`GasEOS`.
    """

    def __init__(self, prometheus_mech):
        """Initialize Prometheus EOS with mechanism class."""
        self._prometheus_mech = prometheus_mech
        self._gamma = 1.4

    def gamma(self):
        """Get specific heat ratio Cp/Cv."""
        return self._gamma

    def gas_const(self):
        """Get specific gas constant R."""
        return self._prometheus_mech.gas_constant

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

    def mass_fractions(self, cv: ConservedVars):
        r"""Get mass fractions :math:`\Y_\alpha` from species densities."""

        return cv.massfractions * make_obj_array([1.0/cv.mass])

    def pressure(self, cv: ConservedVars):
        r"""Get thermodynamic pressure of the gas.

        Gas pressure (p) is calculated from the internal energy (e) as:

        .. :math::

            p = (\gamma - 1)e
        """
        temperature = self.temperature(cv)
        y = self.mass_fractions(cv)
        return self._prometheus_mech.get_pressure(cv.mass, temperature, y)

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
        y = self.mass_fractions(cv)
        e = self.internal_energy(cv) / cv.mass
        tguess = 300.0
        return self._prometheus_mech.get_temperature(e, tguess, y, True)

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
