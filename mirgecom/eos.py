"""
:mod:`mirgecom.eos` provides implementations of gas equations of state.

Equations of State
^^^^^^^^^^^^^^^^^^
This module is designed provide Equation of State objects used to compute and
manage the relationships between and among state and thermodynamic variables.

.. autoclass:: EOSDependentVars
.. autoclass:: GasEOS
.. autoclass:: IdealSingleGas
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
from mirgecom.euler import ConservedVars

r"""
This module is designed provide Equation of State
objects used to compute and manage the relationships
between and among state and thermodynamic variables.
"""


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
    object array representing the state vector (:math:`q`),
    contains the relevant simulation state quantities. Each
    EOS class should document its own state data requirements.

    .. automethod:: pressure
    .. automethod:: temperature
    .. automethod:: sound_speed
    .. automethod:: internal_energy
    .. automethod:: gas_const
    .. automethod:: dependent_vars
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
        r"""Get the specific gas constant (:math:`R`)."""
        raise NotImplementedError()

    def internal_energy(self, cv: ConservedVars):
        """Get the thermal energy of the gas."""
        raise NotImplementedError()

    def dependent_vars(self, cv: ConservedVars) -> EOSDependentVars:
        """Get an agglomerated array of the depedent variables."""
        return EOSDependentVars(
            pressure=self.pressure(cv),
            temperature=self.temperature(cv),
            )


class IdealSingleGas(GasEOS):
    r"""Ideal gas law single-component gas (:math:`p = \rho{R}{T}`).

    The specific gas constant, R, defaults to the air-like 287.1 J/(kg*K),
    but can be set according to simulation units and materials.

    Each interface call expects that the agglomerated
    object array representing the state vector (:math:`q`),
    contains at least the canonical conserved quantities
    mass (:math:`\rho`), energy (:math:`\rho{E}`), and
    momentum (:math:`\rho\vec{V}`).

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

    def internal_energy(self, cv: ConservedVars):
        r"""Get internal thermal energy of gas.

        The internal energy (e) is calculated as:
        .. :math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mom = cv.momentum
        return (cv.energy - 0.5 * np.dot(mom, mom) / cv.mass)

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

    def energy(self, mass, pressure, momentum):
        r"""Get gas total energy from mass, pressure, momentum.

        The total energy density (rhoE) is calculated from
        the mass density (rho) , pressure (p) , and
        momentum (rhoV) as:

        .. :math::

            \rhoE = \frac{p}{(\gamma - 1)} +
            \frac{1}{2}\rho(\vec{v} \cdot \vec{v})
        """
        return (pressure / (self._gamma - 1.0)
                + 0.5 * np.dot(momentum, momentum) / mass)
    #
    #    def __call__(self, q):
    #        return dependent_vars(q)
    #
    #    def split_fields(self, ndim, dv):
    #        return [
    #            ("pressure", dv[0]),
    #            ("temperature", dv[1]),
    #        ]
