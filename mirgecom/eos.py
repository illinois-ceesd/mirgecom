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

import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import flat_obj_array
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

r"""
This module is designed provide Equation of State
objects used to compute and manage the relationships
between and among state and thermodynamic variables.
"""


class GasEOS:
    r"""Implements an object designed to provide methods
    for implementing and computing relations between
    fluid or gas state variables.

    .. automethod :: pressure
    .. automethod :: temperature
    .. automethod :: sound_speed
    .. automethod :: internal_energy
    .. automethod :: gas_const
    """

    def __init__(self, parameters=None):
        r"""Initialize routine should take any relevant
        parameters expected to remain constant throughout
        the simulation.

        Parameters
        ----------
        parameters
            Dictionary containing the eos-relevant parameters.
        """
        raise NotImplementedError()

    def __call__(self, q=None):
        r"""Call to object

        Parameters
        ----------
        q
            State array which expects at least
            the canonical conserved quantities:
            :math:`q=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        object array of floating point arrays with the
        EOS-specific dependent variables.
        (e.g. [ pressure, temperature ] for an ideal gas)
        """
        raise NotImplementedError()

    def pressure(self, q=None):
        r"""Gas pressure

        Parameters
        ----------
        q
            State array which expects at least
            the canonical conserved quantities:
            :math:`q=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point array with the thermodynamic
        pressure at each point
        """
        raise NotImplementedError()

    def temperature(self, q=None):
        r"""Gas temperature

        Parameters
        ----------
        q
            State array which expects at least
            the canonical conserved quantities:
            :math:`q=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point array with the thermodynamic
        temperature at each point
        """
        raise NotImplementedError()

    def sound_speed(self, q=None):
        r"""Speed of sound

        Parameters
        ----------
        q
            State array which expects at least
            the canonical conserved quantities:
            :math:`q=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point number or array with the speed of sound
        for each point
        """
        raise NotImplementedError()

    def gas_const(self, q=None):
        r"""Specific gas constant (R)

        Parameters
        ----------
        q
            State array which expects at least
            the canonical conserved quantities:
            :math:`q=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point number or array with the specific
        gas constant for the domain, or at each point for
        mixture-type EOS.
        """
        raise NotImplementedError()

    def internal_energy(self, q=None):
        r"""Internal energy

        Parameters
        ----------
        q
            State array which expects at least
            the canonical conserved quantities:
            :math:`q=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point array with the internal
        energy at each point.
        """
        raise NotImplementedError()


class IdealSingleGas:
    r"""Implements the ideal gas law (:math:`p = \rho{R}{T}`)
    for a single monatomic gas.
    """

    def __init__(self, gamma=1.4, gas_const=287.1):
        self._gamma = gamma
        self._gas_const = gas_const

    def gamma(self):
        return self._gamma

    def gas_const(self):
        return self._gas_const

    def internal_energy(self, q):
        r"""Internal energy

        The internal energy (e) is calculated as:
        .. :math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mass = q[0]
        energy = q[1]
        mom = q[2:]

        return (energy - 0.5 * np.dot(mom, mom) / mass)

    def pressure(self, q):
        r"""Gas pressure

        The thermodynmic pressure (p) is calculated from
        the internal energy (e) as:

        .. :math::

            p = (\gamma - 1)e
        """
        return self.internal_energy(q) * (self._gamma - 1.0)

    def sound_speed(self, q):
        r"""Speed of sound

        The speed of sound (c) is calculated as:

        .. :math::

            c = \sqrt{\frac{\gamma{p}}{\rho}}
        """
        mass = q[0]
        actx = mass.array_context

        p = self.pressure(q)
        c2 = self._gamma / mass * p
        return actx.np.sqrt(c2)

    def temperature(self, q):
        r"""Gas temperature

        The thermodynmic temperature (T) is calculated from
        the internal energy (e) and specific gas constant (R)
        as:

        .. :math::

            T = \frac{(\gamma - 1)e}{R\rho}
        """
        mass = q[0]
        return (
            ((self._gamma - 1.0) / self._gas_const) * self.internal_energy(q) / mass
        )

    def __call__(self, q):
        return flat_obj_array(self.pressure(q), self.temperature(q))

    def split_fields(self, ndim, dv):
        return [
            ("pressure", dv[0]),
            ("temperature", dv[1]),
        ]
