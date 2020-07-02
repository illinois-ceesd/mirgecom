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
import pyopencl.clmath as clmath
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa


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
        """
        raise NotImplementedError()

    def __call__(self, w=None):
        r"""Call to object

        Parameters
        ----------
        w
            State array which expects at least
            the canonical conserved quantities:
            :math:`w=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        object array of floating point arrays with the
        EOS-specific dependent variables.
        (e.g. [ pressure, temperature ] for an ideal gas)
        """
        raise NotImplementedError()

    def pressure(self, w=None):
        r"""Gas pressure

        Parameters
        ----------
        w
            State array which expects at least
            the canonical conserved quantities:
            :math:`w=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point array with the thermodynamic
        pressure at each point
        """
        raise NotImplementedError()

    def temperature(self, w=None):
        r"""Gas temperature

        Parameters
        ----------
        w
            State array which expects at least
            the canonical conserved quantities:
            :math:`w=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point array with the thermodynamic
        temperature at each point
        """
        raise NotImplementedError()

    def sound_speed(self, w=None):
        r"""Speed of sound

        Parameters
        ----------
        w
            State array which expects at least
            the canonical conserved quantities:
            :math:`w=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point number or array with the speed of sound
        for each point
        """
        raise NotImplementedError()

    def gas_const(self, w=None):
        r"""Specific gas constant (R)

        Parameters
        ----------
        w
            State array which expects at least
            the canonical conserved quantities:
            :math:`w=[\rho,\rho{E},\rho\vec{V}]`
            for the fluid at each point

        Returns
        ----------
        floating point number or array with the specific
        gas constant for the domain, or at each point for
        mixture-type EOS.
        """
        raise NotImplementedError()

    def internal_energy(self, w=None):
        r"""Internal energy

        Parameters
        ----------
        w
            State array which expects at least
            the canonical conserved quantities:
            :math:`w=[\rho,\rho{E},\rho\vec{V}]`
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

    def internal_energy(self, w):
        r"""Internal energy

        The internal energy (e) is calculated as:
        .. :math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mass = w[0]
        energy = w[1]
        mom = w[2:]

        return (energy - 0.5 * np.dot(mom, mom) / mass)

    def pressure(self, w):
        r"""Gas pressure

        The thermodynmic pressure (p) is calculated from
        the internal energy (e) as:

        .. :math::

            p = (\gamma - 1)e
        """
        return (self._gamma - 1.0) * self.internal_energy(w)

    def sound_speed(self, w):
        r"""Speed of sound

        The speed of sound (c) is calculated as:

        .. :math::

            c = \sqrt{\frac{\gamma{p}}{\rho}}
        """
        mass = w[0]
        p = self.pressure(w)
        c2 = self._gamma / mass * p
        return clmath.sqrt(c2)

    def temperature(self, w):
        r"""Gas temperature

        The thermodynmic temperature (T) is calculated from
        the internal energy (e) and specific gas constant (R)
        as:

        .. :math::

            T = \frac{(\gamma - 1)e}{R\rho}
        """
        mass = w[0]
        return (
            ((self._gamma - 1.0) / self._gas_const) * self.internal_energy(w) / mass
        )

    def __call__(self, w):
        return flat_obj_array(self.pressure(w), self.temperature(w))

    def split_fields(self, ndim, dv):
        return [
            ("pressure", dv[0]),
            ("temperature", dv[1]),
        ]
