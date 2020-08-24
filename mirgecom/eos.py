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
from pytools.obj_array import flat_obj_array
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

__doc__ = """
Equations of State
^^^^^^^^^^^^^^^^^^
This module is designed provide Equation of State objects used to compute and
manage the relationships between and among state and thermodynamic variables.

.. autoclass:: GasEOS
.. autoclass:: IdealSingleGas
"""


class GasEOS:
    r"""An abstract interface designed to compute relations between
    fluid or gas state variables.

    Each interface call expects that the agglomerated
    object array representing the state vector (:math:`q`),
    contains the relevant simulation state quantities. Each
    EOS class should document its own state data requirements.

    .. automethod :: get_pressure
    .. automethod :: get_temperature
    .. automethod :: get_sound_speed
    .. automethod :: get_internal_energy
    .. automethod :: get_gas_const
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
            Agglomerated object array representing state data

        Returns
        ----------
        object array of floating point arrays with the
        EOS-specific dependent variables.
        (e.g. [ pressure, temperature ] for an ideal gas)
        """
        raise NotImplementedError()

    def get_pressure(self, q=None):
        r"""Get gas pressure

        Parameters
        ----------
        q
            Agglomerated object array representing state data

        Returns
        ----------
        floating point array with the thermodynamic
        pressure at each point
        """
        raise NotImplementedError()

    def get_temperature(self, q=None):
        r"""Get gas temperature

        Parameters
        ----------
        q
            Agglomerated object array representing state data

        Returns
        ----------
        floating point array with the thermodynamic
        temperature at each point
        """
        raise NotImplementedError()

    def get_sound_speed(self, q=None):
        r"""Get speed of sound

        Parameters
        ----------
        q
            Agglomerated object array representing state data

        Returns
        ----------
        floating point number or array with the speed of sound
        for each point
        """
        raise NotImplementedError()

    def get_gas_const(self, q=None):
        r"""Get specific gas constant (R)

        Parameters
        ----------
        q
            Agglomerated object array representing state data

        Returns
        ----------
        floating point number or array with the specific
        gas constant for the domain, or at each point for
        mixture-type EOS.
        """
        raise NotImplementedError()

    def get_internal_energy(self, q=None):
        r"""get internal energy

        Parameters
        ----------
        q
            Agglomerated object array representing state data

        Returns
        ----------
        floating point array with the internal
        energy at each point.
        """
        raise NotImplementedError()


class IdealSingleGas:
    r"""Implement the ideal gas law (:math:`p = \rho{R}{T}`)
    for a single monatomic gas.

    The specific gas constant, R, defaults to the air-like 287.1 J/(kg*K),
    but can be set according to simulation units and materials.

    Each interface call expects that the agglomerated
    object array representing the state vector (:math:`q`),
    contains at least the canonical conserved quantities
    mass (:math:`\rho`), energy (:math:`\rho{E}`), and
    momentum (:math:`\rho\vec{V}`).

    """

    def __init__(self, gamma=1.4, gas_const=287.1):
        self._gamma = gamma
        self._gas_const = gas_const

    def get_gamma(self):
        return self._gamma

    def get_gas_const(self):
        return self._gas_const

    def get_internal_energy(self, q):
        r"""Get internal energy

        The internal energy (e) is calculated as:
        .. :math::

            e = \rho{E} - \frac{1}{2\rho}(\rho\vec{V} \cdot \rho\vec{V})
        """
        mass = q[0]
        energy = q[1]
        mom = q[2:]

        return (energy - 0.5 * np.dot(mom, mom) / mass)

    def get_pressure(self, q):
        r"""get gas pressure

        The thermodynmic pressure (p) is calculated from
        the internal energy (e) as:

        .. :math::

            p = (\gamma - 1)e
        """
        return self.get_internal_energy(q) * (self._gamma - 1.0)

    def get_sound_speed(self, q):
        r"""Get speed of sound

        The speed of sound (c) is calculated as:

        .. :math::

            c = \sqrt{\frac{\gamma{p}}{\rho}}
        """
        mass = q[0]
        actx = mass.array_context

        p = self.get_pressure(q)
        c2 = self._gamma / mass * p
        return actx.np.sqrt(c2)

    def get_temperature(self, q):
        r"""Get gas temperature

        The thermodynmic temperature (T) is calculated from
        the internal energy (e) and specific gas constant (R)
        as:

        .. :math::

            T = \frac{(\gamma - 1)e}{R\rho}
        """
        mass = q[0]
        return (
            (((self._gamma - 1.0) / self._gas_const)
            * self.get_internal_energy(q) / mass)
        )

    def __call__(self, q):
        """Get the dimensional pressure and temperature
        """
        return flat_obj_array(self.get_pressure(q), self.get_temperature(q))

    def split_fields(self, ndim, dependent_vars):
        return [
            ("pressure", dependent_vars[0]),
            ("temperature", dependent_vars[1]),
        ]
