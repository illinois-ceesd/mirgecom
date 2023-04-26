"""Evaluate composite data.

Pyrolysis-Handling Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Pyrolysis

"""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
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
from meshmode.dof_array import DOFArray
import scipy


class BprimeTable():
    """Class containing the table for wall properties."""

    def __init__(self):

        # bprime contains: B_g, B_c, Temperature T, Wall enthalpy H_W
        bprime_table = (np.genfromtxt("../Bprime_table/B_prime.dat",
                                      skip_header=1)[:, 2:6]).reshape((25,151,4))  # noqa E501

        self._bounds_T = bprime_table[   0, :-1:6, 2]  # noqa E201
        self._bounds_B = bprime_table[::-1, 0, 0]
        self._Bc = bprime_table[::-1, :, 1]
        self._Hw = bprime_table[::-1, :-1:6, 3]

        # create spline to interpolate the wall enthalpy
        self._cs_Hw = np.zeros((25, 4, 24))
        for i in range(0, 25):
            self._cs_Hw[i, :, :] = scipy.interpolate.CubicSpline(
                self._bounds_T, self._Hw[i, :]).c


class Pyrolysis():
    """Pyrolysis class.

    .. automethod:: get_sources
    """

    def __init__(self):
        self._Tcrit = np.array([333.3, 555.6])
        # self._Fij = np.array([0.025, 0.075])
        # self._n_phases = 2

    def get_sources(self, temperature: DOFArray, xi):
        r"""Return the source terms of pyrolysis decomposition.

        The source terms follow as Arrhenius-like equation given by

        .. math::

            \dot{\omega}_i^p = \mathcal{A}_{i} T^{n_{i}}
            \exp\left(- \frac{E_{i}}{RT} \right)
            \left( \frac{\epsilon_i \rho_i -
                \epsilon^c_i \rho^c_i}{\epsilon^0_i \rho^0_i} \right)^{m_i}

        """
        actx = temperature.array_context

        rhs = np.empty((3,), dtype=object)

        rhs[0] = actx.np.where(actx.np.less(temperature, self._Tcrit[0]),
            0.0,
            -(30.*((xi[0] - 0.00)/30.)**3)*12000.*actx.np.exp(-8556.000/temperature)
        )
        rhs[1] = actx.np.where(actx.np.less(temperature, self._Tcrit[1]),
            0.0,
            -(90.*((xi[1] - 60.0)/90.)**3)*4.48e9*actx.np.exp(-20444.44/temperature)
        )

        # include the fiber in the RHS but dont do anything more for now.
        # ignore oxidation, for now...
        # at some point, Y2 model can be included here...
        rhs[2] = temperature*0.0

        return rhs


def solid_enthalpy(temperature, tau):
    """."""
    virgin = (
        - 1.36068885310508e-11*temperature**5 + 1.52102962615076e-07*temperature**4
        - 6.73376995865907e-04*temperature**3 + 1.49708228272951e+00*temperature**2
        + 3.00986515698487e+02*temperature - 1.06276798377448e+06)

    charr = (
        - 1.27988769472902e-11*temperature**5 + 1.49117546528569e-07*temperature**4
        - 6.99459529686087e-04*temperature**3 + 1.69156401810899e+00*temperature**2
        - 3.44183740832012e+01*temperature - 1.23543810449620e+05)

    return virgin*tau + charr*(1.0 - tau)


def solid_heat_capacity(temperature, tau):
    """."""
    actx = temperature.array_context

    virgin = actx.np.where(actx.np.less(temperature, 2222.0),
        + 4.12265891689180e-14*temperature**5 - 4.43093718060442e-10*temperature**4
        + 1.87206033562391e-06*temperature**3 - 3.95146486560366e-03*temperature**2
        + 4.29108093873644e+00*temperature + 1.39759434036202e+01,
        2008.8139143251735)

    charr = (
        + 1.46130366932393e-14*temperature**5 - 1.86248970158190e-10*temperature**4
        + 9.68539883053023e-07*temperature**3 - 2.59975526254095e-03*temperature**2
        + 3.66729551084460e+00*temperature - 7.81621843565539e+01)

    return virgin*tau + charr*(1.0 - tau)


def solid_thermal_conductivity(temperature, tau):
    """."""
    virgin = (
        + 2.31290019732353e-17*temperature**5 - 2.16778503256247e-13*temperature**4
        + 8.24498395180905e-10*temperature**3 - 1.22161245622351e-06*temperature**2
        + 8.46459266618945e-04*temperature + 2.38711268975591e-01)

    charr = (
        - 7.37827990887776e-18*temperature**5 + 4.70935349841195e-14*temperature**4
        + 1.53023689925812e-11*temperature**3 - 2.30561135245248e-07*temperature**2
        + 3.66862488656913e-04*temperature + 3.12089881488869e-01)

    return virgin*tau + charr*(1.0 - tau)


def solid_permeability(tau):
    """."""
    virgin = 1.6e-11
    charr = 2.0e-11
    return virgin*tau + charr*(1.0 - tau)


def solid_tortuosity(temperature, tau):
    """."""
    virgin = 1.2e-11
    charr = 1.1e-11
    return virgin*tau + charr*(1.0 - tau)


def solid_volume_fraction(tau):
    """."""
    fiber = 0.10
    virgin = 0.10
    charr = 0.05
    return virgin*tau + charr*(1.0 - tau) + fiber


def solid_emissivity(tau):
    """."""
    virgin = 0.8
    charr = 0.9
    return virgin*tau + charr*(1.0 - tau)
