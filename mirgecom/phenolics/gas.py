"""Evaluate gas properties based on tabulated data.

Gas-Handling Functions
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GasProperties

Helper Functions
================

.. autofunction:: eval_spline
.. autofunction:: eval_spline_derivative

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
from scipy.interpolate import CubicSpline  # type: ignore[import]


def eval_spline(x, x_bnds, coeffs):
    """Evaluate spline."""
    actx = x.array_context

    val = x*0.0
    for i in range(0, len(x_bnds)-1):
        val = actx.np.where(actx.np.less(x, x_bnds[i+1]),
                actx.np.where(actx.np.greater_equal(x, x_bnds[i]),
                    coeffs[0, i]*(x-x_bnds[i])**3
                    + coeffs[1, i]*(x-x_bnds[i])**2
                    + coeffs[2, i]*(x-x_bnds[i])
                    + coeffs[3, i], 0.0), 0.0) + val

    return val


def eval_spline_derivative(x, x_bnds, coeffs):
    """Evaluate analytical derivative of a spline."""
    actx = x.array_context

    val = x*0.0
    for i in range(0, len(x_bnds)-1):
        val = actx.np.where(actx.np.less(x, x_bnds[i+1]),
                actx.np.where(actx.np.greater_equal(x, x_bnds[i]),
                    3.0*coeffs[0, i]*(x-x_bnds[i])**2
                    + 2.0*coeffs[1, i]*(x-x_bnds[i])
                    + coeffs[2, i], 0.0), 0.0) + val

    return val


class GasProperties():
    """Simplified model of the pyrolysis gas using tabulated data.

    This section is to be used when species conservation is not employed and
    the output gas is assumed to be in chemical equilibrium.
    """

    def __init__(self, prandtl=1.0, lewis=1.0):
        """Return gas tabulated data and interpolating functions."""
        self._prandtl = prandtl
        self._lewis = lewis

        #    T     , M      ,  Cp    , gamma  ,  enthalpy, viscosity
        gas_data = np.array([
            [200.00,  21.996,  1.5119,  1.3334,  -7246.50,  0.086881],
            [350.00,  21.995,  1.7259,  1.2807,  -7006.30,  0.144380],
            [500.00,  21.948,  2.2411,  1.2133,  -6715.20,  0.196150],
            [650.00,  21.418,  4.3012,  1.1440,  -6265.70,  0.243230],
            [700.00,  20.890,  6.3506,  1.1242,  -6004.60,  0.258610],
            [750.00,  19.990,  9.7476,  1.1131,  -5607.70,  0.274430],
            [800.00,  18.644,  14.029,  1.1116,  -5014.40,  0.290920],
            [850.00,  17.004,  17.437,  1.1171,  -4218.50,  0.307610],
            [900.00,  15.457,  17.009,  1.1283,  -3335.30,  0.323490],
            [975.00,  14.119,  8.5576,  1.1620,  -2352.90,  0.344350],
            [1025.0,  13.854,  4.7840,  1.1992,  -2034.20,  0.356630],
            [1100.0,  13.763,  3.5092,  1.2240,  -1741.20,  0.373980],
            [1150.0,  13.737,  3.9008,  1.2087,  -1560.90,  0.385360],
            [1175.0,  13.706,  4.8067,  1.1899,  -1453.50,  0.391330],
            [1200.0,  13.639,  6.2353,  1.1737,  -1315.90,  0.397930],
            [1275.0,  13.256,  8.4790,  1.1633,  -739.700,  0.421190],
            [1400.0,  12.580,  9.0239,  1.1583,  353.3100,  0.458870],
            [1525.0,  11.982,  11.516,  1.1377,  1608.400,  0.483230],
            [1575.0,  11.732,  12.531,  1.1349,  2214.000,  0.487980],
            [1625.0,  11.495,  11.514,  1.1444,  2826.800,  0.491950],
            [1700.0,  11.255,  7.3383,  1.1849,  3529.400,  0.502120],
            [1775.0,  11.139,  5.3118,  1.2195,  3991.000,  0.516020],
            [1925.0,  11.046,  4.2004,  1.2453,  4681.800,  0.545280],
            [2000.0,  11.024,  4.0784,  1.2467,  4991.300,  0.559860],
            [2150.0,  10.995,  4.1688,  1.2382,  5605.400,  0.588820],
            [2300.0,  10.963,  4.5727,  1.2214,  6257.300,  0.617610],
            [2450.0,  10.914,  5.3049,  1.2012,  6993.500,  0.646380],
            [2600.0,  10.832,  6.4546,  1.1815,  7869.600,  0.675410],
            [2750.0,  10.701,  8.1450,  1.1650,  8956.900,  0.705000],
            [2900.0,  10.503,  10.524,  1.1528,  10347.00,  0.735570],
            [3050.0,  10.221,  13.755,  1.1449,  12157.00,  0.767590],
            [3200.0,  9.8394,  17.957,  1.1408,  14523.00,  0.801520],
            [3350.0,  9.3574,  22.944,  1.1401,  17584.00,  0.837430],
        ])

        self._data = gas_data
        self._cs_molar_mass = CubicSpline(gas_data[:, 0], gas_data[:, 1])
        self._cs_enthalpy = CubicSpline(gas_data[:, 0], gas_data[:, 4]*1000.0)
        self._cs_viscosity = CubicSpline(gas_data[:, 0], gas_data[:, 5]*1e-4)

    def gas_enthalpy(self, temperature):
        """Return the gas enthalpy."""
        coeffs = self._cs_enthalpy.c
        bnds = self._cs_enthalpy.x
        return eval_spline(temperature, bnds, coeffs)

    def gas_heat_capacity(self, temperature):
        """Return the gas heat capacity."""
        coeffs = self._cs_enthalpy.c
        bnds = self._cs_enthalpy.x
        return eval_spline_derivative(temperature, bnds, coeffs)

    def gas_molar_mass(self, temperature):
        """Return the gas molar mass."""
        coeffs = self._cs_molar_mass.c
        bnds = self._cs_molar_mass.x
        return eval_spline(temperature, bnds, coeffs)

    def gas_dMdT(self, temperature):  # noqa N802
        """Return the partial derivative of molar mass wrt temperature."""
        coeffs = self._cs_molar_mass.c
        bnds = self._cs_molar_mass.x
        return eval_spline_derivative(temperature, bnds, coeffs)

    def gas_viscosity(self, temperature):
        """Return gas viscosity."""
        coeffs = self._cs_viscosity.c
        bnds = self._cs_viscosity.x
        return eval_spline(temperature, bnds, coeffs)

    def gas_thermal_conductivity(self, temperature):
        r"""Return gas thermal conductivity.

        .. math::

            \kappa = \frac{\mu C_p}{Pr}

        where $\mu$ is the gas viscosity and $C_p$ is the heat capacity at
        constant pressure. The Prandtl number $Pr$ is assumed to be 1.
        """
        cp = self.gas_heat_capacity(temperature)
        mu = self.gas_viscosity(temperature)
        prandtl = 1.0
        return mu*cp/prandtl
