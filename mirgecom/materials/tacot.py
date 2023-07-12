""":mod:`~mirgecom.materials.tacot` evaluates TACOT-related data.

    TACOT is the Theoretical Ablative Composite for Open Testing, a surrogate
    composite material closely related to PICA (Phenolic Impregnated Carbon
    Ablator) proposed as an open-source material for the Ablation Workshop.

.. autoclass:: Pyrolysis
.. autoclass:: SolidProperties
.. autoclass:: GasProperties
.. autoclass:: BprimeTable
.. autoclass:: WallTabulatedEOS

Helper Functions
----------------
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
import scipy  # type: ignore[import]
from scipy.interpolate import CubicSpline  # type: ignore[import]
from meshmode.dof_array import DOFArray
from pytools.obj_array import make_obj_array
from mirgecom.fluid import ConservedVars
from mirgecom.wall_model import WallDependentVars, WallEOS


class BprimeTable:
    """Class containing the table for energy balance at the surface.

    This class is only required for uncoupled cases, where only the wall
    portion is evaluated. This is NOT used for fully-coupled cases.
    """

    def __init__(self, bprime_table):

        # bprime contains: B_g, B_c, Temperature T, Wall enthalpy H_W
        self._bounds_T = bprime_table[   0, :-1:6, 2]  # noqa E201
        self._bounds_B = bprime_table[::-1, 0, 0]
        self._Bc = bprime_table[::-1, :, 1]
        self._Hw = bprime_table[::-1, :-1:6, 3]

        # create spline to interpolate the wall enthalpy
        self._cs_Hw = np.zeros((25, 4, 24))
        for i in range(0, 25):
            self._cs_Hw[i, :, :] = scipy.interpolate.CubicSpline(
                self._bounds_T, self._Hw[i, :]).c


class Pyrolysis:
    r"""Evaluate the source terms for the pyrolysis decomposition.

    The source terms follow as Arrhenius-like equation given by

    .. math::

        \dot{\omega}_i^p = \mathcal{A}_{i} T^{n_{i}}
        \exp\left(- \frac{E_{i}}{RT} \right)
        \left( \frac{\epsilon_i \rho_i -
            \epsilon^c_i \rho^c_i}{\epsilon^0_i \rho^0_i} \right)^{m_i}

    For TACOT, 2 different reactions, which are assumed to only happen after
    a minimum temperature, are considered based on the resin constituents.
    The third reaction is the fiber oxidation, which is not handled here for now.

    .. automethod:: get_source_terms
    """

    def __init__(self):
        """Temperature in which each reaction starts."""
        self._Tcrit = np.array([333.3, 555.6])

    def get_source_terms(self, temperature, chi):
        r"""Return the source terms of pyrolysis decomposition for TACOT.

        Parameters
        ----------
        temperature: :class:`~meshmode.dof_array.DOFArray`
            The temperature of the bulk material.

        chi: numpy.ndarray
            Either the solid mass $\rho_i$ of all fractions of the resin or
            the progress ratio $\chi$ of the decomposition. The actual
            parameter depends on the modeling itself.

        Returns
        -------
        source: numpy.ndarray
            The source terms for the pyrolysis
        """
        actx = temperature.array_context

        return make_obj_array([
            # reaction 1
            actx.np.where(actx.np.less(temperature, self._Tcrit[0]),
                0.0, (
                    -(30.*((chi[0] - 0.00)/30.)**3)*12000.
                    * actx.np.exp(-8556.000/temperature))),
            actx.np.where(actx.np.less(temperature, self._Tcrit[1]),
            # reaction 2
            0.0, (
                -(90.*((chi[1] - 60.0)/90.)**3)*4.48e9
                * actx.np.exp(-20444.44/temperature))),
            # fiber oxidation: include in the RHS but dont do anything with it.
            temperature*0.0])


def eval_spline(x, x_bnds, coeffs) -> DOFArray:
    r"""Evaluate spline $a(x-x_i)^3 + b(x-x_i)^2 + c(x-x_i) + d$.

    Parameters
    ----------
    x: :class:`~meshmode.dof_array.DOFArray`
        The value where $f(x)$ will be evaluated.
    x_bnds: :class:`numpy.ndarray` with shape ``(m,)``
        The $m$ nodes $x_i$ for the different segments of the spline.
    coeffs: :class:`numpy.ndarray` with shape ``(4,m)``
        The 4 coefficients for each segment $i$ of the spline.
    """
    actx = x.array_context

    val = x*0.0
    for i in range(0, len(x_bnds)-1):
        val = (
            actx.np.where(actx.np.less(x, x_bnds[i+1]),
            actx.np.where(actx.np.greater_equal(x, x_bnds[i]),
                coeffs[0, i]*(x-x_bnds[i])**3 + coeffs[1, i]*(x-x_bnds[i])**2
                + coeffs[2, i]*(x-x_bnds[i]) + coeffs[3, i], 0.0), 0.0)) + val

    return val


def eval_spline_derivative(x, x_bnds, coeffs) -> DOFArray:
    """Evaluate analytical derivative of a spline $3a(x-x_i)^2 + 2b(x-x_i) + c$.

    Parameters
    ----------
    x: :class:`~meshmode.dof_array.DOFArray`
        The value where $f(x)$ will be evaluatead.
    x_bnds: :class:`numpy.ndarray` with shape ``(m,)``
        The $m$ nodes $x_i$ for the different segments of the spline.
    coeffs: :class:`numpy.ndarray` with shape ``(4,m)``
        The 4 coefficients for each segment $i$ of the spline.
    """
    actx = x.array_context

    val = x*0.0
    for i in range(0, len(x_bnds)-1):
        val = (
            actx.np.where(actx.np.less(x, x_bnds[i+1]),
            actx.np.where(actx.np.greater_equal(x, x_bnds[i]),
                3.0*coeffs[0, i]*(x-x_bnds[i])**2 + 2.0*coeffs[1, i]*(x-x_bnds[i])
                + coeffs[2, i], 0.0), 0.0)) + val

    return val


class GasProperties:
    """Simplified model of the pyrolysis gas using tabulated data.

    This section is to be used when species conservation is not employed and
    the output gas is assumed to be in chemical equilibrium.
    The table was extracted from the suplementar material from the
    ablation workshop. Some lines were removed to reduce the number of spline
    interpolation segments.

    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: molar_mass
    .. automethod:: dMdT
    .. automethod:: viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: pressure
    """

    def __init__(self, prandtl=1.0, lewis=1.0):
        """Return gas tabulated data and interpolating functions.

        Parameters
        ----------
        prandtl: float
            the Prandtl number of the mixture. Defaults to 1.
        lewis: float
            the Lewis number of the mixture. Defaults to 1.
        """
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

    def enthalpy(self, temperature: DOFArray) -> DOFArray:
        r"""Return the gas enthalpy $h_g$."""
        coeffs = self._cs_enthalpy.c
        bnds = self._cs_enthalpy.x
        return eval_spline(temperature, bnds, coeffs)

    def heat_capacity(self, temperature: DOFArray) -> DOFArray:
        r"""Return the gas heat capacity at constant pressure $C_{p_g}$.

        The heat capacity is the derivative of the enthalpy. Thus, to improve
        accuracy and avoid issues with Newton iteration, this is computed
        exactly as the analytical derivative of the spline for the enthalpy.
        """
        coeffs = self._cs_enthalpy.c
        bnds = self._cs_enthalpy.x
        return eval_spline_derivative(temperature, bnds, coeffs)

    def molar_mass(self, temperature: DOFArray) -> DOFArray:
        r"""Return the gas molar mass $M$."""
        coeffs = self._cs_molar_mass.c
        bnds = self._cs_molar_mass.x
        return eval_spline(temperature, bnds, coeffs)

    def dMdT(self, temperature: DOFArray) -> DOFArray:  # noqa N802
        """Return the partial derivative of molar mass wrt temperature.

        This is necessary to evaluate the temperature using Newton iteration.
        """
        coeffs = self._cs_molar_mass.c
        bnds = self._cs_molar_mass.x
        return eval_spline_derivative(temperature, bnds, coeffs)

    def viscosity(self, temperature: DOFArray) -> DOFArray:
        r"""Return the gas viscosity $\mu$."""
        coeffs = self._cs_viscosity.c
        bnds = self._cs_viscosity.x
        return eval_spline(temperature, bnds, coeffs)

    def thermal_conductivity(self, temperature: DOFArray) -> DOFArray:
        r"""Return the gas thermal conductivity $\kappa_g$.

        .. math::
            \kappa = \frac{\mu C_p}{Pr}

        with gas viscosity $\mu$, heat capacity at constant pressure $C_p$
        and the Prandtl number $Pr$ (default to 1).
        """
        cp = self.heat_capacity(temperature)
        mu = self.viscosity(temperature)
        return mu*cp/self._prandtl

    def pressure(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Return the gas pressure.

        .. math::
            P = \frac{\epsilon_g \rho_g}{\epsilon_g} \frac{R}{M} T
        """
        gas_const = 8314.46261815324/self.molar_mass(temperature)
        return cv.mass*gas_const*temperature


class SolidProperties:
    """Evaluate the properties of the solid state containing resin and fibers.

    A linear weighting between the virgin and chared states is applied to
    yield the material properties. Polynomials were generated offline to avoid
    interpolation and they are not valid for temperatures above 3200K.

    .. automethod:: void_fraction
    .. automethod:: decomposition_progress
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: volume_fraction
    .. automethod:: permeability
    .. automethod:: emissivity
    .. automethod:: tortuosity
    """

    def __init__(self):
        """Solid volumetric density considering all resin constituents."""
        self._char_mass = 220.0
        self._virgin_mass = 280.0

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)

    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the pyrolysis is locally complete and only charred
        material exists:

        .. math::
            \tau = \frac{\rho_0}{\rho_0 - \rho_c}
                    \left( 1 - \frac{\rho_c}{\rho(t)} \right)
        """
        char_mass = self._char_mass
        virgin_mass = self._virgin_mass
        return virgin_mass/(virgin_mass - char_mass)*(1.0 - char_mass/mass)

    def enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Solid enthalpy as a function of pyrolysis progress."""
        virgin = (
            - 1.360688853105e-11*temperature**5 + 1.521029626150e-07*temperature**4
            - 6.733769958659e-04*temperature**3 + 1.497082282729e+00*temperature**2
            + 3.009865156984e+02*temperature - 1.062767983774e+06)

        char = (
            - 1.279887694729e-11*temperature**5 + 1.491175465285e-07*temperature**4
            - 6.994595296860e-04*temperature**3 + 1.691564018109e+00*temperature**2
            - 3.441837408320e+01*temperature - 1.235438104496e+05)

        return virgin*tau + char*(1.0 - tau)

    def heat_capacity(self, temperature: DOFArray,
                      tau: DOFArray) -> DOFArray:
        r"""Solid heat capacity $C_{p_s}$ as a function of pyrolysis progress."""
        actx = temperature.array_context

        virgin = actx.np.where(actx.np.less(temperature, 2222.0),
            + 4.122658916891e-14*temperature**5 - 4.430937180604e-10*temperature**4
            + 1.872060335623e-06*temperature**3 - 3.951464865603e-03*temperature**2
            + 4.291080938736e+00*temperature + 1.397594340362e+01,
            2008.8139143251735)

        char = (
            + 1.461303669323e-14*temperature**5 - 1.862489701581e-10*temperature**4
            + 9.685398830530e-07*temperature**3 - 2.599755262540e-03*temperature**2
            + 3.667295510844e+00*temperature - 7.816218435655e+01)

        return virgin*tau + char*(1.0 - tau)

    def thermal_conductivity(self, temperature: DOFArray,
                             tau: DOFArray) -> DOFArray:
        """Solid thermal conductivity as a function of pyrolysis progress."""
        virgin = (
            + 2.31290019732353e-17*temperature**5 - 2.167785032562e-13*temperature**4
            + 8.24498395180905e-10*temperature**3 - 1.221612456223e-06*temperature**2
            + 8.46459266618945e-04*temperature + 2.387112689755e-01)

        char = (
            - 7.378279908877e-18*temperature**5 + 4.709353498411e-14*temperature**4
            + 1.530236899258e-11*temperature**3 - 2.305611352452e-07*temperature**2
            + 3.668624886569e-04*temperature + 3.120898814888e-01)

        return virgin*tau + char*(1.0 - tau)

    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        fiber = 0.10
        virgin = 0.10
        char = 0.05
        return virgin*tau + char*(1.0 - tau) + fiber

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the composite material."""
        virgin = 1.6e-11
        char = 2.0e-11
        return virgin*tau + char*(1.0 - tau)

    def emissivity(self, tau: DOFArray) -> DOFArray:
        """Emissivity for energy radiation."""
        virgin = 0.8
        char = 0.9
        return virgin*tau + char*(1.0 - tau)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity."""
        virgin = 1.2
        char = 1.1
        return virgin*tau + char*(1.0 - tau)


class WallTabulatedEOS(WallEOS):
    """EOS for wall using tabulated data.

    Inherits WallEOS and add an temperature-evaluation function exclusive
    for TACOT-tabulated data.
    """

    def get_temperature(self, cv, material_densities, tseed, tau, eos, niter=3):
        r"""Evaluate the temperature based on solid+gas properties.

        It uses the assumption of thermal equilibrium between solid and fluid.
        Newton iteration is used to get the temperature based on the internal
        energy/enthalpy and heat capacity for the bulk (solid+gas) material:

        .. math::
            T^{n+1} = T^n -
                \frac
                {\epsilon_g \rho_g(h_g - R_g T^n) + \rho_s h_s - \rho e}
                {\epsilon_g \rho_g \left(
                    C_{p_g} - R_g\left[1 - \frac{\partial M}{\partial T} \right]
                \right)
                + \epsilon_s \rho_s C_{p_s}
                }

        Parameters
        ----------
        cv: ConservedVars

            The fluid conserved variables

        material_densities: np.ndarray

            The density of the different wall constituents

        tseed:

            Temperature to use as a seed for Netwon iteration

        tau: meshmode.dof_array.DOFArray

            Progress ratio of the phenolics decomposition

        eos: GasProperties

            The class containing the tabulated data for TACOT

        Returns
        -------
        temperature: meshmode.dof_array.DOFArray

            The temperature of the gas+solid

        """
        if isinstance(tseed, DOFArray) is False:
            temp = tseed + cv.mass*0.0
        else:
            temp = tseed*1.0

        rho_gas = cv.mass
        rho_solid = self.solid_density(material_densities)
        rhoe = cv.energy
        for _ in range(0, niter):

            # gas constant R/M
            molar_mass = eos.molar_mass(temp)
            gas_const = 8314.46261815324/molar_mass  # noqa N806

            eps_rho_e = (
                rho_gas*(eos.enthalpy(temp) - gas_const*temp)
                + rho_solid*self.enthalpy(temp, tau))

            bulk_cp = (
                rho_gas*(eos.heat_capacity(temp)
                         - gas_const*(1.0 - temp/molar_mass*eos.dMdT(temp)))
                + rho_solid*self.heat_capacity(temp, tau))

            temp = temp - (eps_rho_e - rhoe)/bulk_cp

        return temp

    def pressure_diffusivity(self, cv: ConservedVars, wdv: WallDependentVars,
                             viscosity: DOFArray) -> DOFArray:
        r"""Return the pressure diffusivity for Darcy flow.

        .. math::
            d_{P} = \epsilon_g \rho_g \frac{\mathbf{K}}{\mu \epsilon_g}

        where $\mu$ is the gas viscosity, $\epsilon_g$ is the void fraction
        and $\mathbf{K}$ is the permeability matrix.
        """
        return cv.mass*wdv.permeability/(viscosity*wdv.void_fraction)
