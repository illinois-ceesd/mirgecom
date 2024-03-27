""":mod:`mirgecom.materials.prescribed_porous_material` for user-defined material.

The user can create any material with driver-oriented functions.

.. autoclass:: PrescribedMaterialEOS
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

from typing import Optional
from meshmode.dof_array import DOFArray
from mirgecom.wall_model import PorousWallEOS


class PrescribedMaterialEOS(PorousWallEOS):
    """Evaluate the properties of a user-defined material.

    .. automethod:: void_fraction
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: volume_fraction
    .. automethod:: permeability
    .. automethod:: emissivity
    .. automethod:: tortuosity
    .. automethod:: decomposition_progress
    """

    def __init__(self, enthalpy_func, heat_capacity_func, thermal_conductivity_func,
                 volume_fraction_func, permeability_func, emissivity_func,
                 tortuosity_func, decomposition_progress_func):

        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func
        self._volume_fraction_func = volume_fraction_func
        self._permeability_func = permeability_func
        self._emissivity_func = emissivity_func
        self._tortuosity_func = tortuosity_func
        self._decomposition_progress_func = decomposition_progress_func

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the oxidation
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)

    def enthalpy(self, temperature: DOFArray, tau: Optional[DOFArray]) -> DOFArray:
        r"""Evaluate the solid enthalpy $h_s$."""
        return self._enthalpy_func(temperature)

    def heat_capacity(self, temperature: DOFArray,
                      tau: Optional[DOFArray]) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$."""
        return self._heat_capacity_func(temperature)

    def thermal_conductivity(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Evaluate the thermal conductivity $\kappa$"""
        return self._thermal_conductivity_func(temperature)

    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        actx = tau.array_context
        return self._volume_fraction_func(tau) + actx.np.zeros_like(tau)

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        actx = tau.array_context
        return self._permeability_func(tau) + actx.np.zeros_like(tau)

    def emissivity(self, temperature=None, tau=None) -> DOFArray:
        """Emissivity for energy radiation."""
        actx = tau.array_context
        return self._emissivity_func(tau) + actx.np.zeros_like(tau)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity."""
        actx = tau.array_context
        return self._tortuosity_func(tau) + actx.np.zeros_like(tau)

    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$."""
        return self._decomposition_progress_func(mass)
