""":mod:`mirgecom.multiphysics.oxidation` handles Y3 oxidation model."""

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

from meshmode.dof_array import DOFArray

from mirgecom.multiphysics.wall_model import (
    WallDegradationModel, WallConservedVars
)


class OxidationWallModel(WallDegradationModel):
    """Oxidation model used for the Y3 oxidation model.

    .. automethod:: decomposition_progress
    .. automethod:: void_fraction
    .. automethod:: solid_density
    .. automethod:: solid_thermal_conductivity
    .. automethod:: solid_enthalpy
    .. automethod:: solid_permeability
    .. automethod:: solid_emissivity
    .. automethod:: solid_heat_capacity
    .. automethod:: solid_tortuosity
    """

    def __init__(self, solid_data):
        self._fiber = solid_data

    def decomposition_progress(self, current_mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        return self._solid_data.solid_decomposition_progress(current_mass)

    def intrinsic_density(self):
        r"""Return the intrinsic density of carbon."""
        return self._fiber.intrinsic_density()

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.
        """
        return 1.0 - self._fiber.solid_volume_fraction(tau)

    def solid_density(self, wv: WallConservedVars) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$.

        The material density is relative to the entire control volume, and
        is not to be confused with the intrinsic density, hence the $\epsilon$
        dependence. For carbon fiber, it has a single constituent.
        """
        return wv.mass

    def solid_thermal_conductivity(self, temperature, tau) -> DOFArray:
        r"""Return the solid thermal conductivity, $f(\rho, \tau, T)$."""
        return self._fiber.solid_thermal_conductivity(temperature, tau)

    def solid_permeability(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall permeability $K$ based on the progress ratio $\tau$."""
        return self._solid_data.solid_permeability(tau)

    def solid_enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Return the solid enthalpy $h_s$."""
        return self._fiber.solid_enthalpy(temperature, tau)

    def solid_emissivity(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall emissivity based on the progress ratio $\tau$."""
        return self._fiber.solid_emissivity(tau)

    def solid_heat_capacity(self, temperature: DOFArray,
                               tau: DOFArray) -> DOFArray:
        r"""Return the solid heat capacity $C_{p_s}$."""
        return self._fiber.solid_heat_capacity(temperature, tau)

    def solid_tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Return the solid tortuosity $\eta$."""
        return self._fiber.solid_tortuosity(tau)
