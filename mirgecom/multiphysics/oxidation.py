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

from dataclasses import dataclass, fields  # noqa

import numpy as np

from meshmode.dof_array import DOFArray

from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    # get_container_context_recursively
)
from mirgecom.multiphysics.wall_model import (
    WallDegradationModel, WallConservedVars
)


class OxidationWallModel(WallDegradationModel):
    """Oxidation model used for the Y3 oxidation model."""

    def __init__(self, solid_data):
        self._fiber = solid_data

    def eval_tau(self, wv: WallConservedVars) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        virgin = (self._fiber.intrinsic_density()
                  * self._fiber.solid_volume_fraction(tau=1.0))
        return 1.0 - (virgin - wv.mass)/virgin

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.
        """
        return 1.0 - self._fiber.solid_volume_fraction(tau)

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
        r"""."""
        return self._fiber.solid_tortuosity(tau)
