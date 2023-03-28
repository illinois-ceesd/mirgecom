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
from dataclasses import dataclass
from arraycontext import dataclass_array_container
import numpy as np
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from mirgecom.fluid import ConservedVars
from mirgecom.eos import GasEOS, GasDependentVars


@dataclass_array_container
@dataclass(frozen=True)
class WallTransportVars:
    """."""

    viscosity: np.ndarray
    thermal_conductivity: np.ndarray
    species_diffusivity: np.ndarray


class WallTransportModel:
    r""".

    .. automethod:: viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    .. automethod:: transport_vars
    """


    def viscosity(self, cv: ConservedVars,
                  dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        raise NotImplementedError()

    def thermal_conductivity(self, cv: ConservedVars,
                             dv: Optional[GasDependentVars] = None,
                             eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""
        raise NotImplementedError()

    def species_diffusivity(self, cv: ConservedVars,
                            dv: Optional[GasDependentVars] = None,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        raise NotImplementedError()

    def transport_vars(self, cv: ConservedVars,
                       dv: Optional[GasDependentVars] = None,
                       eos: Optional[GasEOS] = None) -> GasTransportVars:
        r"""Compute the transport properties from the conserved state."""
        return GasTransportVars(
            viscosity=self.viscosity(cv=cv, dv=dv),
            thermal_conductivity=self.thermal_conductivity(cv=cv, dv=dv, eos=eos),
            species_diffusivity=self.species_diffusivity(cv=cv, dv=dv, eos=eos)
        )


class SimpleTransport(TransportModel):
    r"""Transport model with uniform, constant properties.

    Inherits from (and implements) :class:`TransportModel`.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    def __init__(self, viscosity=0, thermal_conductivity=0,
                 species_diffusivity=None):
        self._mu = viscosity
        self._kappa = thermal_conductivity
        self._d_alpha = species_diffusivity

    def viscosity(self, cv: ConservedVars,
                  dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        return self._mu*(0*cv.mass + 1.0)

    def thermal_conductivity(self, cv: ConservedVars,
                             dv: Optional[GasDependentVars] = None,
                             eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""
        return self._kappa*(0*cv.mass + 1.0)

    def species_diffusivity(self, cv: ConservedVars,
                            dv: Optional[GasDependentVars] = None,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        return self._d_alpha*(0*cv.mass + 1.0)
