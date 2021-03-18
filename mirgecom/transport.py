"""
:mod:`mirgecom.transport` provides methods/utils for tranport properties.

Transport Models
^^^^^^^^^^^^^^^^
This module is designed provide Transport Model objects used to compute and
manage the transport properties in viscous flows.

.. autoclass:: TransportDependentVars
.. autoclass:: TransportModel
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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

from dataclasses import dataclass
import numpy as np
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import ConservedVars
from mirgecom.eos import GasEOS


@dataclass
class TransportDependentVars:
    """State-dependent quantities for :class:`TransportModel`.

    Prefer individual methods for model use, use this
    structure for visualization or probing.

    .. attribute:: bulk_viscosity
    .. attribute:: viscosity
    .. attribute:: thermal_conductivity
    .. attribute:: species_diffusivity
    """

    bulk_viscosity: float
    viscosity: np.ndarray
    thermal_conductivity: np.ndarray
    species_diffusivity: np.ndarray


class TransportModel:
    r"""Abstract interface to thermo-diffusive transport model class.

    Transport model classes are responsible for
    computing relations between fluid or gas state variables and
    thermo-diffusive transport properties for those fluids.

    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    """

    def bulk_viscosity(self):
        r"""Get the bulk viscosity for the gas ($\mu_{B})."""
        raise NotImplementedError()

    def viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas dynamic viscosity, $\mu$."""
        raise NotImplementedError()

    def thermal_conductivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas thermal_conductivity, $\kappa$."""
        raise NotImplementedError()

    def species_diffusivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the vector of species diffusivities ($D_{\alpha}$)."""
        raise NotImplementedError()


class SimpleTransport(TransportModel):
    r"""Transport model with uniform, constant properties.

    .. automethod:: __init__

    Inherits from (and implements) :class:`TransportModel`.
    """

    def __init__(self, bulk_viscosity=0, viscosity=0,
                 thermal_conductivity=0,
                 species_diffusivity=np.empty((0,), dtype=object)):
        """Initialize uniform, constant transport properties."""
        self._mu_bulk = bulk_viscosity
        self._mu = viscosity
        self._kappa = thermal_conductivity
        self._d_alpha = species_diffusivity

    def bulk_viscosity(self):
        r"""Get the bulk viscosity for the gas, $\mu_{B}."""
        return self._mu_bulk

    def viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas dynamic viscosity, $\mu$."""
        actx = cv.mass.array_context
        ones = actx.ones_like(cv.mass)
        return self._mu * ones

    def thermal_conductivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas thermal_conductivity, $\kappa$."""
        actx = cv.mass.array_context
        ones = actx.ones_like(cv.mass)
        return self._kappa * ones

    def species_diffusivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the vector of species diffusivities, $D_{\alpha}$."""
        actx = cv.mass.array_context
        ones = actx.ones_like(cv.mass)
        nspecies = len(cv.species_mass)
        assert nspecies == len(self._d_alpha)
        return self._d_alpha * ones
