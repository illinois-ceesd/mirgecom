"""
:mod:`mirgecom.transport` provides methods/utils for tranport properties.

Transport Models
^^^^^^^^^^^^^^^^
This module is designed provide Transport Model objects used to compute and
manage the transport properties in viscous flows.

.. autoclass:: TransportDependentVars
.. autoclass:: TransportModel
.. autoclass:: SimpleTransport
.. autoclass:: PowerLawTransport
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

    bulk_viscosity: np.ndarray
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

    def bulk_viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the bulk viscosity for the gas (${\mu}_{B}$)."""
        raise NotImplementedError()

    def viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas dynamic viscosity, $\mu$."""
        raise NotImplementedError()

    def thermal_conductivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas thermal_conductivity, $\kappa$."""
        raise NotImplementedError()

    def species_diffusivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the vector of species diffusivities (${d}_{\alpha}$)."""
        raise NotImplementedError()


class SimpleTransport(TransportModel):
    r"""Transport model with uniform, constant properties.

    .. automethod:: __init__

    Inherits from (and implements) :class:`TransportModel`.
    """

    def __init__(self, bulk_viscosity=0, viscosity=0,
                 thermal_conductivity=0,
                 species_diffusivity=None):
        """Initialize uniform, constant transport properties."""
        if species_diffusivity is None:
            species_diffusivity = np.empty((0,), dtype=object)
        self._mu_bulk = bulk_viscosity
        self._mu = viscosity
        self._kappa = thermal_conductivity
        self._d_alpha = species_diffusivity

    def _make_array(self, something, cv):
        """Make an appropriate shaped array from the constant properties."""
        return something * cv.mass / cv.mass

    def bulk_viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the bulk viscosity for the gas, $\mu_{B}."""
        return self._make_array(self._mu_bulk, cv)

    def viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas dynamic viscosity, $\mu$."""
        return self._make_array(self._mu, cv)

    def thermal_conductivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas thermal_conductivity, $\kappa$."""
        return self._make_array(self._kappa, cv)

    def species_diffusivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        nspecies = len(cv.species_mass)
        assert nspecies == len(self._d_alpha)
        return self._make_array(self._d_alpha, cv)


class PowerLawTransport(TransportModel):
    r"""Transport model with simple power law properties.

    .. automethod:: __init__

    Inherits from (and implements) :class:`TransportModel`.
    """

    # air-like defaults here
    def __init__(self, alpha=0.6, beta=4.093e-7, sigma=2.5, n=.666,
                 species_diffusivity=None):
        """Initialize power law coefficients and parameters."""
        if species_diffusivity is None:
            species_diffusivity = np.empty((0,), dtype=object)
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        self._n = n
        self._d_alpha = species_diffusivity

    def _make_array(self, something, cv):
        """Make an appropriate shaped array from the constant properties."""
        return something * cv.mass / cv.mass

    def bulk_viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the bulk viscosity for the gas, $\mu_{B}.

        $\mu_{B} = \alpha\mu$
        """
        return self._alpha * self.viscosity(eos, cv)

    def viscosity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas dynamic viscosity, $\mu$.

        $\mu = \beta{T}^n$
        """
        actx = cv.array_context
        gas_t = eos.temperature(cv)
        return self._beta * actx.np.pow(gas_t, self._n)

    def thermal_conductivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the gas thermal_conductivity, $\kappa$.

        $\kappa = \sigma\mu{C}_{v}$
        """
        return self._sigma * self.viscosity(eos, cv) * eos.heat_capacity_cv(cv)

    def species_diffusivity(self, eos: GasEOS, cv: ConservedVars):
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        nspecies = len(cv.species_mass)
        assert nspecies == len(self._d_alpha)
        return self._make_array(self._d_alpha, cv)
