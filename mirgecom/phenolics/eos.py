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

from typing import Union, Optional
from dataclasses import dataclass
import numpy as np
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from mirgecom.fluid import ConservedVars, make_conserved
from abc import ABCMeta, abstractmethod
from arraycontext import dataclass_array_container


@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsDependentVars:
    """State-dependent quantities for :class:`GasEOS`.

    Prefer individual methods for model use, use this
    structure for visualization or probing.

    .. attribute:: temperature
    .. attribute:: pressure
    .. attribute:: velocity
    """

    temperature: DOFArray
    pressure: DOFArray
    velocity: DOFArray
    progress: DOFArray

    #XXX why are these nd.array and not DOFArray?
    viscosity: np.ndarray
    thermal_conductivity: np.ndarray
    species_diffusivity: np.ndarray


class PhenolicsEOS(metaclass=ABCMeta):
    r"""Abstract interface to equation of state class.

    .. automethod:: 
    """

    @abstractmethod
    def pressure(self, cv: ConservedVars, temperature: DOFArray):
        """Get the gas pressure."""

    @abstractmethod
    def temperature(self, cv: ConservedVars,
                    temperature_seed: Optional[DOFArray] = None) -> DOFArray:
        """Get the gas temperature."""

    @abstractmethod
    def gas_const(self, cv: ConservedVars):
        r"""Get the specific gas constant ($R_s$)."""

    @abstractmethod
    def heat_capacity_cp(self, cv: ConservedVars, temperature: DOFArray):
        r"""Get the specific heat capacity at constant pressure ($C_p$)."""

    @abstractmethod
    def viscosity(self, cv: ConservedVars) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""

    @abstractmethod
    def thermal_conductivity(self, cv: ConservedVars) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""

    @abstractmethod
    def species_diffusivity(self, cv: ConservedVars) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""

    @abstractmethod
    def emissivity(self, cv: ConservedVars, temperature: DOFArray):
        r"""."""

    @abstractmethod
    def permeability(self, cv: ConservedVars, temperature: DOFArray):
        r"""."""

    @abstractmethod
    def progress(self, cv: ConservedVars, temperature: DOFArray):
        r"""."""

    #FIXME
    def dependent_vars(self, cv: ConservedVars,
            temperature_seed: Optional[DOFArray] = None) -> PhenolicsDependentVars:
        """Get an agglomerated array of the dependent variables."""
        temperature = self.temperature(cv, temperature_seed)
        return PhenolicsDependentVars(
            progress=
            temperature=temperature,
            pressure=self.pressure(cv, temperature),
            viscosity=
            thermal_conductivity=
            species_diffusivity=
            emissivity=
            permeability=
        )


class SimpleEquationOfState(PhenolicsEOS):

    def __init__(self, composite):
        self._degradation_model = composite

    def heat_capacity_cp(self, cv: Optional[ConservedVars] = None, temperature=None):
        return 

    def gas_const(self, cv: Optional[ConservedVars] = None):
        """Get specific gas constant R."""
        return 

    def pressure(self, cv: ConservedVars, temperature=None):
        return 


    
