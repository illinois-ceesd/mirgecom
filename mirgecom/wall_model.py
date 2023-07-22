""":mod:`mirgecom.wall_model` handles the EOS for wall model.

.. autoclass:: SolidWallConservedVars
.. autoclass:: SolidWallDependentVars
.. autoclass:: SolidWallState
.. autoclass:: SolidWallModel
.. autoclass:: PorousWallVars
.. autoclass:: PorousWallProperties
.. autoclass:: PorousFlowModel
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


from dataclasses import dataclass
from abc import abstractmethod
from typing import Union
import numpy as np
from meshmode.dof_array import DOFArray
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)
from mirgecom.fluid import ConservedVars
from mirgecom.eos import GasEOS
from mirgecom.transport import TransportModel


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class SolidWallConservedVars:
    """Wall conserved variables for heat conduction only material."""

    mass: DOFArray
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`SolidWallConservedVars` object."""
        return get_container_context_recursively(self.mass)


@dataclass_array_container
@dataclass(frozen=True)
class SolidWallDependentVars:
    """Wall dependent variables for heat conduction only materials."""

    thermal_conductivity: DOFArray
    temperature: DOFArray


@dataclass_array_container
@dataclass(frozen=True)
class SolidWallState:
    """Wall state for heat conduction only materials."""

    cv: SolidWallConservedVars
    dv: SolidWallDependentVars


class SolidWallModel:
    """Model for calculating wall quantities for heat conduction only materials.

    .. automethod:: density
    .. automethod:: heat_capacity
    .. automethod:: enthalpy
    .. automethod:: thermal_diffusivity
    .. automethod:: thermal_conductivity
    .. automethod:: get_temperature
    """

    def __init__(self, density_func, enthalpy_func, heat_capacity_func,
                 thermal_conductivity_func):
        self._density_func = density_func
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def density(self):
        """Return the wall density for all components."""
        return self._density_func()

    def heat_capacity(self, temperature=None):
        """Return the wall heat_capacity for all components."""
        return self._heat_capacity_func(temperature)

    def enthalpy(self, temperature):
        """Return the wall enthalpy for all components."""
        return self._enthalpy_func(temperature)

    def thermal_diffusivity(self, mass, temperature,
                            thermal_conductivity=None):
        """Return the wall thermal diffusivity for all components."""
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(temperature)
        return thermal_conductivity/(mass * self.heat_capacity(temperature))

    def thermal_conductivity(self, temperature):
        """Return the wall thermal conductivity for all components."""
        return self._thermal_conductivity_func(temperature)

    def get_temperature(self, wv, tseed=None):
        """Evaluate the temperature based on the energy."""
        if tseed is not None:
            temp = tseed*1.0
            for _ in range(0, 3):
                h = self.enthalpy(temp)
                cp = self.heat_capacity(temp)
                temp = temp - (h - wv.energy/wv.mass)/cp
            return temp

        return wv.energy/(self.density()*self.heat_capacity())

    def dependent_vars(self, wv, tseed=None):
        """Return solid wall dependent variables."""
        temperature = self.get_temperature(wv, tseed)
        kappa = self.thermal_conductivity(temperature)
        return SolidWallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)


class PorousWallProperties:
    """Abstract interface for porous media domains.

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

    @abstractmethod
    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ filled by gas around the solid."""
        raise NotImplementedError()

    @abstractmethod
    def enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Evaluate the enthalpy $h_s$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def heat_capacity(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def thermal_conductivity(self, temperature: DOFArray,
                             tau: DOFArray) -> DOFArray:
        r"""Evaluate the thermal conductivity $\kappa$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        raise NotImplementedError()

    @abstractmethod
    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        raise NotImplementedError()

    @abstractmethod
    def emissivity(self, tau: DOFArray) -> DOFArray:
        """Emissivity for energy radiation."""
        raise NotImplementedError()

    @abstractmethod
    def tortuosity(self, tau: DOFArray) -> DOFArray:
        """Tortuosity of the porous material."""
        raise NotImplementedError()

    @abstractmethod
    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition."""
        raise NotImplementedError()


@dataclass_array_container
@dataclass(frozen=True, eq=False)
class PorousWallVars:
    """Variables for the (porous) fluid state.

    .. attribute:: material_densities
    .. attribute:: tau
    .. attribute:: void_fraction
    .. attribute:: emissivity
    .. attribute:: permeability
    .. attribute:: tortuosity
    .. attribute:: density
    """

    material_densities: Union[DOFArray, np.ndarray]
    tau: DOFArray
    void_fraction: DOFArray
    emissivity: DOFArray
    permeability: DOFArray
    tortuosity: DOFArray
    density: DOFArray


class PorousFlowEOS:
    """Abstract interface to equation of porous flow class."""

    @abstractmethod
    def decomposition_progress(self, material_densities) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition."""
        raise NotImplementedError()

    @abstractmethod
    def solid_density(self, material_densities) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$."""
        raise NotImplementedError()

    @abstractmethod
    def get_temperature(self, cv: ConservedVars, wv: PorousWallVars,
                        tseed: DOFArray, niter=3) -> DOFArray:
        r"""Evaluate the temperature based on solid+gas properties."""
        raise NotImplementedError()

    @abstractmethod
    def get_pressure(self, cv: ConservedVars, wv: PorousWallVars,
                     temperature: DOFArray) -> DOFArray:
        """Return the pressure of the gas considering the void fraction."""
        raise NotImplementedError()

    @abstractmethod
    def internal_energy(self, cv: ConservedVars, wv: PorousWallVars,
                 temperature: DOFArray) -> DOFArray:
        """Return the enthalpy of the gas+solid material."""
        raise NotImplementedError()

    @abstractmethod
    def heat_capacity(self, cv: ConservedVars, wv: PorousWallVars,
                 temperature: DOFArray) -> DOFArray:
        """Return the heat capacity of the gas+solid material."""
        raise NotImplementedError()


@dataclass(frozen=True)
class PorousFlowModel(PorousFlowEOS):
    """
    <A very important description>...

    .. attribute:: wall_model
    .. attribute:: eos
    .. attribute:: transport

    .. automethod:: solid_density
    .. automethod:: get_temperature
    .. automethod:: get_pressure
    .. automethod:: internal_energy
    .. automethod:: heat_capacity
    .. automethod:: decomposition_progress
    """

    eos: GasEOS
    transport: TransportModel
    wall_model: PorousWallProperties

    def decomposition_progress(self, material_densities) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(material_densities)
        return self.wall_model.decomposition_progress(mass)

    def solid_density(self, material_densities) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$.

        The material density is relative to the entire control volume, and
        is not to be confused with the intrinsic density, hence the $\epsilon$
        dependence. It is computed as the sum of all N solid phases:

        .. math::
            \epsilon_s \rho_s = \sum_i^N \epsilon_i \rho_i
        """
        if isinstance(material_densities, DOFArray):
            return material_densities
        return sum(material_densities)

    def get_temperature(self, cv: ConservedVars, wv: PorousWallVars,
                        tseed: DOFArray, niter=3) -> DOFArray:
        r"""Evaluate the temperature based on solid+gas properties.

        It uses the assumption of thermal equilibrium between solid and fluid.
        Newton iteration is used to get the temperature based on the internal
        energy/enthalpy and heat capacity for the bulk (solid+gas) material:

        .. math::
            T^{n+1} = T^n -
                \frac
                {\epsilon_g \rho_g e_g + \rho_s h_s - \rho e}
                {\epsilon_g \rho_g C_{p_g} + \epsilon_s \rho_s C_{p_s}}

        """
        if isinstance(tseed, DOFArray) is False:
            actx = cv.array_context
            temp = tseed + actx.np.zeros_like(wv.tau)
        else:
            temp = tseed*1.0

        internal_energy = cv.energy - 0.5/cv.mass*np.dot(cv.momentum, cv.momentum)

        for _ in range(0, niter):
            eps_rho_e = self.internal_energy(cv, wv, temp)
            bulk_cp = self.heat_capacity(cv, wv, temp)
            temp = temp - (eps_rho_e - internal_energy)/bulk_cp

        return temp

    def get_pressure(self, cv: ConservedVars, wv: PorousWallVars,
                     temperature: DOFArray) -> DOFArray:
        """Return the pressure of the gas considering the void fraction."""
        return self.eos.pressure(cv, temperature)/wv.void_fraction

    def internal_energy(self, cv: ConservedVars, wv: PorousWallVars,
                 temperature: DOFArray) -> DOFArray:
        """Return the enthalpy of the gas+solid material."""
        return (cv.mass*self.eos.get_internal_energy(temperature,
                                                cv.species_mass_fractions)
                + wv.density*self.wall_model.enthalpy(temperature, wv.tau))

    def heat_capacity(self, cv: ConservedVars, wv: PorousWallVars,
                 temperature: DOFArray) -> DOFArray:
        """Return the heat capacity of the gas+solid material."""
        return (cv.mass*self.eos.heat_capacity_cv(cv, temperature)
                + wv.density*self.wall_model.heat_capacity(temperature, wv.tau))
