""":mod:`mirgecom.wall_model` handles the EOS for wall model.

.. autoclass:: SolidWallConservedVars
.. autoclass:: SolidWallDependentVars
.. autoclass:: SolidWallState
.. autoclass:: SolidWallEOS
.. autoclass:: PorousFlowDependentVars
.. autoclass:: PorousWallDependentVars
.. autoclass:: PorousWallProperties
.. autoclass:: PorousFlowEOS
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
from mirgecom.eos import MixtureDependentVars
from mirgecom.transport import GasTransportVars
from mirgecom.eos import GasEOS


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


class SolidWallEOS:
    """Model for calculating wall quantities for heat conduction only materials."""

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

    def eval_temperature(self, wv, tseed=None):
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
        temperature = self.eval_temperature(wv, tseed)
        kappa = self.thermal_conductivity(temperature)
        return SolidWallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)


@dataclass_array_container
@dataclass(frozen=True, eq=False)
# FIXME in practice, the density of the wall materials follow a conservation
# equation. To avoid big changes in the code atm, it is squeezed as a
# dependent variable for "state.dv", although it is not.
# TODO if we decide to refactor it, modify this accordingly
class PorousFlowDependentVars(MixtureDependentVars):
    """Dependent variables for the (porous) fluid state.

    .. attribute:: material_densities
    """

    material_densities: Union[DOFArray, np.ndarray]


@dataclass_array_container
@dataclass(frozen=True, eq=False)
class PorousWallDependentVars:
    """Dependent variables for the wall degradation model.

    .. attribute:: tau
    .. attribute:: void_fraction
    .. attribute:: emissivity
    .. attribute:: permeability
    .. attribute:: tortuosity
    .. attribute:: density
    """

    tau: DOFArray
    void_fraction: DOFArray
    emissivity: DOFArray
    permeability: DOFArray
    tortuosity: DOFArray
    density: DOFArray


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


class PorousFlowEOS:
    """EOS for porous media flows, including degradation models.

    This class evaluates the variables dependent on the wall decomposition
    state and the gas permeating the material.

    .. automethod:: __init__
    .. automethod:: solid_density
    .. automethod:: void_fraction
    .. automethod:: get_temperature
    .. automethod:: get_pressure
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    .. automethod:: decomposition_progress
    .. automethod:: dependent_vars
    """

    # TODO create a class that encapsulate the `PorousWallProperties` and
    # `GasModel` ???
    def __init__(self, wall_material):
        """
        Initialize the model.

        Parameters
        ----------
        wall_material:
            The class with the solid properties of the desired material.
        """
        self._material = wall_material

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

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ of the sample filled with gas."""
        return self._material.void_fraction(tau)

    def get_temperature(self, cv: ConservedVars, wdv: PorousWallDependentVars,
                        tseed: DOFArray, eos: GasEOS, niter=3) -> DOFArray:
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
            temp = tseed + actx.np.zeros_like(wdv.tau)
        else:
            temp = tseed*1.0

        internal_energy = cv.energy - 0.5/cv.mass*np.dot(cv.momentum, cv.momentum)

        for _ in range(0, niter):

            gas_internal_energy = \
                eos.get_internal_energy(temp, cv.species_mass_fractions)

            gas_heat_capacity_cv = eos.heat_capacity_cv(cv, temp)

            eps_rho_e = cv.mass*gas_internal_energy \
                      + wdv.density*self._material.enthalpy(temp, wdv.tau)

            bulk_cp = cv.mass*gas_heat_capacity_cv \
                      + wdv.density*self._material.heat_capacity(temp, wdv.tau)

            temp = temp - (eps_rho_e - internal_energy)/bulk_cp

        return temp

    def get_pressure(self, cv: ConservedVars, wdv: PorousWallDependentVars,
                     temperature: DOFArray, eos: GasEOS) -> DOFArray:
        """Return the pressure of the gas considering the void fraction."""
        return eos.pressure(cv, temperature)/wdv.void_fraction

    # make it return the enthalpy of gas+solid?
    def enthalpy(self, temperature: DOFArray, tau: DOFArray):
        """Return the enthalpy of the test material."""
        return self._material.enthalpy(temperature, tau)

    # make it return the heat capacity of gas+solid?
    def heat_capacity(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Return the heat capacity of the test material."""
        return self._material.heat_capacity(temperature, tau)

    # TODO: maybe create a WallTransportVars?
    def viscosity(self, wdv: PorousWallDependentVars, temperature: DOFArray,
                  gas_tv: GasTransportVars) -> DOFArray:
        """Viscosity of the gas through the (porous) wall."""
        return gas_tv.viscosity/wdv.void_fraction

    # TODO: maybe create a WallTransportVars?
    def thermal_conductivity(self, cv: ConservedVars,
                             wdv: PorousWallDependentVars,
                             temperature: DOFArray,
                             gas_tv: GasTransportVars) -> DOFArray:
        r"""Return the effective thermal conductivity of the gas+solid.

        It is a function of temperature and degradation progress. As the
        fibers are oxidized, they reduce their cross area and, consequently,
        their ability to conduct heat.

        It is evaluated using a mass-weighted average given by

        .. math::
            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}
        """
        y_g = cv.mass/(cv.mass + wdv.density)
        y_s = 1.0 - y_g
        kappa_s = self._material.thermal_conductivity(temperature, wdv.tau)
        kappa_g = gas_tv.thermal_conductivity

        return y_s*kappa_s + y_g*kappa_g

    # TODO: maybe create a WallTransportVars?
    def species_diffusivity(self, wdv: PorousWallDependentVars,
            temperature: DOFArray, gas_tv: GasTransportVars) -> DOFArray:
        """Mass diffusivity of gaseous species through the (porous) wall."""
        return gas_tv.species_diffusivity/wdv.tortuosity

    def decomposition_progress(self, material_densities) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(material_densities)
        return self._material.decomposition_progress(mass)

    def dependent_vars(self, material_densities: Union[DOFArray, np.ndarray]):
        """Get the state-dependent variables.

        Returns
        -------
        dependent_vars: :class:`PorousWallDependentVars`
            dependent variables of the wall-only state. These are complementary
            to the fluid dependent variables, but not stacked together.
        """
        tau = self.decomposition_progress(material_densities)
        return PorousWallDependentVars(
            tau=tau,
            void_fraction=self.void_fraction(tau),
            emissivity=self._material.emissivity(tau),
            permeability=self._material.permeability(tau),
            tortuosity=self._material.tortuosity(tau),
            density=self.solid_density(material_densities)
        )
