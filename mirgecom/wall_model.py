""":mod:`mirgecom.wall_model` handles the EOS for wall model.

.. autoclass:: SolidWallConservedVars
.. autoclass:: SolidWallDependentVars
.. autoclass:: SolidWallState
.. autoclass:: SolidWallEOS
.. autoclass:: PorousFlowDependentVars
.. autoclass:: PorousWallDependentVars
.. autoclass:: PorousWallDegradationModel
.. autoclass:: PorousWallEOS
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
    .. attribute:: density
    """

    tau: DOFArray
    void_fraction: DOFArray
    emissivity: DOFArray
    permeability: DOFArray
    density: DOFArray


class PorousWallDegradationModel:
    """Abstract interface for wall degradation model.

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

    @abstractmethod
    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ filled by gas around the solid."""
        raise NotImplementedError()

    @abstractmethod
    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition."""
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


class PorousWallEOS:
    """Interface for porous material degradation models.

    This class evaluates the variables dependent on the wall decomposition
    state and its different parts. For that, the user must supply external
    functions in order to consider the different materials employed at wall
    (for instance, carbon fibers, graphite, alumina, steel etc). The number
    and types of the materials is case-dependent and, hence, must be defined
    in the respective simulation driver.

    .. automethod:: __init__
    .. automethod:: solid_density
    .. automethod:: decomposition_progress
    .. automethod:: void_fraction
    .. automethod:: get_temperature
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    .. automethod:: dependent_vars
    """

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

    def decomposition_progress(self, material_densities) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(material_densities)
        return self._material.decomposition_progress(mass)

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ of the sample filled with gas."""
        return self._material.void_fraction(tau)

    def get_temperature(self, cv: ConservedVars,
                        material_densities: Union[DOFArray, np.ndarray],
                        tseed: DOFArray, tau: DOFArray, gas_model,
                        niter=3) -> DOFArray:
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
        actx = cv.array_context

        if isinstance(tseed, DOFArray) is False:
            temp = tseed + actx.np.zeros_like(tau)
        else:
            temp = tseed*1.0

        rho_gas = cv.mass
        rho_solid = self.solid_density(material_densities)

        rhoe = cv.energy - 0.5/cv.mass*np.dot(cv.momentum, cv.momentum)

        for _ in range(0, niter):

            gas_internal_energy = \
                gas_model.eos.get_internal_energy(temp, cv.species_mass_fractions)

            gas_heat_capacity_cv = gas_model.eos.heat_capacity_cv(cv, temp)

            eps_rho_e = rho_gas*gas_internal_energy \
                      + rho_solid*self.enthalpy(temp, tau)

            bulk_cp = rho_gas*gas_heat_capacity_cv \
                      + rho_solid*self.heat_capacity(temp, tau)

            temp = temp - (eps_rho_e - rhoe)/bulk_cp

        return temp

    def enthalpy(self, temperature: DOFArray, tau: DOFArray):
        """Return the enthalpy of the test material as a function of temperature."""
        return self._material.enthalpy(temperature, tau)

    def heat_capacity(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Return the heat capacity of the test material."""
        return self._material.heat_capacity(temperature, tau)

    # TODO: maybe create a WallTransportVars?
    def viscosity(self, temperature: DOFArray, tau: DOFArray,
                  gas_tv: GasTransportVars) -> DOFArray:
        """Viscosity of the gas through the (porous) wall."""
        epsilon = self._material.void_fraction(tau)
        return gas_tv.viscosity/epsilon

    # TODO: maybe create a WallTransportVars?
    def thermal_conductivity(self, cv: ConservedVars,
                             material_densities: Union[DOFArray, np.ndarray],
                             temperature: DOFArray, tau: DOFArray,
                             gas_tv: GasTransportVars):
        r"""Return the effective thermal conductivity of the gas+solid.

        It is a function of temperature and degradation progress. As the
        fibers are oxidized, they reduce their cross area and, consequently,
        their ability to conduct heat.

        It is evaluated using a mass-weighted average given by

        .. math::
            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}

        Returns
        -------
        thermal_conductivity: meshmode.dof_array.DOFArray
            the thermal_conductivity, including all parts of the solid
        """
        y_g = cv.mass/(cv.mass + self.solid_density(material_densities))
        y_s = 1.0 - y_g
        kappa_s = self._material.thermal_conductivity(temperature, tau)
        kappa_g = gas_tv.thermal_conductivity

        return y_s*kappa_s + y_g*kappa_g

    # TODO: maybe create a WallTransportVars?
    def species_diffusivity(self, temperature: DOFArray, tau: DOFArray,
                            gas_tv: GasTransportVars):
        """Mass diffusivity of gaseous species through the (porous) wall.

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        tortuosity = self._material.tortuosity(tau)
        return gas_tv.species_diffusivity/tortuosity

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
            density=self.solid_density(material_densities)
        )
