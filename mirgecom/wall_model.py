""":mod:`mirgecom.multiphysics.wall_model` handles the EOS for wall model.

.. autoclass:: WallEOS
.. autoclass:: WallDependentVars
.. autoclass:: WallDegradationModel
"""

from dataclasses import dataclass

from abc import abstractmethod
import numpy as np

from typing import Union

from meshmode.dof_array import DOFArray

from arraycontext import (
    dataclass_array_container,
    # with_container_arithmetic,
    # get_container_context_recursively
)
from mirgecom.fluid import ConservedVars
from mirgecom.eos import MixtureDependentVars
from mirgecom.transport import GasTransportVars


@dataclass_array_container
@dataclass(frozen=True)
class PorousFlowDependentVars(MixtureDependentVars):
    """Dependent variables for the wall model.

    .. attribute:: wall_density
    """

    wall_density: Union[DOFArray, np.ndarray]
    wall_sample_mask: Union[DOFArray, np.ndarray]


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    """Dependent variables for the wall degradation model.

    .. attribute:: tau
    .. attribute:: void_fraction
    .. attribute:: emissivity
    .. attribute:: permeability
    """

    tau: DOFArray
    void_fraction: DOFArray
    emissivity: DOFArray
    permeability: DOFArray


class WallDegradationModel:
    """Abstract interface for wall degradation model."""

    @abstractmethod
    def intrinsic_density(self):
        r"""Return the intrinsic density $\rho$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def void_fraction(self, tau: DOFArray):
        r"""Void fraction $\epsilon$ filled by gas around the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_density(self, wall_density: Union[DOFArray, np.ndarray]) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$."""
        raise NotImplementedError()

    @abstractmethod
    def heat_capacity(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Evaluate the enthalpy $h_s$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def thermal_conductivity(self, temperature: DOFArray, tau: DOFArray):
        r"""Evaluate the thermal conductivity $\kappa$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def emissivity(self, tau: DOFArray):
        """Emissivity for energy radiation."""
        raise NotImplementedError()

    @abstractmethod
    def tortuosity(self, tau: DOFArray):
        """Tortuosity of the porous material."""
        raise NotImplementedError()


class WallEOS:
    """Interface for the wall model used for the Y3 prediction.

    This class evaluates the variables dependent on the wall decomposition state
    and its different parts. For that, the user must supply external functions
    in order to consider the different materials employed at wall (for instance,
    carbon fibers, graphite, alumina, steel etc). The number and types of the
    materials is case-dependent and, hence, must be defined in the respective
    simulation driver.

    .. automethod:: __init__
    .. automethod:: solid_density
    .. automethod:: decomposition_progress
    .. automethod:: void_fraction
    .. automethod:: get_temperature
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    .. automethod:: viscosity
    .. automethod:: dependent_vars
    """

    def __init__(self, wall_material, enthalpy_func, heat_capacity_func,
                 thermal_conductivity_func):
        """
        Initialize the model.

        Parameters
        ----------
        wall_material:
            The class with the solid properties of the desired material.
        wall_sample_mask: meshmode.dof_array.DOFArray
            Array with 1 for the reactive part of the wall and 0 otherwise.
        enthalpy_func:
            function that computes the enthalpy of the entire wall.
            Must include the non-reactive part of the wall, if existing.
        heat_capacity_func:
            function that computes the heat capacity of the entire wall
            Must include the non-reactive part of the wall, if existing.
        thermal_conductivity_func:
            function that computes the thermal conductivity of the entire wall.
            Must include the non-reactive part of the wall, if existing.
        species_diffusivity_func:
            function that computes the species mass diffusivity inside the wall.
            Must include the non-reactive part of the wall, if existing.
        """
        self._material = wall_material
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def solid_density(self, wall_density) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$.

        The material density is relative to the entire control volume, and
        is not to be confused with the intrinsic density, hence the $\epsilon$
        dependence. It is computed as the sum of all N solid phases:

        .. math::
            \epsilon_s \rho_s = \sum_i^N \epsilon_i \rho_i
        """
        if isinstance(wall_density, DOFArray):
            return wall_density
        return sum(wall_density)

    def decomposition_progress(self, wall_density, wall_sample_mask) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(wall_density)
        tau_sample = self._material.decomposition_progress(mass)
        return (
            tau_sample * wall_sample_mask[0]  # reactive material
            - 1.0*(1.0 - wall_sample_mask[0])  # inert material
        )

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ of the sample filled with gas."""
        actx = tau.array_context
        wall_sample_mask = actx.np.where(actx.np.greater(tau, 0.0), 1.0, 0.0)
        return self._material.void_fraction(tau) * wall_sample_mask

    def get_temperature(self, cv: ConservedVars,
                        wall_density: Union[DOFArray, np.ndarray],
                        wall_sample_mask,
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
        if isinstance(tseed, DOFArray) is False:
            temp = tseed + tau*0.0
        else:
            temp = tseed*1.0

        actx = tau.array_context
        eos = gas_model.eos

        rho_gas = cv.mass
        rho_solid = self.solid_density(wall_density)

        rhoe = actx.np.where(
            actx.np.greater(wall_sample_mask[0], 0.1),
            cv.energy - 0.5/cv.mass*np.dot(cv.momentum, cv.momentum),
            cv.energy)

        for _ in range(0, niter):

            gas_internal_energy = actx.np.where(
                actx.np.greater(wall_sample_mask[0], 0.1),
                eos.get_internal_energy(temp, cv.species_mass_fractions),
                0.0)

            gas_heat_capacity_cv = actx.np.where(
                actx.np.greater(wall_sample_mask[0], 0.1),
                eos.heat_capacity_cv(cv, temp),
                0.0)

            eps_rho_e = rho_gas*gas_internal_energy \
                      + rho_solid*self.enthalpy(temp, tau, wall_sample_mask)

            bulk_cp = rho_gas*gas_heat_capacity_cv \
                      + rho_solid*self.heat_capacity(temp, tau, wall_sample_mask)

            temp = temp - (eps_rho_e - rhoe)/bulk_cp

        return temp

    def enthalpy(self, temperature: DOFArray, tau: DOFArray, wall_sample_mask):
        """Return the enthalpy of the wall as a function of temperature.

        Returns
        -------
        enthalpy: meshmode.dof_array.DOFArray
            the wall enthalpy, including all parts of the solid
        """
        return self._enthalpy_func(temperature=temperature, tau=tau,
                                   wall_sample_mask=wall_sample_mask)

    def heat_capacity(self, temperature: DOFArray, tau: DOFArray, wall_sample_mask):
        """Return the heat capacity of the wall.

        Returns
        -------
        heat capacity: meshmode.dof_array.DOFArray
            the wall heat capacity, including all parts of the solid
        """
        return self._heat_capacity_func(temperature=temperature, tau=tau,
                                        wall_sample_mask=wall_sample_mask)

    def viscosity(self, temperature: DOFArray, tau: DOFArray,
                  gas_tv: GasTransportVars) -> DOFArray:
        """Viscosity of the gas through the (porous) wall."""
        actx = tau.array_context
        epsilon = self._material.void_fraction(tau)
        wall_sample_mask = actx.np.where(actx.np.greater(tau, 0.0), 1.0, 0.0)
        return gas_tv.viscosity/epsilon * wall_sample_mask

    def thermal_conductivity(self, cv: ConservedVars,
                             wall_density: Union[DOFArray, np.ndarray],
                             wall_sample_mask: np.ndarray,
                             temperature: DOFArray, tau: DOFArray,
                             gas_tv: GasTransportVars):
        r"""Return the effective thermal conductivity of the gas+solid.

        It is a function of temperature and degradation progress. As the fibers
        are oxidized, they reduce their cross area and, consequenctly, their
        hability to conduct heat.

        It is evaluated using a mass-weighted average given by

        .. math::
            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}

        Returns
        -------
        thermal_conductivity: meshmode.dof_array.DOFArray
            the thermal_conductivity, including all parts of the solid
        """
        y_g = cv.mass/(cv.mass + self.solid_density(wall_density))
        y_s = 1.0 - y_g
        kappa_s = self._thermal_conductivity_func(temperature=temperature, tau=tau,
                                                  wall_sample_mask=wall_sample_mask)
        kappa_g = gas_tv.thermal_conductivity

        return y_s*kappa_s + y_g*kappa_g

    def species_diffusivity(self, temperature: DOFArray, tau: DOFArray,
                            gas_tv: GasTransportVars):
        """Mass diffusivity of gaseous species through the (porous) wall.

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        actx = tau.array_context
        tortuosity = self._material.tortuosity(tau)
        wall_sample_mask = actx.np.where(actx.np.greater(tau, 0.0), 1.0, 0.0)
        return gas_tv.species_diffusivity/tortuosity * wall_sample_mask

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        actx = tau.array_context
        wall_sample_mask = actx.np.where(actx.np.greater(tau, 0.0), 1.0, 0.0)
        return self._material.permeability(tau) * wall_sample_mask

    def dependent_vars(self, wall_density: Union[DOFArray, np.ndarray],
                       wall_sample_mask, temperature: DOFArray) -> WallDependentVars:
        """Get the state-dependent variables."""
        tau = self.decomposition_progress(wall_density, wall_sample_mask)
        return WallDependentVars(
            tau=tau,
            void_fraction=self.void_fraction(tau),
            emissivity=self._material.emissivity(tau),
            permeability=self._material.permeability(tau)
        )
