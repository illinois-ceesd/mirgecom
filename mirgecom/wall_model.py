""":mod:`mirgecom.multiphysics.wall_model` handles the EOS for wall model.

.. autoclass:: PorousFlowDependentVars
.. autoclass:: WallEOS
.. autoclass:: WallDependentVars
.. autoclass:: WallDegradationModel
"""

from dataclasses import dataclass
from abc import abstractmethod
from typing import Union
import numpy as np
from meshmode.dof_array import DOFArray
from arraycontext import dataclass_array_container
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


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
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


class WallDegradationModel:
    """Abstract interface for wall degradation model."""

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
    """Interface for porous material degradation models.

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

    def __init__(self, wall_material):
        """
        Initialize the model.

        Parameters
        ----------
        wall_material:
            The class with the solid properties of the desired material.
        """
        self._material = wall_material

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

    def decomposition_progress(self, wall_density) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(wall_density)
        return self._material.decomposition_progress(mass)

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ of the sample filled with gas."""
        return self._material.void_fraction(tau)

    def get_temperature(self, cv: ConservedVars,
                        wall_density: Union[DOFArray, np.ndarray],
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

        eos = gas_model.eos

        rho_gas = cv.mass
        rho_solid = self.solid_density(wall_density)

        rhoe = cv.energy - 0.5/cv.mass*np.dot(cv.momentum, cv.momentum)

        for _ in range(0, niter):

            gas_internal_energy = \
                eos.get_internal_energy(temp, cv.species_mass_fractions)

            gas_heat_capacity_cv = eos.heat_capacity_cv(cv, temp)

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

    def viscosity(self, temperature: DOFArray, tau: DOFArray,
                  gas_tv: GasTransportVars) -> DOFArray:
        """Viscosity of the gas through the (porous) wall."""
        epsilon = self._material.void_fraction(tau)
        return gas_tv.viscosity/epsilon

    def thermal_conductivity(self, cv: ConservedVars,
                             wall_density: Union[DOFArray, np.ndarray],
                             temperature: DOFArray, tau: DOFArray,
                             gas_tv: GasTransportVars):
        r"""Return the effective thermal conductivity of the gas+solid.

        It is a function of temperature and degradation progress. As the fibers
        are oxidized, they reduce their cross area and, consequently, their
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
        kappa_s = self._material.thermal_conductivity(temperature, tau)
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
        tortuosity = self._material.tortuosity(tau)
        return gas_tv.species_diffusivity/tortuosity

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        return self._material.permeability(tau)

    def dependent_vars(
            self, wall_density: Union[DOFArray, np.ndarray]) -> WallDependentVars:
        """Get the state-dependent variables."""
        tau = self.decomposition_progress(wall_density)
        return WallDependentVars(
            tau=tau,
            void_fraction=self.void_fraction(tau),
            emissivity=self._material.emissivity(tau),
            permeability=self._material.permeability(tau),
            density=self.solid_density(wall_density)
        )
