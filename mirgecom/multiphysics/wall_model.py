""":mod:`mirgecom.multiphysics.wall_model` handles the EOS for wall model.

.. autoclass:: WallEOS
.. autoclass:: WallConservedVars
.. autoclass:: WallDependentVars
.. autoclass:: WallDegradationModel
.. autoclass:: PorousTransportVars
"""

from dataclasses import dataclass

from abc import abstractmethod
import numpy as np

from meshmode.dof_array import DOFArray

from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)
from mirgecom.fluid import ConservedVars
from mirgecom.eos import GasEOS
from mirgecom.transport import GasTransportVars


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallConservedVars:
    """Conserved variables exlusively for wall degradation.

    This is used in conjunction with ConservedVars.

    .. attribute:: mass
    """

    mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    """Dependent variables for the Y3 oxidation model.

    .. attribute:: tau
    .. attribute:: void_fraction
    .. attribute:: solid_emissivity
    .. attribute:: solid_permeability
    """

    tau: DOFArray
    void_fraction: DOFArray
    solid_emissivity: DOFArray
    solid_permeability: DOFArray


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
    def solid_density(self, wv: WallConservedVars) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$."""
        raise NotImplementedError()

    @abstractmethod
    def solid_heat_capacity(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Evaluate the solid enthalpy $h_s$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_thermal_conductivity(self, temperature: DOFArray, tau: DOFArray):
        r"""Evaluate the thermal conductivity $\kappa$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_emissivity(self, tau: DOFArray):
        """Emissivity for energy radiation."""
        raise NotImplementedError()

    @abstractmethod
    def solid_tortuosity(self, tau: DOFArray):
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
    .. automethod:: decomposition_progress
    .. automethod:: get_temperature
    .. automethod:: void_fraction
    .. automethod:: solid_density
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    .. automethod:: viscosity
    .. automethod:: pressure_diffusivity
    .. automethod:: dependent_vars
    """

    def __init__(self, wall_material, wall_sample_mask,
                 enthalpy_func, heat_capacity_func, thermal_conductivity_func):
        """
        Initialize the model.

        Parameters
        ----------
        solid_data:
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
        self._sample_mask = wall_sample_mask
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def solid_density(self, wv: WallConservedVars) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$."""
        return self._material.solid_density(wv)

    def decomposition_progress(self, wv: WallConservedVars) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(wv)
        tau_sample = self._material.solid_decomposition_progress(mass)
        return (
            tau_sample * self._sample_mask  # reactive material
            + 1.0*(1.0 - self._sample_mask)  # inert material
        )

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ of the sample filled with gas."""
        return self._material.void_fraction(tau) * self._sample_mask

    def get_temperature(self, cv: ConservedVars, wv: WallConservedVars,
                         tseed: DOFArray, tau: DOFArray, eos: GasEOS,
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
            temp = tseed + wv.mass*0.0
        else:
            temp = tseed*1.0

        rho_gas = cv.mass
        rho_solid = self.solid_density(wv)
        rhoe = cv.energy  # FIXME minus kinetic energy (non-existent for now)
        for _ in range(0, niter):

            gas_internal_energy = \
                eos.get_internal_energy(temp, cv.species_mass_fractions)

            eps_rho_e = rho_gas*gas_internal_energy \
                      + rho_solid*self.enthalpy(temp, tau)

            gas_heat_capacity_cv = eos.heat_capacity_cv(cv, temp)
            bulk_cp = rho_gas*gas_heat_capacity_cv \
                      + rho_solid*self.heat_capacity(temp, tau)

            temp = temp - (eps_rho_e - rhoe)/bulk_cp

        return temp

    def enthalpy(self, temperature: DOFArray, tau: DOFArray):
        """Return the enthalpy of the wall as a function of temperature.

        Returns
        -------
        enthalpy: meshmode.dof_array.DOFArray
            the wall enthalpy, including all parts of the solid
        """
        return self._enthalpy_func(temperature=temperature, tau=tau)

    def heat_capacity(self, temperature: DOFArray, tau: DOFArray):
        """Return the heat capacity of the wall.

        Returns
        -------
        heat capacity: meshmode.dof_array.DOFArray
            the wall heat capacity, including all parts of the solid
        """
        return self._heat_capacity_func(temperature=temperature, tau=tau)

    def thermal_conductivity(self, cv: ConservedVars, wv: WallConservedVars,
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
        y_g = cv.mass/(cv.mass + self.solid_density(wv))
        y_s = 1.0 - y_g
        kappa_s = self._thermal_conductivity_func(temperature=temperature, tau=tau)
        kappa_g = gas_tv.thermal_conductivity

        return y_s*kappa_s + y_g*kappa_g

    def viscosity(self, temperature: DOFArray, tau: DOFArray,
                  gas_tv: GasTransportVars) -> DOFArray:
        """Viscosity of the gas through the (porous) wall."""
        epsilon = self._material.void_fraction(tau)
        return gas_tv.viscosity/epsilon * self._sample_mask

    def species_diffusivity(self, temperature: DOFArray, tau: DOFArray,
                            gas_tv: GasTransportVars) -> DOFArray:
        """Mass diffusivity of gaseous species through the (porous) wall.

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        tortuosity = self._material.solid_tortuosity(tau)
        return gas_tv.species_diffusivity/tortuosity * self._sample_mask

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        return self._material.solid_permeability(tau) * self._sample_mask

    def pressure_diffusivity(self, cv: ConservedVars, tau: DOFArray,
                             gas_tv: GasTransportVars) -> DOFArray:
        r"""Return the pressure diffusivity for Darcy flow.

        .. math::
            d_{P} = \epsilon_g \rho_g \frac{\mathbf{K}}{\mu \epsilon_g}

        where $\mu$ is the gas viscosity, $\epsilon_g$ is the void fraction
        and $\mathbf{K}$ is the permeability matrix.
        """
        mu = gas_tv.viscosity
        epsilon = self.void_fraction(tau)
        permeability = self.permeability(tau)
        return cv.mass*permeability/(mu*epsilon) * self._sample_mask

    def dependent_vars(self, wv: WallConservedVars,
                       temperature: DOFArray) -> WallDependentVars:
        """Get the state-dependent variables."""
        tau = self.decomposition_progress(wv)
        return WallDependentVars(
            tau=tau,
            void_fraction=self.void_fraction(tau),
            solid_emissivity=self._material.solid_emissivity(tau),
            solid_permeability=self._material.solid_permeability(tau)
        )


@dataclass_array_container
@dataclass(frozen=True)
class PorousTransportVars(GasTransportVars):
    """Extra transport variable for porous media flow.

    .. attribute:: pressure_diffusivity
    """

    pressure_diffusivity: DOFArray
