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

    This class evaluates the variables dependent on the wall state. It requires
    functions are inputs in order to consider the different materials employed
    at wall (for instance, carbon fibers, graphite, alumina, steel etc). The
    number and type of material is case-dependent and, hence, must be defined
    at the respective simulation driver.

    .. automethod:: __init__
    .. automethod:: decomposition_progress
    .. automethod:: eval_temperature
    .. automethod:: void_fraction
    .. automethod:: solid_density
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: thermal_diffusivity
    .. automethod:: species_diffusivity
    .. automethod:: viscosity
    .. automethod:: pressure_diffusivity
    .. automethod:: dependent_vars
    """

    def __init__(self, wall_degradation_model, wall_sample_mask,
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
        self._model = wall_degradation_model
        self._sample_mask = wall_sample_mask
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def solid_density(self, wv):
        r"""Return the solid density $\epsilon_s \rho_s$."""
        return self._model.solid_density(wv)

    def decomposition_progress(self, wv: WallConservedVars) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(wv)
        tau_sample = self._model.decomposition_progress(mass)
        return (
            tau_sample * self._sample_mask  # reactive material
            + 1.0*(1.0 - self._sample_mask)  # inert material
        )

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ filled by gas."""
        # FIXME pass a function to make it depend on the holder as well
        return self._model.void_fraction(tau)

    def eval_temperature(self, cv: ConservedVars, wv: WallConservedVars,
                         tseed: DOFArray, tau: DOFArray, eos: GasEOS,
                         niter=3) -> DOFArray:
        r"""Evaluate the temperature.

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
        rhoe = cv.energy  # FIXME minus kinetic energy
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
            the wall enthalpy, including all of its components
        """
        return self._enthalpy_func(temperature=temperature, tau=tau)

    def heat_capacity(self, temperature: DOFArray, tau: DOFArray):
        """Return the heat capacity of the wall.

        Returns
        -------
        heat capacity: meshmode.dof_array.DOFArray
            the wall heat capacity, including all of its components
        """
        return self._heat_capacity_func(temperature=temperature, tau=tau)

    def thermal_conductivity(self, cv: ConservedVars, wv: WallConservedVars,
                             temperature: DOFArray, tau: DOFArray,
                             gas_tv: GasTransportVars) -> DOFArray:
        r"""Return the effective thermal conductivity of the gas+solid.

        It is a function of temperature and degradation progress. As the fibers
        are oxidized, they reduce their cross area and, consequenctly, their
        hability to conduct heat.

        It is evaluated using a mass-weighted average given by

        .. math::
            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}
        """
        y_g = cv.mass/(cv.mass + self.solid_density(wv))
        y_s = 1.0 - y_g
        kappa_s = self._thermal_conductivity_func(temperature=temperature, tau=tau)
        kappa_g = gas_tv.thermal_conductivity

        return y_s*kappa_s + y_g*kappa_g

    def viscosity(self, temperature: DOFArray, tau: DOFArray, gas_tv):
        """Viscosity of the gas through the (porous) wall.

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        epsilon = self._model.void_fraction(tau)
        return gas_tv.viscosity/epsilon

    def species_diffusivity(self, temperature: DOFArray, tau: DOFArray, gas_tv):
        """Mass diffusivity of gaseous species through the (porous) wall.

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        tortuosity = self._model.solid_tortuosity(tau)
        #  FIXME
        #  return self._species_diffusivity_func(temperature)/tortuosity
        return gas_tv.species_diffusivity/tortuosity

    def pressure_diffusivity(self, cv: ConservedVars, tau: DOFArray,
                             gas_tv: GasTransportVars) -> DOFArray:
        r"""Return the normalized pressure diffusivity.

        .. math::
            d_{P} = \epsilon_g \rho_g \frac{\mathbf{K}}{\mu \epsilon_g}

        where $\mu$ is the gas viscosity, $\epsilon_g$ is the void fraction
        and $\mathbf{K}$ is the permeability matrix.
        """
        mu = gas_tv.viscosity
        epsilon = self.void_fraction(tau)
        permeability = self._model.solid_permeability(tau)
        return cv.mass*permeability/(mu*epsilon)

    def dependent_vars(self, wv: WallConservedVars,
                       temperature: DOFArray) -> WallConservedVars:
        """Get the state-dependent variables."""
        tau = self.decomposition_progress(wv)
        return WallDependentVars(
            tau=tau,
            void_fraction=self.void_fraction(tau),
            solid_emissivity=self._model.solid_emissivity(tau),
            solid_permeability=self._model.solid_permeability(tau)
        )


@dataclass_array_container
@dataclass(frozen=True)
class PorousTransportVars(GasTransportVars):
    """Extra transport variable for porous media flow."""

    pressure_diffusivity: DOFArray
