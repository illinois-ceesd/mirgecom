"""."""

from dataclasses import dataclass, fields  # noqa

import numpy as np

from meshmode.dof_array import DOFArray

from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    # get_container_context_recursively
)
from abc import (
    abstractmethod,
    # ABCMeta
)
from mirgecom.fluid import ConservedVars
from mirgecom.eos import GasEOS


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallConservedVars:

    mass: DOFArray


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    """Dependent variables for the Y3 oxidation model."""

    tau: DOFArray
    thermal_conductivity: DOFArray
    species_diffusivity: DOFArray
    species_viscosity: DOFArray
    temperature: DOFArray


class WallDegradationModel:

    @abstractmethod
    def intrinsic_density(self):
        r"""Return the intrinsic density $\rho$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_heat_capacity(self, temperature: DOFArray) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_enthalpy(self, temperature: DOFArray) -> DOFArray:
        r"""Evaluate the solid enthalpy $h_s$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_thermal_conductivity(self, temperature: DOFArray, tau: DOFArray):
        r"""Evaluate the thermal conductivity $\kappa$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_volume_fraction(self, tau: DOFArray):
        r"""Void fraction $\epsilon$ filled by gas around the solid."""
        raise NotImplementedError()

    @abstractmethod
    def solid_emissivity(self, tau: DOFArray):
        """Emissivity for energy radiation."""
        raise NotImplementedError()

    @abstractmethod
    def solid_tortuosity(self, tau: DOFArray):
        """."""
        raise NotImplementedError()


class WallEOS:
    """Oxidation model used for the Y3 wall model.

    This class evaluates the variables dependent on the wall state. It requires
    functions are inputs in order to consider the different materials employed
    at wall (for instance, carbon fibers, graphite, alumina, steel etc). The
    number and type of material is case dependent and, hence, must be defined
    at the respective simulation driver.

    .. automethod:: __init__
    .. automethod:: eval_tau
    .. automethod:: eval_temperature
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: thermal_diffusivity
    .. automethod:: species_diffusivity
    .. automethod:: viscosity
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
        self._sample_model = wall_degradation_model
        self._sample_mask = wall_sample_mask
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def eval_tau(self, wv: WallConservedVars) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        tau_sample = self._sample_model.eval_tau(wv)
        return (
            tau_sample * self._sample_mask  # reactive material
            + 1.0*(1.0 - self._sample_mask)  # inert material
        )

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
                {\epsilon_g \rho_g C_{p_g} + \epsilon_s \rho_s C_{p_s}
                }

        """
        if isinstance(tseed, DOFArray) is False:
            temp = tseed + wv.mass*0.0
        else:
            temp = tseed*1.0

        rho_gas = cv.mass
        rho_solid = self.solid_density(wv)
        rhoe = cv.energy #FIXME minus kinetic energy
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

    def solid_density(self, wv):
        return sum(wv.mass)
#        return wv.mass

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

    def thermal_conductivity(self, cv, wv, temperature, tau, gas_tv) -> DOFArray:
        r"""Return the effective thermal conductivity of the gas+solid.

        It is a function of temperature and degradation progress. As the fibers
        are oxidized, they reduce their cross area and, consequenctly, their
        hability to conduct heat.

        It is evaluated using a mass-weighted average given by

        .. math::
            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}

        Parameters
        ----------
        wv: :class:`WallConservedVars`
        temperature: meshmode.dof_array.DOFArray
        tau: meshmode.dof_array.DOFArray
        """
        y_g = cv.mass/(cv.mass + self.solid_density(wv))
        y_s = 1.0 - y_g
        kappa_s = self._thermal_conductivity_func(temperature=temperature, tau=tau)
        kappa_g = gas_tv.thermal_conductivity

        return y_s*kappa_s + y_g*kappa_g

#    def thermal_diffusivity(self, mass, temperature, tau,
#                            thermal_conductivity=None):
#        """Thermal diffusivity of the wall.

#        Parameters
#        ----------
#        mass: meshmode.dof_array.DOFArray
#        temperature: meshmode.dof_array.DOFArray
#        tau: meshmode.dof_array.DOFArray
#        thermal_conductivity: meshmode.dof_array.DOFArray
#            Optional. If not given, it will be evaluated using
#            :func:`thermal_conductivity`

#        Returns
#        -------
#        thermal_diffusivity: meshmode.dof_array.DOFArray
#            the wall thermal diffusivity, including all of its components
#        """
#        if thermal_conductivity is None:
#            thermal_conductivity = self.thermal_conductivity(
#                temperature=temperature, tau=tau)
#        return thermal_conductivity/(mass * self.heat_capacity(temperature))

    def viscosity(self, temperature: DOFArray, tau: DOFArray, gas_tv):
        """Viscosity of the gas through the (porous) wall.

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        epsilon = self._sample_model.void_fraction(tau)
        return gas_tv.viscosity/epsilon

    def species_diffusivity(self, temperature: DOFArray, tau: DOFArray, gas_tv):
        """Mass diffusivity of gaseous species through the (porous) wall.

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        tortuosity = self._sample_model.solid_tortuosity(tau)
        #FIXME
        #return self._species_diffusivity_func(temperature)/tortuosity
        return gas_tv.species_diffusivity/tortuosity

#    def dependent_vars(self, cv, wv, tseed):
#        """Get the dependent variables."""
#        tau = self.eval_tau(wv=wv)
#        temperature = self.eval_temperature(cv=cv, wv=wv, tseed=tseed, tau=tau)
#        kappa = self.thermal_conductivity(temperature=temperature, tau=tau)
#        species_diffusivity = self.species_diffusivity(temperature=temperature)
#        return WallDependentVars(
#            tau=tau,
#            thermal_conductivity=kappa,
#            temperature=temperature,
#            species_diffusivity=species_diffusivity)
