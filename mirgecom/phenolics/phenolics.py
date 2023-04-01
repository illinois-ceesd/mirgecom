"""
TODO list

X - get the material properties
X - evaluate Newton iterations to compute the temperature
X - operator and heat conduction???
4 - get the gas data... How to handle the tabulated interpolation?
"""

##############################################################################

import numpy as np
from meshmode.dof_array import DOFArray
from dataclasses import dataclass, fields, field
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)
from abc import ABCMeta, abstractmethod

import sys  # noqa


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsConservedVars:
    r"""."""

    solid_species_mass: np.ndarray
    gas_density: DOFArray
    gas_species_mass: DOFArray
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.energy)

    @property
    def nphase(self):
        """Return the number of phases in the composite material."""
        return len(self.solid_species_mass)


def initializer(composite, solid_species_mass, gas_density, gas_species_mass=None,
                energy=None, temperature=None, progress=0.0):

    tau = 1.0 - progress

    if energy is None and temperature is None:
        raise ValueError("Must specify one of 'energy' or 'temperature'")

    if isinstance(tau, DOFArray) is False:
        tau = tau + 0*gas_density

    if isinstance(temperature, DOFArray) is False:
        temperature = temperature + 0.0*gas_density

    if energy is None:
        solid_density = sum(solid_species_mass)
        energy = solid_density*composite.solid_enthalpy(temperature, tau)
                    #+ gas (internal and kinetic)?

    if gas_species_mass is None:
        gas_species_mass = gas_density*1.0

    return PhenolicsConservedVars(solid_species_mass=solid_species_mass,
        energy=energy, gas_density=gas_density, gas_species_mass=gas_species_mass                    
    )


def make_conserved(solid_species_mass, gas_density, gas_species_mass, energy):
    return PhenolicsConservedVars(solid_species_mass=solid_species_mass,
        energy=energy, gas_density=gas_density, gas_species_mass=gas_species_mass                    
    )


@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsDependentVars:
    """State-dependent quantities.

    .. attribute:: temperature
    .. attribute:: pressure
    .. attribute:: velocity
    ... and so on...
    """

    temperature: DOFArray

#    pressure: DOFArray
#    velocity: DOFArray
    molar_mass: DOFArray
    viscosity: DOFArray
    thermal_conductivity: DOFArray
#    species_diffusivity: np.ndarray

    progress: DOFArray
    emissivity: DOFArray
    permeability: DOFArray
    volume_fraction: DOFArray
    solid_density: DOFArray


class PhenolicsEOS():

    def __init__(self, composite, gas):
        self._degradation_model = composite
        self._gas_data = gas

    def gas_molar_mass(self, temp):
        return self._gas_data.gas_molar_mass(temp)

    def gas_viscosity(self, temp):
        return self._gas_data.gas_viscosity(temp)

#    # FIXME
#    def pressure(self, wv, temperature, tau):
#        return temperature*0.0

#    # FIXME
#    def velocity(self, wv, temperature, tau):
#        return temperature*0.0

#    # FIXME
#    def species_diffusivity(self, wv, temperature, tau):
#        return temperature*0.0

    #~~~~~~~~~~~~ solid 
    def enthalpy(self, temp, tau):
        return self._degradation_model.solid_enthalpy(temp, tau)

    def heat_capacity_cp(self, temp, tau):
        return self._degradation_model.solid_heat_capacity(temp, tau)

    def thermal_conductivity(self, temp, tau):
        return self._degradation_model.solid_thermal_conductivity(temp, tau)

    def permeability(self, temp, tau):
        return self._degradation_model.solid_permeability(temp, tau)

    def volume_fraction(self, temp, tau):
        return self._degradation_model.solid_volume_fraction(temp, tau)

    def emissivity(self, temp, tau):
        return self._degradation_model.solid_emissivity(temp, tau)

    def progress(self, wv):
        return 280.0/(280.0 - 220.0)*( 1.0 - 220.0/self.solid_density(wv) )

    def solid_density(self, wv):
        return sum(wv.solid_species_mass)

    def eval_temperature(self, wv, eos, tseed, tau):
        """Temperature assumes thermal equilibrium between solid and fluid."""

        niter = 3
        T = tseed*1.0

        rho_solid = self.solid_density(wv)
        e = wv.energy
        for ii in range(0, niter):

            #M = gas.molar_mass(T)
            eps_rho_e_g = 0.0 #wv[1]*( gas.h(T) - R/M*T )
            eps_rho_e_s = rho_solid*( self.enthalpy(T, tau) )

            f_prime_g = 0.0 #wv[1]*( gas.cp(T) - R/M*( 1.0 - T/M*gas.dMdT(T) ) )
            f_prime_s = rho_solid*( self.heat_capacity_cp(T, tau) )

            f = eps_rho_e_s + eps_rho_e_g
            f_prime = f_prime_s + f_prime_g

            T = T - (f - e)/f_prime

        return T

    def dependent_vars(self, wv: PhenolicsConservedVars,
            eos,
            temperature_seed: DOFArray) -> PhenolicsDependentVars:
        """Get an agglomerated array of the dependent variables."""
        progress = self.progress(wv)
        temperature = self.eval_temperature(wv, eos, temperature_seed, progress)
        return PhenolicsDependentVars(
            progress=progress,
            temperature=temperature,
            #pressure=self.pressure(wv, temperature, progress),
            #velocity=self.velocity(wv, temperature, progress),
            viscosity=self.gas_viscosity(temperature),
            molar_mass=self.gas_molar_mass(temperature),
            #species_diffusivity=self.species_diffusivity(wv, temperature, progress),
            #enthalpy
            #heat_capacity
            thermal_conductivity=self.thermal_conductivity(temperature, progress),
            emissivity=self.emissivity(temperature, progress),
            permeability=self.permeability(temperature, progress),
            volume_fraction=self.volume_fraction(temperature, progress),
            solid_density=self.solid_density(wv)
        )
