"""
TODO list

1 - get the material properties
2 - evaluate Newton iterations to compute the temperature
3 - operator and heat conduction???
"""

##############################################################################

import numpy as np  # noqa
from meshmode.dof_array import DOFArray  # noqa
from dataclasses import dataclass, fields, field
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)
from abc import ABCMeta, abstractmethod


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
        solid_cp = composite.solid_heat_capacity(temperature, tau)
        gas_cp = 0.0
        energy = (solid_density*solid_cp*temperature + 
                    gas_density*gas_cp*temperature)

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
    """State-dependent quantities for :class:`GasEOS`.

    Prefer individual methods for model use, use this
    structure for visualization or probing.

    .. attribute:: temperature
    .. attribute:: pressure
    .. attribute:: velocity
    ... and so on...
    """

    temperature: DOFArray
    pressure: DOFArray
    velocity: DOFArray
    progress: DOFArray

    #XXX why are these nd.array and not DOFArray?
    viscosity: np.ndarray
    thermal_conductivity: np.ndarray
    species_diffusivity: np.ndarray

    emissivity: DOFArray
    permeability: DOFArray
    volume_fraction: DOFArray
    solid_density: DOFArray


class PhenolicsEOS():

    def __init__(self, composite):
        self._degradation_model = composite

    def gas_const(self, wv, temperature):
        return temperature*0.0

    def pressure(self, wv, temperature, tau):
        return temperature*0.0

    def velocity(self, wv, temperature, tau):
        return temperature*0.0

    def viscosity(self, wv, temperature, tau):
        return temperature*0.0

    def thermal_conductivity(self, wv, temperature, tau):
        return self._degradation_model.solid_thermal_conductivity(temperature, tau)

    def species_diffusivity(self, wv, temperature, tau):
        return temperature*0.0

    def heat_capacity_cp(self, wv, temperature, tau):
        return self._degradation_model.solid_heat_capacity(temperature, tau)

    def permeability(self, wv, temperature, tau):
        return self._degradation_model.solid_permeability(temperature, tau)

    def volume_fraction(self, wv, temperature, tau):
        return self._degradation_model.solid_volume_fraction(temperature, tau)

    def emissivity(self, wv, temperature, tau):
        return self._degradation_model.solid_emissivity(temperature, tau)

    def progress(self, wv):
        return 280.0/(280.0 - 220.0)*( 1.0 - 220.0/self.solid_density(wv) )

    def solid_density(self, wv):
        return sum(wv.solid_species_mass)

    def temperature(self, wv, tseed):
        # do some Newton iterations here
        return 0.0 + tseed*0.0

    def dependent_vars(self, wv: PhenolicsConservedVars,
            temperature_seed: DOFArray) -> PhenolicsDependentVars:
        """Get an agglomerated array of the dependent variables."""
        progress = self.progress(wv)
        temperature = self.temperature(wv, temperature_seed)
        return PhenolicsDependentVars(
            progress=progress,
            temperature=temperature,
            pressure=self.pressure(wv, temperature, progress),
            velocity=self.velocity(wv, temperature, progress),
            viscosity=self.viscosity(wv, temperature, progress),
            thermal_conductivity=self.thermal_conductivity(wv, temperature, progress),
            emissivity=self.emissivity(wv, temperature, progress),
            permeability=self.permeability(wv, temperature, progress),
            volume_fraction=self.volume_fraction(wv, temperature, progress),
            species_diffusivity=temperature*0.0,
            solid_density=self.solid_density(wv)
        )
