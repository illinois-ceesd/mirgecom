""":mod:`mirgecom.phenolics.phenolics` handles phenolics modeling.

    Additional details are provided in
    https://github.com/illinois-ceesd/phenolics-notes

Conserved Quantities
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PhenolicsConservedVars

Helper Functions
================
.. autofunction:: initializer
.. autofunction:: make_conserved

Equations of State
^^^^^^^^^^^^^^^^^^
.. autoclass:: PhenolicsDependentVars
.. autoclass:: PhenolicsEOS

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

import numpy as np
from meshmode.dof_array import DOFArray
from dataclasses import dataclass
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)

import sys  # noqa


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsConservedVars:
    r"""Class of conserved variables.

    The conserved variables are

    .. math::

        \epsilon_i \rho_i, \mbox{for i = 1:N constituents}

        \epsilon_g \rho_g, \mbox{assuming single gas in equilibrium}

        \rho e, \mbox{the energy of solid+gas}
    """

    # the "epsilon_density" of each phase in the solid
    solid_species_mass: np.ndarray

    # it includes the epsilon/void fraction
    gas_density: DOFArray

    # bulk energy = solid + gas energy
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.energy)

    @property
    def nphase(self):
        """Return the number of phases in the composite material."""
        return len(self.solid_species_mass)


def initializer(eos, solid_species_mass, temperature, gas_density=None,
                pressure=None, progress=0.0):
    """Initialize state of composite material."""
    if gas_density is None and pressure is None:
        raise ValueError("Must specify one of 'gas_density' or 'pressure'")

    if isinstance(temperature, DOFArray) is False:
        raise ValueError("Temperature does not have the proper shape")

    tau = 280.0/(280.0 - 220.0)*(1.0 - 220.0/sum(solid_species_mass))

    # gas constant
    Rg = 8314.46261815324/eos.gas_molar_mass(temperature)  # noqa N806

    if gas_density is None:
        eps_gas = eos.void_fraction(tau)
        eps_rho_gas = eps_gas*pressure/(Rg*temperature)

    eps_rho_solid = sum(solid_species_mass)
    bulk_energy = (
        eps_rho_solid*eos.solid_enthalpy(temperature, tau)
        + eps_rho_gas*(eos.gas_enthalpy(temperature) - Rg*temperature)
    )

    return PhenolicsConservedVars(solid_species_mass=solid_species_mass,
        energy=bulk_energy, gas_density=eps_rho_gas)


def make_conserved(solid_species_mass, gas_density, energy):  # noqa D103
    return PhenolicsConservedVars(solid_species_mass=solid_species_mass,
        energy=energy, gas_density=gas_density)


@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsDependentVars:
    """State-dependent quantities.

    .. attribute:: progress
    .. attribute:: temperature
    .. attribute:: void_fraction
    .. attribute:: gas_pressure
    .. attribute:: gas_molar_mass
    .. attribute:: gas_viscosity
    .. attribute:: thermal_conductivity
    .. attribute:: solid_emissivity
    .. attribute:: solid_permeability
    .. attribute:: solid_density
    """

    tau: DOFArray
    progress: DOFArray
    temperature: DOFArray

    thermal_conductivity: DOFArray
    void_fraction: DOFArray

    gas_pressure: DOFArray
    gas_enthalpy: DOFArray
    gas_molar_mass: DOFArray
    gas_viscosity: DOFArray

    solid_enthalpy: DOFArray
    solid_emissivity: DOFArray
    solid_permeability: DOFArray
    solid_density: DOFArray


# TODO maybe split this in two, one for "gas" and another for "solid"??
class PhenolicsEOS():
    """.

    .. automethod:: __init__
    .. automethod:: eval_tau
    .. automethod:: eval_temperature
    .. automethod:: void_fraction
    .. automethod:: thermal_conductivity
    .. automethod:: gas_enthalpy
    .. automethod:: gas_molar_mass
    .. automethod:: gas_viscosity
    .. automethod:: gas_thermal_conductivity
    .. automethod:: gas_heat_capacity_cp
    .. automethod:: gas_pressure_diffusivity
    .. automethod:: gas_pressure
    .. automethod:: solid_density
    .. automethod:: solid_permeability
    .. automethod:: solid_emissivity
    .. automethod:: solid_enthalpy
    .. automethod:: solid_heat_capacity_cp
    """

    # FIXME Ideally, we wanna do something similar to the pyrometheus interface
    def __init__(self, composite, gas):
        """Initialize EOS for composite."""
        self._composite_model = composite
        self._gas_data = gas

    # ~~~~~~~~~~~~
    def eval_tau(self, wv: PhenolicsConservedVars) -> DOFArray:
        r"""Evaluate the progress ratio of the phenolics decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the pyrolysis is locally complete and only charred
        material exists.
        """
        return 280.0/(280.0 - 220.0)*(1.0 - 220.0/self.solid_density(wv))

    def eval_temperature(self, wv: PhenolicsConservedVars, tseed: DOFArray,
                         tau: DOFArray, niter=3) -> DOFArray:
        """Evaluate the temperature.

        It uses the assumption of thermal equilibrium between solid and fluid.
        Newton iteration are used to get the temperature based on the internal
        energy/enthalpy and heat capacity for the bulk (solid+gas) material.
        """
        temp = tseed*1.0

        rho_gas = wv.gas_density
        rho_solid = self.solid_density(wv)
        rhoe = wv.energy
        for _ in range(0, niter):

            # gas constant R/M
            molar_mass = self.gas_molar_mass(temp)
            Rg = 8314.46261815324/molar_mass  # noqa N806

            eps_rho_e = (
                rho_gas*(self.gas_enthalpy(temp) - Rg*temp)
                + rho_solid*self.solid_enthalpy(temp, tau))

            bulk_cp = (
                rho_gas*(self.gas_heat_capacity_cp(temp)
                         - Rg*(1.0 - temp/molar_mass*self._gas_data.gas_dMdT(temp)))
                + rho_solid*self.solid_heat_capacity_cp(temp, tau))

            temp = temp - (eps_rho_e - rhoe)/bulk_cp

        return temp

    # ~~~~~~~~~~~~ bulk gas+solid properties
    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.
        """
        return 1.0 - self._composite_model.solid_volume_fraction(tau)

    def thermal_conductivity(self, wv: PhenolicsConservedVars,
                             temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Return the bulk thermal conductivity, $f(\rho, \tau, T)$.

        It is evaluated using a mass-weighted average given by

        .. math::

            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}
        """
        y_g = wv.gas_density/(wv.gas_density + self.solid_density(wv))
        y_s = 1.0 - y_g
        return (
            y_s*self._composite_model.solid_thermal_conductivity(temperature, tau)
            + y_g*self.gas_thermal_conductivity(temperature)
        )

    # ~~~~~~~~~~~~ gas
    def gas_enthalpy(self, temperature: DOFArray) -> DOFArray:
        """Return the gas enthalpy."""
        return self._gas_data.gas_enthalpy(temperature)

    def gas_molar_mass(self, temperature: DOFArray) -> DOFArray:
        """Return the gas molar mass."""
        return self._gas_data.gas_molar_mass(temperature)

    def gas_viscosity(self, temperature: DOFArray) -> DOFArray:
        """Return the gas viscosity."""
        return self._gas_data.gas_viscosity(temperature)

    def gas_thermal_conductivity(self, temperature: DOFArray) -> DOFArray:
        r"""Return the gas thermal conductivity.

        .. math::

            \kappa = \frac{\mu C_p}{Pr}

        where $\mu$ is the gas viscosity and $C_p$ is the heat capacity at
        constant pressure. The Prandtl number $Pr$ is assumed to be 1.
        """
        cp = self.gas_heat_capacity_cp(temperature)
        mu = self._gas_data.gas_viscosity(temperature)
        prandtl = 1.0
        return mu*cp/prandtl

    def gas_pressure_diffusivity(self, temperature: DOFArray,
                                 tau: DOFArray) -> DOFArray:
        r"""Return the normalized pressure diffusivity.

        .. math::

            \frac{1}{\epsilon_g \rho_g} d_{P} = \frac{\mathbf{K}}{\mu \epsilon_g}

        where $\mu$ is the gas viscosity, $\epsilon_g$ is the void fraction
        and $\mathbf{K}$ is the permeability matrix.
        """
        mu = self.gas_viscosity(temperature)
        epsilon = self.void_fraction(tau)
        permeability = self.solid_permeability(tau)
        return permeability/(mu*epsilon)

    def gas_pressure(self, wv: PhenolicsConservedVars,
                     temperature: DOFArray, tau: DOFArray) -> DOFArray:
        r"""Return the gas pressure.

        .. math::

            P = \frac{\epsilon_g \rho_g}{\epsilon_g} \frac{R}{M} T
        """
        Rg = 8314.46261815324/self.gas_molar_mass(temperature)  # noqa N806
        eps_gas = self.void_fraction(tau)
        return (1.0/eps_gas)*wv.gas_density*Rg*temperature

#    # FIXME
#    def velocity(self, wv, temperature, tau):
#        return temperature*0.0

#    # FIXME
#    def species_diffusivity(self, wv, temperature, tau):
#        return temperature*0.0

    # ~~~~~~~~~~~~ solid
    def solid_density(self, wv: PhenolicsConservedVars) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$.

        The material density is relative to the entire control volume and it
        is computed as the sum of all N solid phases:

        .. math::

            \epsilon_s \rho_s = \sum_i^N \epsilon_i \rho_i
        """
        return sum(wv.solid_species_mass)

    def solid_enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Return the solid enthalpy."""
        return self._composite_model.solid_enthalpy(temperature, tau)

    def solid_permeability(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall permeability based on the progress ratio $\tau$."""
        return self._composite_model.solid_permeability(tau)

    def solid_emissivity(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall emissivity based on the progress ratio $\tau$."""
        return self._composite_model.solid_emissivity(tau)

    # ~~~~~~~~~~~~ auxiliary functions
    def gas_heat_capacity_cp(self, temperature: DOFArray) -> DOFArray:
        """Return the gas heat capacity."""
        return self._gas_data.gas_viscosity(temperature)

    def solid_heat_capacity_cp(self, temperature: DOFArray,
                               tau: DOFArray) -> DOFArray:
        """Return the solid heat capacity."""
        return self._composite_model.solid_heat_capacity(temperature, tau)

    # ~~~~~~~~~~~~
    def dependent_vars(self, wv: PhenolicsConservedVars,
            temperature_seed: DOFArray) -> PhenolicsDependentVars:
        """Get the dependent variables."""
        tau = self.eval_tau(wv)
        temperature = self.eval_temperature(wv, temperature_seed, tau)
        return PhenolicsDependentVars(
            tau=tau,
            progress=1.0-tau,
            temperature=temperature,
            thermal_conductivity=self.thermal_conductivity(wv, temperature, tau),
            void_fraction=self.void_fraction(tau),
            gas_pressure=self.gas_pressure(wv, temperature, tau),
            # velocity
            gas_viscosity=self.gas_viscosity(temperature),
            gas_molar_mass=self.gas_molar_mass(temperature),
            # species_diffusivity
            gas_enthalpy=self.gas_enthalpy(temperature),
            solid_enthalpy=self.solid_enthalpy(temperature, tau),
            # heat_capacity
            solid_density=self.solid_density(wv),
            solid_emissivity=self.solid_emissivity(tau),
            solid_permeability=self.solid_permeability(tau)
        )
