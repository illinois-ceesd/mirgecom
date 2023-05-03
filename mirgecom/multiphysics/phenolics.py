""":mod:`mirgecom.multiphysics.phenolics` handles phenolics modeling."""

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

from dataclasses import dataclass, fields
import numpy as np
from meshmode.dof_array import DOFArray
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsConservedVars:
    r"""Class of conserved variables.

    .. attribute:: solid_species_mass
    .. attribute:: gas_density
    .. attribute:: energy
    """

    # the "epsilon_density" of each phase in the solid
    solid_species_mass: np.ndarray

    # it includes the epsilon/void fraction
    gas_density: DOFArray

    # bulk energy = solid + gas energy
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for :class:`PhenolicsConservedVars` object."""
        return get_container_context_recursively(self.energy)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (PhenolicsConservedVars, tuple(getattr(self, f.name)
                                    for f in fields(PhenolicsConservedVars)))

    @property
    def nphase(self):
        """Return the number of phases in the composite material."""
        return len(self.solid_species_mass)


def initializer(wall_model, solid_species_mass, temperature, gas_density=None,
                pressure=None):
    """Initialize state of composite material.

    Parameters
    ----------
    wall_model
        :class:`PhenolicsWallModel`

    solid_species_mass: numpy.ndarray
        The initial bulk density of each one of the resin constituents.
        It has shape ``(nphase,)``

    temperature: :class:`~meshmode.dof_array.DOFArray`
        The initial temperature of the gas+solid

    gas_density: :class:`~meshmode.dof_array.DOFArray`
        Optional argument with the gas density. If not provided, the pressure
        will be used to evaluate the density.

    pressure: :class:`~meshmode.dof_array.DOFArray`
        Optional argument with the gas pressure. It will be used to evaluate
        the gas density.
    """
    if gas_density is None and pressure is None:
        raise ValueError("Must specify one of 'gas_density' or 'pressure'")

    if isinstance(temperature, DOFArray) is False:
        raise ValueError("Temperature does not have the proper shape")

    char_mass = wall_model._solid_data._char_mass
    virgin_mass = wall_model._solid_data._virgin_mass
    current_mass = sum(solid_species_mass)
    tau = virgin_mass/(virgin_mass - char_mass)*(1.0 - char_mass/current_mass)

    # gas constant
    Rg = 8314.46261815324/wall_model.gas_molar_mass(temperature)  # noqa N806

    if gas_density is None:
        eps_gas = wall_model.void_fraction(tau)
        eps_rho_gas = eps_gas*pressure/(Rg*temperature)

    eps_rho_solid = sum(solid_species_mass)
    bulk_energy = (
        eps_rho_solid*wall_model.solid_enthalpy(temperature, tau)
        + eps_rho_gas*(wall_model.gas_enthalpy(temperature) - Rg*temperature)
    )

    return PhenolicsConservedVars(solid_species_mass=solid_species_mass,
        energy=bulk_energy, gas_density=eps_rho_gas)


@dataclass_array_container
@dataclass(frozen=True)
class PhenolicsDependentVars:
    """State-dependent quantities.

    .. attribute:: tau
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


class PhenolicsWallModel:
    """Variables dependent on the wall state.

    .. automethod:: __init__
    .. automethod:: eval_tau
    .. automethod:: eval_temperature
    .. automethod:: void_fraction
    .. automethod:: thermal_conductivity
    .. automethod:: gas_enthalpy
    .. automethod:: gas_molar_mass
    .. automethod:: gas_viscosity
    .. automethod:: gas_heat_capacity_cp
    .. automethod:: gas_pressure_diffusivity
    .. automethod:: gas_pressure
    .. automethod:: solid_density
    .. automethod:: solid_permeability
    .. automethod:: solid_emissivity
    .. automethod:: solid_enthalpy
    .. automethod:: solid_heat_capacity_cp
    """

    def __init__(self, solid_data, gas_data):
        """Initialize wall model for composite.

        solid_data: 
            The class with the solid properties of the desired material.
        gas_data: 
            The class with properties of the product gases.
        """
        self._solid_data = solid_data
        self._gas_data = gas_data

    # ~~~~~~~~~~~~
    def eval_tau(self, wv: PhenolicsConservedVars) -> DOFArray:
        r"""Evaluate the progress ratio of the phenolics decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the pyrolysis is locally complete and only charred
        material exists.

        Parameters
        ----------
        wv: :class:`PhenolicsConservedVars`
            the class of conserved variables for the pyrolysis

        Returns
        -------
        tau: meshmode.dof_array.DOFArray
        """
        char_mass = self._solid_data._char_mass
        virgin_mass = self._solid_data._virgin_mass
        current_mass = self.solid_density(wv)
        return virgin_mass/(virgin_mass - char_mass)*(1.0 - char_mass/current_mass)

    def eval_temperature(self, wv, tseed, tau, niter=3):
        """Evaluate the temperature.

        It uses the assumption of thermal equilibrium between solid and fluid.
        Newton iteration are used to get the temperature based on the internal
        energy/enthalpy and heat capacity for the bulk (solid+gas) material.

        Parameters
        ----------
        wv: :class:`PhenolicsConservedVars`
        tseed: float or :class:`~meshmode.dof_array.DOFArray`
            Optional data from which to seed temperature calculation.
        tau: :class:`~meshmode.dof_array.DOFArray`
            the progress ratio
        niter:
            Optional argument with the number of iterations

        Returns
        -------
        temperature: meshmode.dof_array.DOFArray
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
    def void_fraction(self, tau):
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.

        Parameters
        ----------
        tau: meshmode.dof_array.DOFArray

        Returns
        -------
        void_fraction: meshmode.dof_array.DOFArray
        """
        return 1.0 - self._solid_data.solid_volume_fraction(tau)

    def thermal_conductivity(self, wv, temperature, tau):
        r"""Return the bulk thermal conductivity, $f(\rho, \tau, T)$.

        It is evaluated using a mass-weighted average given by

        .. math::

            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}

        Parameters
        ----------
        wv: :class:`PhenolicsConservedVars`
        temperature: meshmode.dof_array.DOFArray
        tau: meshmode.dof_array.DOFArray

        Returns
        -------
        thermal_conductivity: meshmode.dof_array.DOFArray
        """
        y_g = wv.gas_density/(wv.gas_density + self.solid_density(wv))
        y_s = 1.0 - y_g
        return (
            y_s*self._solid_data.solid_thermal_conductivity(temperature, tau)
            + y_g*self._gas_data.gas_thermal_conductivity(temperature)
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
        return self._solid_data.solid_enthalpy(temperature, tau)

    def solid_permeability(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall permeability based on the progress ratio $\tau$."""
        return self._solid_data.solid_permeability(tau)

    def solid_emissivity(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall emissivity based on the progress ratio $\tau$."""
        return self._solid_data.solid_emissivity(tau)

    # ~~~~~~~~~~~~ auxiliary functions
    def gas_heat_capacity_cp(self, temperature: DOFArray) -> DOFArray:
        """Return the gas heat capacity."""
        return self._gas_data.gas_heat_capacity(temperature)

    def solid_heat_capacity_cp(self, temperature: DOFArray,
                               tau: DOFArray) -> DOFArray:
        """Return the solid heat capacity."""
        return self._solid_data.solid_heat_capacity(temperature, tau)

    # ~~~~~~~~~~~~
    def dependent_vars(self, wv: PhenolicsConservedVars,
            temperature_seed: DOFArray) -> PhenolicsDependentVars:
        """Get the dependent variables."""
        tau = self.eval_tau(wv)
        temperature = self.eval_temperature(wv, temperature_seed, tau)
        return PhenolicsDependentVars(
            tau=tau,
            progress=1.0-tau,  # dummy
            temperature=temperature,
            thermal_conductivity=self.thermal_conductivity(wv, temperature, tau),
            void_fraction=self.void_fraction(tau),
            gas_pressure=self.gas_pressure(wv, temperature, tau),
            gas_viscosity=self.gas_viscosity(temperature),
            gas_molar_mass=self.gas_molar_mass(temperature),  # dummy
            # species_diffusivity
            gas_enthalpy=self.gas_enthalpy(temperature),
            solid_enthalpy=self.solid_enthalpy(temperature, tau),  # dummy
            solid_density=self.solid_density(wv),
            solid_emissivity=self.solid_emissivity(tau),
            solid_permeability=self.solid_permeability(tau)
        )
