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
from mirgecom.multiphysics.wall_model import (
    WallDegradationModel, WallConservedVars
)
from pytools.obj_array import make_obj_array
from mirgecom.multiphysics.wall_model import WallEOS


class WallTabulatedEOS(WallEOS):

    def eval_temperature(self, cv, wv, tseed, tau, eos, niter=3) -> DOFArray:
        r"""Evaluate the temperature.

        It uses the assumption of thermal equilibrium between solid and fluid.
        Newton iteration is used to get the temperature based on the internal
        energy/enthalpy and heat capacity for the bulk (solid+gas) material:

        .. math::
            T^{n+1} = T^n -
                \frac
                {\epsilon_g \rho_g(h_g - R_g T^n) + \rho_s h_s - \rho e}
                {\epsilon_g \rho_g \left(
                    C_{p_g} - R_g\left[1 - \frac{\partial M}{\partial T} \right]
                \right)
                + \epsilon_s \rho_s C_{p_s}
                }

        Parameters
        ----------
        wv: :class:`WallConservedVars`
        tseed: float or :class:`~meshmode.dof_array.DOFArray`
            Optional data from which to seed temperature calculation.
        tau: :class:`~meshmode.dof_array.DOFArray`
            the progress ratio
        niter:
            Optional argument with the number of iterations
        """
        if isinstance(tseed, DOFArray) is False:
            temp = tseed + wv.energy*0.0
        else:
            temp = tseed*1.0

        rho_gas = cv.mass
        rho_solid = self.solid_density(wv)
        rhoe = cv.energy
        for _ in range(0, niter):

            # gas constant R/M
            molar_mass = eos.gas_molar_mass(temp)
            Rg = 8314.46261815324/molar_mass  # noqa N806

            eps_rho_e = (
                rho_gas*(eos.gas_enthalpy(temp) - Rg*temp)
                + rho_solid*self.enthalpy(temp, tau))

            bulk_cp = (
                rho_gas*(eos.gas_heat_capacity(temp)
                         - Rg*(1.0 - temp/molar_mass*eos.gas_dMdT(temp)))
                + rho_solid*self.heat_capacity(temp, tau))

            temp = temp - (eps_rho_e - rhoe)/bulk_cp

        return temp


def initializer(dcoll, gas_model, solid_species_mass, temperature, gas_density=None,
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

    Returns
    -------
    wv: :class:`WallConservedVars`
        The wall conserved variables
    """
    if gas_density is None and pressure is None:
        raise ValueError("Must specify one of 'gas_density' or 'pressure'")

    if isinstance(temperature, DOFArray) is False:
        raise ValueError("Temperature does not have the proper shape")

    wv = WallConservedVars(mass=solid_species_mass)

    tau = gas_model.wall.eval_tau(wv)

    # gas constant
    Rg = 8314.46261815324/gas_model.eos.gas_molar_mass(temperature)  # noqa N806

    if gas_density is None:
        eps_gas = gas_model.wall._sample_model.void_fraction(tau)
        eps_rho_gas = eps_gas*pressure/(Rg*temperature)

    eps_rho_solid = sum(solid_species_mass)
    bulk_energy = (
        eps_rho_solid*gas_model.wall.enthalpy(temperature, tau)
        + eps_rho_gas*(gas_model.eos.gas_enthalpy(temperature) - Rg*temperature)
    )

    dim = dcoll.dim
    momentum = make_obj_array([tau*0.0 for i in range(dim)])

    from mirgecom.fluid import ConservedVars
    cv = ConservedVars(mass=eps_rho_gas, energy=bulk_energy, momentum=momentum)

    return cv, wv


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    """State-dependent quantities.

    .. attribute:: tau
    .. attribute:: temperature
    .. attribute:: void_fraction
    .. attribute:: gas_pressure
    .. attribute:: gas_viscosity
    .. attribute:: thermal_conductivity
    .. attribute:: solid_emissivity
    .. attribute:: solid_permeability
    .. attribute:: solid_density
    """

    tau: DOFArray
    temperature: DOFArray

    thermal_conductivity: DOFArray
    void_fraction: DOFArray

    gas_pressure: DOFArray
    gas_enthalpy: DOFArray
    gas_viscosity: DOFArray

    solid_enthalpy: DOFArray
    solid_emissivity: DOFArray
    solid_permeability: DOFArray
    solid_density: DOFArray


class PhenolicsWallModel(WallDegradationModel):
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

    def __init__(self, solid_data):
        """Initialize wall model for composite.

        solid_data:
            The class with the solid properties of the desired material. For
            example, :class:`mirgecom.materials.tacot.SolidProperties` contains
            data for TACOT.
        gas_data:
            The class with properties of the product gases. For example,
            :class:`mirgecom.materials.tacot.GasProperties` contains data for
            gaseous products of TACOT decomposition.
        """
        self._solid_data = solid_data

    # ~~~~~~~~~~~~
    def eval_tau(self, wv: WallConservedVars) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the pyrolysis is locally complete and only charred
        material exists:

        .. math::
            \tau = \frac{\rho_0}{\rho_0 - \rho_c}
                    \left( 1 - \frac{\rho_c}{\rho(t)} \right)

        Parameters
        ----------
        wv: :class:`WallConservedVars`
            the class of conserved variables for the pyrolysis
        """
        char_mass = self._solid_data._char_mass
        virgin_mass = self._solid_data._virgin_mass
        current_mass = self.solid_density(wv)
        return virgin_mass/(virgin_mass - char_mass)*(1.0 - char_mass/current_mass)

    # ~~~~~~~~~~~~ bulk gas+solid properties
    def void_fraction(self, tau) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.

        Parameters
        ----------
        tau: meshmode.dof_array.DOFArray
        """
        return 1.0 - self._solid_data.solid_volume_fraction(tau)

    def solid_density(self, wv: WallConservedVars) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$.

        The material density is relative to the entire control volume, and
        is not to be confused with the intrinsic density, hence the $\epsilon$
        dependence. It is computed as the sum of all N solid phases:

        .. math::
            \epsilon_s \rho_s = \sum_i^N \epsilon_i \rho_i
        """
        return sum(wv.mass)

    def solid_thermal_conductivity(self, temperature, tau) -> DOFArray:
        r"""Return the solid thermal conductivity, $f(\rho, \tau, T)$."""
        return self._solid_data.solid_thermal_conductivity(temperature, tau)

    def solid_enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Return the solid enthalpy $h_s$."""
        return self._solid_data.solid_enthalpy(temperature, tau)

    def solid_permeability(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall permeability $K$ based on the progress ratio $\tau$."""
        return self._solid_data.solid_permeability(tau)

    def solid_emissivity(self, tau: DOFArray) -> DOFArray:
        r"""Return the wall emissivity based on the progress ratio $\tau$."""
        return self._solid_data.solid_emissivity(tau)

    def solid_heat_capacity_cp(self, temperature: DOFArray,
                               tau: DOFArray) -> DOFArray:
        r"""Return the solid heat capacity $C_{p_s}$."""
        return self._solid_data.solid_heat_capacity(temperature, tau)
