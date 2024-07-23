""":mod:`mirgecom.wall_model` handles solid walls and fluid-wall models.

Solid materials
^^^^^^^^^^^^^^^

Impermeable-wall variables
---------------------------
.. autoclass:: SolidWallConservedVars
.. autoclass:: SolidWallDependentVars
.. autoclass:: SolidWallState

Impermeable material properties
-------------------------------
    The material properties are defined exclusively at the driver. The user
    must provide density, enthalpy, heat capacity and thermal conductivity of
    all constituents and their spatial distribution using
    :mod:`~mirgecom.utils.mask_from_elements`.

Impermeable wall model
----------------------
.. autoclass:: SolidWallModel

Porous materials
^^^^^^^^^^^^^^^^

Porous-wall variables
---------------------
.. autoclass:: PorousWallVars

Porous Media Transport
----------------------
.. autoclass:: PorousWallTransport

Porous material properties
--------------------------
    The properties of the materials are defined in specific files and used by
    :class:`PorousWallEOS`.

.. autoclass:: PorousWallEOS

Porous Media Model
------------------
.. autoclass:: PorousFlowModel
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


from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from arraycontext import (
    dataclass_array_container,
    get_container_context_recursively,
    with_container_arithmetic,
)
from meshmode.dof_array import DOFArray

from mirgecom.eos import GasDependentVars, GasEOS, MixtureEOS
from mirgecom.fluid import ConservedVars
from mirgecom.transport import GasTransportVars, TransportModel


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class SolidWallConservedVars:
    """Wall conserved variables for heat conduction only material."""

    mass: DOFArray
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`SolidWallConservedVars` object."""
        return get_container_context_recursively(self.mass)


@dataclass_array_container
@dataclass(frozen=True, eq=False)
class SolidWallDependentVars:
    """Wall dependent variables for heat conduction only materials."""

    thermal_conductivity: DOFArray
    temperature: DOFArray


@dataclass_array_container
@dataclass(frozen=True)
class SolidWallState:
    """Wall state for heat conduction only materials."""

    cv: SolidWallConservedVars
    dv: SolidWallDependentVars


class SolidWallModel:
    """Model for calculating wall quantities for heat conduction only materials.

    Since different materials can be coupled together in the same domain
    governed by diffusion equation, this wall model works as a wrapper of
    driver-defined functions that prescribe the actual properties and their
    spatial distribution.

    .. automethod:: density
    .. automethod:: heat_capacity
    .. automethod:: enthalpy
    .. automethod:: thermal_diffusivity
    .. automethod:: thermal_conductivity
    .. automethod:: get_temperature
    .. automethod:: dependent_vars
    """

    def __init__(self, density_func, enthalpy_func, heat_capacity_func,
                 thermal_conductivity_func):
        self._density_func = density_func
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def density(self) -> DOFArray:
        """Return the wall density for all components."""
        return self._density_func()

    def heat_capacity(self, temperature: DOFArray = None) -> DOFArray:
        """Return the wall heat_capacity for all components."""
        return self._heat_capacity_func(temperature)

    def enthalpy(self, temperature: DOFArray) -> DOFArray:
        """Return the wall enthalpy for all components."""
        return self._enthalpy_func(temperature)

    def thermal_diffusivity(self, mass: DOFArray, temperature: DOFArray,
            thermal_conductivity: DOFArray = None) -> DOFArray:
        """Return the wall thermal diffusivity for all components."""
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(temperature)
        return thermal_conductivity/(mass * self.heat_capacity(temperature))

    def thermal_conductivity(self, temperature: DOFArray) -> DOFArray:
        """Return the wall thermal conductivity for all components."""
        return self._thermal_conductivity_func(temperature)

    def get_temperature(self, wv: SolidWallConservedVars,
            tseed: DOFArray = None, niter: int = 3) -> DOFArray:
        """Evaluate the temperature based on the energy."""
        if tseed is not None:
            temp = tseed*1.0
            for _ in range(0, niter):
                h = self.enthalpy(temp)
                cp = self.heat_capacity(temp)
                temp = temp - (h - wv.energy/wv.mass)/cp
            return temp

        return wv.energy/(self.density()*self.heat_capacity())

    def dependent_vars(self, wv: SolidWallConservedVars,
            tseed: DOFArray = None, niter: int = 3) -> SolidWallDependentVars:
        """Return solid wall dependent variables."""
        temperature = self.get_temperature(wv, tseed, niter)
        kappa = self.thermal_conductivity(temperature)
        return SolidWallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)


@dataclass_array_container
@dataclass(frozen=True, eq=False)
class PorousWallVars:
    """Variables for the porous material.

    It combines conserved variables, such as material_densities, with those
    that depend on the current status of the wall, which is uniquely defined
    based on the material densities.

    .. attribute:: material_densities
    .. attribute:: tau
    .. attribute:: void_fraction
    .. attribute:: permeability
    .. attribute:: tortuosity
    .. attribute:: density
    """

    material_densities: Union[DOFArray, np.ndarray]
    tau: DOFArray
    void_fraction: DOFArray
    permeability: Union[DOFArray, np.ndarray]
    tortuosity: DOFArray
    density: DOFArray


class PorousWallEOS:
    """Abstract interface for porous media domains.

    The properties are material-dependent and specified in individual files
    that inherit this class.

    .. automethod:: void_fraction
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: volume_fraction
    .. automethod:: permeability
    .. automethod:: emissivity
    .. automethod:: tortuosity
    .. automethod:: decomposition_progress
    """

    @abstractmethod
    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Void fraction $\epsilon$ filled by gas around the solid."""
        raise NotImplementedError()

    @abstractmethod
    def enthalpy(self, temperature: DOFArray,
                 tau: Optional[DOFArray]) -> DOFArray:
        r"""Evaluate the enthalpy $h_s$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def heat_capacity(self, temperature: DOFArray,
                      tau: Optional[DOFArray]) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def thermal_conductivity(self, temperature: DOFArray,
                             tau: Optional[DOFArray]) -> DOFArray:
        r"""Evaluate the thermal conductivity $\kappa$ of the solid."""
        raise NotImplementedError()

    @abstractmethod
    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        raise NotImplementedError()

    @abstractmethod
    def permeability(self, tau: DOFArray) -> Union[np.ndarray, DOFArray]:
        r"""Permeability $K$ of the porous material."""
        raise NotImplementedError()

    @abstractmethod
    def emissivity(self, temperature: Optional[DOFArray] = None,
            tau: Optional[DOFArray] = None) -> DOFArray:
        """Emissivity for energy radiation."""
        raise NotImplementedError()

    @abstractmethod
    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ of the porous material."""
        raise NotImplementedError()

    @abstractmethod
    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition."""
        raise NotImplementedError()


# FIXME: Generalize TransportModel interface to accept state variables
# other than fluid cv following
# https://github.com/illinois-ceesd/mirgecom/pull/935#discussion_r1298730910
class PorousWallTransport:
    r"""Transport model for porous media flow.

    Takes any gas-only transport model and modifies it to consider the
    interaction with the porous materials.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    """

    def __init__(self, base_transport: TransportModel):
        """Initialize base transport model for fluid-only."""
        self.base_transport = base_transport

    def bulk_viscosity(self, cv: ConservedVars, dv: GasDependentVars,
            wv: PorousWallVars, eos: GasEOS) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$."""
        return self.base_transport.bulk_viscosity(cv, dv, eos)

    def volume_viscosity(self, cv: ConservedVars, dv: GasDependentVars,
            wv: PorousWallVars, eos: GasEOS) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$."""
        return (self.bulk_viscosity(cv, dv, wv, eos)
            - 2./3. * self.viscosity(cv, dv, wv, eos))

    def viscosity(self, cv: ConservedVars, dv: GasDependentVars,
            wv: PorousWallVars, eos: GasEOS) -> DOFArray:
        """Viscosity of the gas through the porous wall."""
        return 1.0/wv.void_fraction*(
            self.base_transport.viscosity(cv, dv, eos))

    def thermal_conductivity(self, cv: ConservedVars, dv: GasDependentVars,
            wv: PorousWallVars, eos: GasEOS,
            wall_eos: PorousWallEOS) -> Union[np.ndarray, DOFArray]:
        r"""Return the effective thermal conductivity of the gas+solid.

        It is a function of temperature and degradation progress. As the
        fibers are oxidized, they reduce their cross area and, consequently,
        their ability to conduct heat.

        It is evaluated using a mass-weighted average given by

        .. math::
            \frac{\rho_s \kappa_s + \rho_g \kappa_g}{\rho_s + \rho_g}
        """
        kappa_s = wall_eos.thermal_conductivity(dv.temperature, wv.tau)
        kappa_g = self.base_transport.thermal_conductivity(cv, dv, eos)
        return (wv.density*kappa_s + cv.mass*kappa_g)/(cv.mass + wv.density)

    def species_diffusivity(self, cv: ConservedVars, dv: GasDependentVars,
            wv: PorousWallVars, eos: GasEOS) -> DOFArray:
        """Mass diffusivity of gaseous species through the porous wall."""
        return 1.0/wv.tortuosity*(
            self.base_transport.species_diffusivity(cv, dv, eos))

    def transport_vars(self, cv: ConservedVars, dv: GasDependentVars,
            wv: PorousWallVars, eos: GasEOS,
            wall_eos: PorousWallEOS) -> GasTransportVars:
        r"""Compute the transport properties."""
        return GasTransportVars(
            bulk_viscosity=self.bulk_viscosity(cv, dv, wv, eos),
            viscosity=self.viscosity(cv, dv, wv, eos),
            thermal_conductivity=self.thermal_conductivity(cv, dv, wv, eos,
                                                           wall_eos),
            species_diffusivity=self.species_diffusivity(cv, dv, wv, eos)
        )


@dataclass(frozen=True)
class PorousFlowModel:
    """Main class of the porous media flow.

    It is the equivalent to :class:`~mirgecom.gas_model.GasModel` and wraps:

    .. attribute:: wall_eos

        The thermophysical properties of the wall material and its EOS.

    .. attribute:: eos

        The thermophysical properties of the gas and its EOS. For now, only
        mixtures are considered.

    .. attribute:: transport

        Transport class that governs how the gas flows through the porous
        media. This is accounted for in :class:`PorousWallTransport`

    .. attribute:: temperature_iteration

        Number of iterations for temperature evaluation using Newton's method.
        Defaults to 3 if not specified.

    It also include functions that combine the properties of the porous
    material and the gas that permeates, yielding the actual porous flow EOS:

    .. automethod:: solid_density
    .. automethod:: decomposition_progress
    .. automethod:: get_temperature
    .. automethod:: get_pressure
    .. automethod:: internal_energy
    .. automethod:: heat_capacity
    """

    eos: MixtureEOS
    wall_eos: PorousWallEOS
    transport: PorousWallTransport
    temperature_iteration: int = 3

    def solid_density(self, material_densities) -> DOFArray:
        r"""Return the solid density $\epsilon_s \rho_s$.

        The material density is relative to the entire control volume, and
        is not to be confused with the intrinsic density, hence the $\epsilon$
        dependence. It is computed as the sum of all N solid phases:

        .. math::
            \epsilon_s \rho_s = \sum_i^N \epsilon_i \rho_i
        """
        if isinstance(material_densities, DOFArray):
            return material_densities
        return sum(material_densities)

    def decomposition_progress(self, material_densities) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.
        """
        mass = self.solid_density(material_densities)
        return self.wall_eos.decomposition_progress(mass)

    def get_temperature(self, cv: ConservedVars, wv: PorousWallVars,
                        tseed: DOFArray) -> DOFArray:
        r"""Evaluate the temperature based on solid+gas properties.

        It uses the assumption of thermal equilibrium between solid and fluid.
        Newton iteration is used to get the temperature based on the internal
        energy/enthalpy and heat capacity for the bulk (solid+gas) material:

        .. math::
            T^{n+1} = T^n -
                \frac
                {\epsilon_g \rho_g e_g + \rho_s h_s - \rho e}
                {\epsilon_g \rho_g C_{v_g} + \epsilon_s \rho_s C_{p_s}}

        """
        if isinstance(tseed, DOFArray) is False:
            actx = cv.array_context
            temp = tseed + actx.np.zeros_like(wv.tau)
        else:
            temp = tseed*1.0

        internal_energy = cv.energy - 0.5/cv.mass*np.dot(cv.momentum, cv.momentum)

        for _ in range(0, self.temperature_iteration):
            eps_rho_e = self.internal_energy(cv, wv, temp)
            bulk_cp = self.heat_capacity(cv, wv, temp)
            temp = temp - (eps_rho_e - internal_energy)/bulk_cp

        return temp

    def get_pressure(self, cv: ConservedVars, wv: PorousWallVars,
                     temperature: DOFArray) -> DOFArray:
        r"""Return the pressure of the gas considering the void fraction.

        Since the density is evaluated based on the entire bulk material, i.e.,
        considering the void fraction, the pressure is evaluated as

        .. math::
            P = \frac{\epsilon \rho}{\epsilon} \frac{R}{M} T

        where $\epsilon \rho$ is stored in :class:`~mirgecom.fluid.ConservedVars`
        and $M$ is the molar mass of the mixture.
        """
        return self.eos.pressure(cv, temperature)/wv.void_fraction

    def internal_energy(self, cv: ConservedVars, wv: PorousWallVars,
                 temperature: DOFArray) -> DOFArray:
        r"""Return the internal energy of the gas+solid material.

        .. math::
            \rho e = \epsilon_s \rho_s e_s + \epsilon_g \rho_g e_g
        """
        return (cv.mass*self.eos.get_internal_energy(temperature,
                                                     cv.species_mass_fractions)
                + wv.density*self.wall_eos.enthalpy(temperature, wv.tau))

    def heat_capacity(self, cv: ConservedVars, wv: PorousWallVars,
                 temperature: DOFArray) -> DOFArray:
        r"""Return the heat capacity of the gas+solid material.

        .. math::
            \rho C_v = \epsilon_s \rho_s {C_p}_s + \epsilon_g \rho_g {C_v}_g
        """
        return (cv.mass*self.eos.heat_capacity_cv(cv, temperature)
                + wv.density*self.wall_eos.heat_capacity(temperature, wv.tau))
