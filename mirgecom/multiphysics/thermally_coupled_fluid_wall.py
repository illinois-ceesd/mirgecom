r""":mod:`mirgecom.multiphysics.thermally_coupled_fluid_wall` for thermally-coupled
fluid and wall.

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
(:mod:`mirgecom.navierstokes`) with a wall subdomain governed by the heat
equation (:mod:`mirgecom.diffusion`) through temperature and heat flux. This
coupling can optionally include a sink term representing emitted radiation.
In the non-radiating case, coupling enforces continuity of temperature and heat flux

.. math::
    T_\text{wall} &= T_\text{fluid} \\
    -\kappa_\text{wall} \nabla T_\text{wall} \cdot \hat{n} &=
        -\kappa_\text{fluid} \nabla T_\text{fluid} \cdot \hat{n},

and in the radiating case, coupling enforces a similar condition but with an
additional radiation sink term in the heat flux

.. math::
    -\kappa_\text{wall} \nabla T_\text{wall} \cdot \hat{n} =
        -\kappa_\text{fluid} \nabla T_\text{fluid} \cdot \hat{n}
        + \epsilon \sigma (T^4 - T_\text{ambient}^4).

Boundary Setup Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: add_interface_boundaries_no_grad
.. autofunction:: add_interface_boundaries

Basic Coupled Operators
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: basic_coupled_ns_heat_operator

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: InterfaceFluidBoundary
.. autoclass:: InterfaceFluidSlipBoundary
.. autoclass:: InterfaceFluidNoslipBoundary
.. autoclass:: InterfaceWallBoundary
.. autoclass:: InterfaceWallRadiationBoundary
"""

__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
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

from dataclasses import dataclass, replace
import numpy as np
from abc import abstractmethod
from functools import partial

from arraycontext import dataclass_array_container
from meshmode.dof_array import DOFArray
from grudge.trace_pair import (
    TracePair,
)
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    as_dofdesc,
)
import grudge.geometry as geo
import grudge.op as op

from mirgecom.math import harmonic_mean
from mirgecom.boundary import (
    MengaldoBoundaryCondition,
    _SlipBoundaryComponent,
    _NoSlipBoundaryComponent,
    _ImpermeableBoundaryComponent,
    IsothermalSlipWallBoundary,
    IsothermalWallBoundary)
from mirgecom.flux import num_flux_central
from mirgecom.viscous import viscous_facial_flux_harmonic
from mirgecom.gas_model import (
    replace_fluid_state,
    make_operator_fluid_states,
)
from mirgecom.navierstokes import (
    grad_t_operator as fluid_grad_t_operator,
    ns_operator,
)
from mirgecom.diffusion import (
    grad_facial_flux_weighted,
    diffusion_flux,
    diffusion_facial_flux_harmonic,
    DiffusionBoundary,
    grad_operator as wall_grad_t_operator,
    diffusion_operator,
)
from mirgecom.multiphysics import make_interface_boundaries
from mirgecom.utils import project_from_base


class _ThermalDataNoGradInterVolTag:
    pass


class _ThermalDataInterVolTag:
    pass


class _GradTemperatureInterVolTag:
    pass


class _FluidOpStatesTag:
    pass


class _FluidGradTag:
    pass


class _FluidOperatorTag:
    pass


class _WallGradTag:
    pass


class _WallOperatorTag:
    pass


# FIXME: Interior penalty should probably use an average of the lengthscales on
# both sides of the interface
class InterfaceFluidBoundary(MengaldoBoundaryCondition):
    r"""
    Abstract interface for the fluid side of the fluid-wall interface.

    Extends :class:`~mirgecom.boundary.MengaldoBoundaryCondition` to include
    an interior penalty on the heat flux:

    .. math::
        q_\text{penalty} = \tau (T^+ - T^-).

    where $\tau = c \frac{\kappa_\text{bc}}{h^-}$. Here $c$ is a
    user-defined constant and $h^-$ is the characteristic mesh spacing
    on the fluid side of the interface.

    Base class implementations
    --------------------------
    .. automethod:: __init__
    .. automethod:: viscous_divergence_flux

    Abstract interface
    ------------------
    .. automethod:: temperature_plus
    """
    def __init__(self, heat_flux_penalty_amount, lengthscales_minus):
        r"""
        Initialize InterfaceFluidBoundary.

        Parameters
        ----------
        heat_flux_penalty_amount: float

            Coefficient $c$ for the interior penalty on the heat flux.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray`

            Characteristic mesh spacing $h^-$.
        """
        self._penalty_amount = heat_flux_penalty_amount
        self._lengthscales_minus = lengthscales_minus

    @abstractmethod
    def temperature_plus(self, dcoll, dd_bdry, state_minus, **kwargs):
        r"""Get the external temperature, $T^+$.

        Parameters
        ----------
        dcoll: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary.

        Returns
        -------
        :class:`meshmode.dof_array.DOFArray`
        """

    def viscous_divergence_flux(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, numerical_flux_func=viscous_facial_flux_harmonic,
            **kwargs):
        """
        Return the viscous flux as defined by
        :meth:`mirgecom.boundary.MengaldoBoundaryCondition.viscous_divergence_flux`
        with the additional heat flux interior penalty term.
        """
        dd_bdry = as_dofdesc(dd_bdry)

        state_bc = self.state_bc(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

        flux_without_penalty = super().viscous_divergence_flux(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, numerical_flux_func=numerical_flux_func,
            grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus, **kwargs)

        lengthscales_minus = project_from_base(
            dcoll, dd_bdry, self._lengthscales_minus)

        if isinstance(state_bc.thermal_conductivity, np.ndarray):
            # orthotropic materials
            actx = state_minus.array_context
            normal = geo.normal(actx, dcoll, dd_bdry)
            kappa_bc = np.dot(normal, state_bc.thermal_conductivity*normal)
        else:
            kappa_bc = state_bc.thermal_conductivity

        tau = self._penalty_amount * kappa_bc / lengthscales_minus

        t_minus = state_minus.temperature
        t_plus = self.temperature_plus(
            dcoll, dd_bdry=dd_bdry, state_minus=state_minus, **kwargs)

        return replace(
            flux_without_penalty,
            # NS and diffusion use opposite sign conventions for flux; hence penalty
            # is added here instead of subtracted
            energy=flux_without_penalty.energy + tau * (t_plus - t_minus))


class _ThermallyCoupledHarmonicMeanBoundaryComponent:
    def __init__(
            self, kappa_plus, t_plus, grad_t_plus=None):
        self._kappa_plus = kappa_plus
        self._t_plus = t_plus
        self._grad_t_plus = grad_t_plus

    def _normal_kappa_plus(self, dcoll, dd_bdry):
        # project kappa plus in case of overintegration
        if isinstance(self._kappa_plus, np.ndarray):
            # orthotropic materials
            actx = self._t_plus.array_context
            normal = geo.normal(actx, dcoll, dd_bdry)
            kappa_plus = project_from_base(dcoll, dd_bdry, self._kappa_plus)
            return np.dot(normal, kappa_plus*normal)
        return project_from_base(dcoll, dd_bdry, self._kappa_plus)

    def _normal_kappa_minus(self, dcoll, dd_bdry, kappa):
        # state minus is already in the quadrature domain
        if isinstance(kappa, np.ndarray):
            # orthotropic materials
            actx = self._t_plus.array_context
            normal = geo.normal(actx, dcoll, dd_bdry)
            return np.dot(normal, kappa*normal)
        return kappa

    def kappa_bc(self, dcoll, dd_bdry, kappa_minus):
        return harmonic_mean(kappa_minus,
                             project_from_base(dcoll, dd_bdry, self._kappa_plus))

    def temperature_plus(self, dcoll, dd_bdry):
        return project_from_base(dcoll, dd_bdry, self._t_plus)

    def temperature_bc(self, dcoll, dd_bdry, kappa_minus, t_minus):
        t_plus = project_from_base(dcoll, dd_bdry, self._t_plus)
        actx = t_minus.array_context
        kappa_plus = self._normal_kappa_plus(dcoll, dd_bdry)
        kappa_minus = self._normal_kappa_minus(dcoll, dd_bdry,
                                               kappa_minus + t_minus*0.0)
        kappa_sum = actx.np.where(
            actx.np.greater(kappa_minus + kappa_plus, 0*kappa_minus),
            kappa_minus + kappa_plus,
            0*kappa_minus + 1)
        return (t_minus * kappa_minus + t_plus * kappa_plus)/kappa_sum

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus):
        if self._grad_t_plus is None:
            raise TypeError(
                "Boundary does not have external temperature gradient data.")
        grad_t_plus = project_from_base(dcoll, dd_bdry, self._grad_t_plus)
        return (grad_t_plus + grad_t_minus)/2


def _replace_kappa(state, kappa):
    """Replace the thermal conductivity in fluid state *state* with *kappa*."""
    new_tv = replace(state.tv, thermal_conductivity=kappa)
    return replace(state, tv=new_tv)


class InterfaceFluidSlipBoundary(InterfaceFluidBoundary):
    """
    Boundary for the fluid side of the fluid-wall interface, with slip.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_plus
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(
            self, kappa_plus, t_plus, grad_t_plus=None,
            heat_flux_penalty_amount=None, lengthscales_minus=None):
        r"""
        Initialize InterfaceFluidSlipBoundary.

        Arguments *grad_t_plus*, *heat_flux_penalty_amount*, and
        *lengthscales_minus* are only required if the boundary will be used to
        compute the viscous flux.

        Parameters
        ----------
        kappa_plus: float or :class:`meshmode.dof_array.DOFArray`

            Thermal conductivity from the wall side.

        t_plus: :class:`meshmode.dof_array.DOFArray`

            Temperature from the wall side.

        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None

            Temperature gradient from the wall side.

        heat_flux_penalty_amount: float or None

            Coefficient $c$ for the interior penalty on the heat flux.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None

            Characteristic mesh spacing $h^-$.
        """
        InterfaceFluidBoundary.__init__(
            self,
            heat_flux_penalty_amount=heat_flux_penalty_amount,
            lengthscales_minus=lengthscales_minus)

        self._thermally_coupled = _ThermallyCoupledHarmonicMeanBoundaryComponent(
            kappa_plus=kappa_plus,
            t_plus=t_plus,
            grad_t_plus=grad_t_plus)
        self._slip = _SlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        actx = state_minus.array_context

        # Grab a unit normal to the boundary
        nhat = geo.normal(actx, dcoll, dd_bdry)

        # Reflect the normal momentum
        mom_plus = self._slip.momentum_plus(state_minus.momentum_density, nhat)

        # Don't bother replacing kappa since this is just for inviscid
        return replace_fluid_state(
            state_minus, gas_model, momentum=mom_plus, outline=True)

    def state_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        actx = state_minus.array_context

        cv_minus = state_minus.cv

        kappa_minus = state_minus.tv.thermal_conductivity

        # Grab a unit normal to the boundary
        nhat = geo.normal(actx, dcoll, dd_bdry)

        # set the normal momentum to 0
        mom_bc = self._slip.momentum_bc(state_minus.momentum_density, nhat)

        t_bc = self._thermally_coupled.temperature_bc(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

        internal_energy_bc = (
            cv_minus.mass
            * gas_model.eos.get_internal_energy(
                temperature=t_bc,
                species_mass_fractions=cv_minus.species_mass_fractions))
        total_energy_bc = (
            internal_energy_bc
            + 0.5*np.dot(mom_bc, mom_bc)/cv_minus.mass)

        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

        return _replace_kappa(
            replace_fluid_state(
                state_minus, gas_model,
                energy=total_energy_bc,
                momentum=mom_bc,
                temperature_seed=t_bc,
                outline=True),
            kappa_bc)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            normal, **kwargs):  # noqa: D102
        dd_bdry = as_dofdesc(dd_bdry)
        state_bc = self.state_bc(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

        grad_v_bc = self._slip.grad_velocity_bc(
            state_minus, state_bc, grad_cv_minus, normal)

        grad_mom_bc = (
            state_bc.mass_density * grad_v_bc
            + np.outer(state_bc.velocity, grad_cv_minus.mass))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return grad_cv_minus.replace(
            momentum=grad_mom_bc,
            species_mass=grad_species_mass_bc)

    def temperature_plus(
            self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
        return self._thermally_coupled.temperature_plus(dcoll, dd_bdry)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
        kappa_minus = state_minus.tv.thermal_conductivity
        return self._thermally_coupled.temperature_bc(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):  # noqa: D102
        return self._thermally_coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)


class InterfaceFluidNoslipBoundary(InterfaceFluidBoundary):
    """
    Boundary for the fluid side of the fluid-wall interface, without slip.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_plus
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(
            self, kappa_plus, t_plus, grad_t_plus=None,
            heat_flux_penalty_amount=None, lengthscales_minus=None):
        r"""
        Initialize InterfaceFluidNoslipBoundary.

        Arguments *grad_t_plus*, *heat_flux_penalty_amount*, and
        *lengthscales_minus* are only required if the boundary will be used to
        compute the viscous flux.

        Parameters
        ----------
        kappa_plus: float or :class:`meshmode.dof_array.DOFArray`

            Thermal conductivity from the wall side.

        t_plus: :class:`meshmode.dof_array.DOFArray`

            Temperature from the wall side.

        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None

            Temperature gradient from the wall side.

        heat_flux_penalty_amount: float or None

            Coefficient $c$ for the interior penalty on the heat flux.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None

            Characteristic mesh spacing $h^-$.
        """
        InterfaceFluidBoundary.__init__(
            self,
            heat_flux_penalty_amount=heat_flux_penalty_amount,
            lengthscales_minus=lengthscales_minus)

        self._thermally_coupled = _ThermallyCoupledHarmonicMeanBoundaryComponent(
            kappa_plus=kappa_plus,
            t_plus=t_plus,
            grad_t_plus=grad_t_plus)
        self._no_slip = _NoSlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        dd_bdry = as_dofdesc(dd_bdry)
        mom_plus = self._no_slip.momentum_plus(state_minus.momentum_density)

        # Don't bother replacing kappa since this is just for inviscid
        return replace_fluid_state(
            state_minus, gas_model, momentum=mom_plus, outline=True)

    def state_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        dd_bdry = as_dofdesc(dd_bdry)

        kappa_minus = state_minus.tv.thermal_conductivity

        mom_bc = self._no_slip.momentum_bc(state_minus.momentum_density)

        t_bc = self._thermally_coupled.temperature_bc(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc,
            species_mass_fractions=state_minus.species_mass_fractions)

        # Velocity is pinned to 0 here, no kinetic energy
        total_energy_bc = state_minus.mass_density*internal_energy_bc

        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

        return _replace_kappa(
            replace_fluid_state(
                state_minus, gas_model,
                energy=total_energy_bc,
                momentum=mom_bc,
                outline=True),
            kappa_bc)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus, normal,
            **kwargs):  # noqa: D102
        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return grad_cv_minus.replace(species_mass=grad_species_mass_bc)

    def temperature_plus(
            self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
        return self._thermally_coupled.temperature_plus(dcoll, dd_bdry)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
        kappa_minus = state_minus.tv.thermal_conductivity
        return self._thermally_coupled.temperature_bc(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):  # noqa: D102
        return self._thermally_coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)


# FIXME: Interior penalty should probably use an average of the lengthscales on
# both sides of the interface
class InterfaceWallBoundary(DiffusionBoundary):
    """
    Boundary for the wall side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(self, kappa_plus, u_plus, grad_u_plus=None):
        r"""
        Initialize InterfaceWallBoundary.

        Argument *grad_u_plus* is only required if the boundary will be used to
        compute the heat flux.

        Parameters
        ----------
        kappa_plus: float or :class:`meshmode.dof_array.DOFArray`

            Thermal conductivity from the fluid side.

        u_plus: :class:`meshmode.dof_array.DOFArray`

            Temperature from the fluid side.

        grad_u_plus: :class:`meshmode.dof_array.DOFArray` or None

            Temperature gradient from the fluid side.
        """
        self.kappa_plus = kappa_plus
        self.u_plus = u_plus
        self.grad_u_plus = grad_u_plus

    def get_grad_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):  # noqa: D102
        actx = u_minus.array_context
        normal = geo.normal(actx, dcoll, dd_bdry)

        kappa_plus = project_from_base(dcoll, dd_bdry, self.kappa_plus)

        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_plus)

        u_plus = project_from_base(dcoll, dd_bdry, self.u_plus)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)

        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):  # noqa: D102
        if self.grad_u_plus is None:
            raise TypeError(
                "Boundary does not have external gradient data.")

        actx = u_minus.array_context
        normal = geo.normal(actx, dcoll, dd_bdry)

        kappa_plus = project_from_base(dcoll, dd_bdry, self.kappa_plus)

        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_plus)

        u_plus = project_from_base(dcoll, dd_bdry, self.u_plus)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)

        grad_u_plus = project_from_base(dcoll, dd_bdry, self.grad_u_plus)
        grad_u_tpair = TracePair(
            dd_bdry, interior=grad_u_minus, exterior=grad_u_plus)

        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)

        return numerical_flux_func(
            kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal,
            penalty_amount=penalty_amount)


class InterfaceWallRadiationBoundary(DiffusionBoundary):
    r"""
    Boundary for the wall side of the fluid-wall interface (radiating).

    Enforces the heat flux to be that entering the fluid side plus a radiation sink
    term:

    .. math::
        -\kappa_\text{wall} \nabla T_\text{wall} \cdot \hat{n} =
            -\kappa_\text{fluid} \nabla T_\text{fluid} \cdot \hat{n}
            + \epsilon \sigma (T^4 - T_\text{ambient}^4),

    where $\epsilon$ is the wall material's emissivity and $\sigma$ is the
    Stefan-Boltzmann constant.

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(
            self, kappa_plus, grad_u_plus=None, emissivity=None, sigma=None,
            u_ambient=None):
        r"""
        Initialize InterfaceWallRadiationBoundary.

        Arguments *grad_u_plus*, *emissivity*, *sigma*, and *u_ambient* are only
        required if the boundary will be used to compute the heat flux.

        Parameters
        ----------
        kappa_plus: float or :class:`meshmode.dof_array.DOFArray`

            Thermal conductivity from the fluid side.

        grad_u_plus: :class:`meshmode.dof_array.DOFArray` or None

            Temperature gradient from the fluid side.

        emissivity: float or :class:`meshmode.dof_array.DOFArray` or None

            Emissivity of the wall material.

        sigma: float or None

            Stefan-Boltzmann constant.

        u_ambient: :class:`meshmode.dof_array.DOFArray` or None

            Ambient temperature of the environment.
        """
        self.kappa_plus = kappa_plus
        self.emissivity = emissivity
        self.sigma = sigma
        self.u_ambient = u_ambient
        self.grad_u_plus = grad_u_plus

    def get_grad_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):  # noqa: D102
        actx = u_minus.array_context
        normal = geo.normal(actx, dcoll, dd_bdry)

        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_minus)

        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):  # noqa: D102
        if self.grad_u_plus is None:
            raise TypeError("External temperature gradient is not specified.")
        if self.emissivity is None:
            raise TypeError("Wall emissivity is not specified.")
        if self.sigma is None:
            raise TypeError("Stefan-Boltzmann constant value is not specified.")
        if self.u_ambient is None:
            raise TypeError("Ambient temperature is not specified.")

        actx = u_minus.array_context
        normal = geo.normal(actx, dcoll, dd_bdry)

        kappa_plus = project_from_base(dcoll, dd_bdry, self.kappa_plus)
        grad_u_plus = project_from_base(dcoll, dd_bdry, self.grad_u_plus)
        emissivity = project_from_base(dcoll, dd_bdry, self.emissivity)
        u_ambient = project_from_base(dcoll, dd_bdry, self.u_ambient)

        # Note: numerical_flux_func is ignored
        return (
            np.dot(diffusion_flux(kappa_plus, grad_u_plus), normal)
            + emissivity * self.sigma * (u_minus**4 - u_ambient**4))


@dataclass_array_container
@dataclass(frozen=True)
class _ThermalDataNoGrad:
    kappa: DOFArray
    temperature: DOFArray


@dataclass_array_container
@dataclass(frozen=True)
class _ThermalData(_ThermalDataNoGrad):
    grad_temperature: np.ndarray


def _make_thermal_data(kappa, temperature, grad_temperature=None):
    if not isinstance(kappa, DOFArray):
        kappa = kappa * (0*temperature + 1)

    if grad_temperature is not None:
        thermal_data = _ThermalData(kappa, temperature, grad_temperature)
    else:
        thermal_data = _ThermalDataNoGrad(kappa, temperature)

    return thermal_data


def _get_interface_trace_pairs_no_grad(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        *,
        comm_tag=None):
    pairwise_thermal_data = {
        (fluid_dd, wall_dd): (
            _make_thermal_data(fluid_kappa, fluid_temperature),
            _make_thermal_data(wall_kappa, wall_temperature))}

    from grudge.trace_pair import inter_volume_trace_pairs  \
        # pylint: disable=no-name-in-module

    return inter_volume_trace_pairs(
        dcoll, pairwise_thermal_data,
        comm_tag=(_ThermalDataNoGradInterVolTag, comm_tag))


def _get_interface_trace_pairs(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        fluid_grad_temperature, wall_grad_temperature,
        *,
        comm_tag=None):
    pairwise_thermal_data = {
        (fluid_dd, wall_dd): (
            _make_thermal_data(
                fluid_kappa,
                fluid_temperature,
                fluid_grad_temperature),
            _make_thermal_data(
                wall_kappa,
                wall_temperature,
                wall_grad_temperature))}

    from grudge.trace_pair import inter_volume_trace_pairs  \
        # pylint: disable=no-name-in-module

    return inter_volume_trace_pairs(
        dcoll, pairwise_thermal_data,
        comm_tag=(_ThermalDataInterVolTag, comm_tag))


def _get_interface_boundaries_no_grad(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        *,
        interface_noslip=True,
        interface_radiation=False,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    interface_tpairs = _get_interface_trace_pairs_no_grad(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        comm_tag=comm_tag)

    if interface_radiation:
        def make_fluid_boundary(interface_tpair):
            if interface_noslip:
                fluid_bc_class = IsothermalWallBoundary
            else:
                fluid_bc_class = IsothermalSlipWallBoundary
            return fluid_bc_class(interface_tpair.ext.temperature)

        def make_wall_boundary(interface_tpair):
            return InterfaceWallRadiationBoundary(
                interface_tpair.ext.kappa)

    else:
        def make_fluid_boundary(interface_tpair):
            if interface_noslip:
                fluid_bc_class = InterfaceFluidNoslipBoundary
            else:
                fluid_bc_class = InterfaceFluidSlipBoundary
            return fluid_bc_class(
                interface_tpair.ext.kappa,
                interface_tpair.ext.temperature)

        def make_wall_boundary(interface_tpair):
            return InterfaceWallBoundary(
                interface_tpair.ext.kappa,
                interface_tpair.ext.temperature)

    bdry_factories = {
        (wall_dd, fluid_dd): make_fluid_boundary,
        (fluid_dd, wall_dd): make_wall_boundary}

    interface_boundaries = make_interface_boundaries(
        bdry_factories, interface_tpairs)

    fluid_interface_boundaries = interface_boundaries[wall_dd, fluid_dd]
    wall_interface_boundaries = interface_boundaries[fluid_dd, wall_dd]

    return fluid_interface_boundaries, wall_interface_boundaries


def _get_interface_boundaries(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        fluid_grad_temperature, wall_grad_temperature,
        *,
        interface_noslip=True,
        interface_radiation=False,
        wall_emissivity=None,
        sigma=None,
        ambient_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    if wall_penalty_amount is None:
        # FIXME: After verifying the form of the penalty term, figure out what value
        # makes sense to use as a default here
        wall_penalty_amount = 0.05

    interface_tpairs = _get_interface_trace_pairs(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        fluid_grad_temperature, wall_grad_temperature,
        comm_tag=comm_tag)

    if interface_radiation:
        def make_fluid_boundary(interface_tpair):
            if interface_noslip:
                fluid_bc_class = IsothermalWallBoundary
            else:
                fluid_bc_class = IsothermalSlipWallBoundary
            return fluid_bc_class(interface_tpair.ext.temperature)

        if (
                wall_emissivity is None
                or sigma is None
                or ambient_temperature is None):
            raise TypeError(
                "Arguments 'wall_emissivity', 'sigma' and 'ambient_temperature'"
                "are required if using surface radiation.")

        def make_wall_boundary(interface_tpair):
            emissivity_minus = op.project(dcoll, wall_dd, interface_tpair.dd,
                                          wall_emissivity)
            return InterfaceWallRadiationBoundary(
                interface_tpair.ext.kappa,
                interface_tpair.ext.grad_temperature,
                emissivity_minus, sigma,
                ambient_temperature)

    else:
        # Diffusion operator passes lengthscales_minus into the boundary flux
        # functions, but NS doesn't; thus we need to pass lengthscales into
        # the fluid boundary condition constructor
        from grudge.dt_utils import characteristic_lengthscales
        fluid_lengthscales = (
            characteristic_lengthscales(
                fluid_temperature.array_context, dcoll, fluid_dd)
            * (0*fluid_temperature+1))

        def make_fluid_boundary(interface_tpair):
            if interface_noslip:
                fluid_bc_class = InterfaceFluidNoslipBoundary
            else:
                fluid_bc_class = InterfaceFluidSlipBoundary
            return fluid_bc_class(
                interface_tpair.ext.kappa,
                interface_tpair.ext.temperature,
                interface_tpair.ext.grad_temperature,
                heat_flux_penalty_amount=wall_penalty_amount,
                lengthscales_minus=op.project(dcoll,
                    fluid_dd, interface_tpair.dd, fluid_lengthscales))

        def make_wall_boundary(interface_tpair):
            return InterfaceWallBoundary(
                interface_tpair.ext.kappa,
                interface_tpair.ext.temperature,
                interface_tpair.ext.grad_temperature)

    bdry_factories = {
        (wall_dd, fluid_dd): make_fluid_boundary,
        (fluid_dd, wall_dd): make_wall_boundary}

    interface_boundaries = make_interface_boundaries(
        bdry_factories, interface_tpairs)

    fluid_interface_boundaries = interface_boundaries[wall_dd, fluid_dd]
    wall_interface_boundaries = interface_boundaries[fluid_dd, wall_dd]

    return fluid_interface_boundaries, wall_interface_boundaries


def add_interface_boundaries_no_grad(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_kappa, wall_temperature,
        fluid_boundaries, wall_boundaries,
        *,
        interface_noslip=True,
        interface_radiation=False,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    """
    Include the fluid-wall interface boundaries (without temperature gradient).

    Return a tuple `(fluid_all_boundaries, wall_all_boundaries)` that adds boundaries
    to *fluid_boundaries* and *wall_boundaries* that represent the volume interfaces.
    One entry is added for the collection of faces whose opposite face reside on the
    current MPI rank and one-per-rank for each collection of faces whose opposite
    face resides on a different rank.

    Parameters
    ----------

    dcoll: class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    fluid_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the fluid volume.

    wall_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the wall volume.

    fluid_state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state and dependent
        quantities for the fluid volume.

    wall_kappa: float or :class:`meshmode.dof_array.DOFArray`

        Thermal conductivity for the wall volume.

    wall_temperature: :class:`meshmode.dof_array.DOFArray`

        Temperature for the wall volume.

    fluid_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the fluid subdomain.

    wall_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the wall subdomain.

    interface_noslip: bool

        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    interface_radiation: bool

        If `True`, interface includes a radiation sink term in the heat flux. See
        :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall.InterfaceWallRadiationBoundary`
        for details.

    quadrature_tag

        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    comm_tag: Hashable
        Tag for distributed communication
    """
    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        _get_interface_boundaries_no_grad(
            dcoll,
            fluid_dd, wall_dd,
            fluid_state.tv.thermal_conductivity, wall_kappa,
            fluid_state.temperature, wall_temperature,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            quadrature_tag=quadrature_tag,
            comm_tag=comm_tag)

    fluid_all_boundaries_no_grad = {}
    fluid_all_boundaries_no_grad.update(fluid_boundaries)
    fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    wall_all_boundaries_no_grad = {}
    wall_all_boundaries_no_grad.update(wall_boundaries)
    wall_all_boundaries_no_grad.update(wall_interface_boundaries_no_grad)

    return fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad


def add_interface_boundaries(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_kappa, wall_temperature,
        fluid_grad_temperature, wall_grad_temperature,
        fluid_boundaries, wall_boundaries,
        *,
        interface_noslip=True,
        interface_radiation=False,
        wall_emissivity=None,
        sigma=None,
        ambient_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    """
    Include the fluid-wall interface boundaries.

    Return a tuple `(fluid_all_boundaries, wall_all_boundaries)` that adds boundaries
    to *fluid_boundaries* and *wall_boundaries* that represent the volume interfaces.
    One entry is added for the collection of faces whose opposite face reside on the
    current MPI rank and one-per-rank for each collection of faces whose opposite
    face resides on a different rank.

    Parameters
    ----------

    dcoll: class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    fluid_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the fluid volume.

    wall_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the wall volume.

    fluid_state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state and dependent
        quantities for the fluid volume.

    wall_kappa: float or :class:`meshmode.dof_array.DOFArray`

        Thermal conductivity for the wall volume.

    wall_temperature: :class:`meshmode.dof_array.DOFArray`

        Temperature for the wall volume.

    fluid_grad_temperature: numpy.ndarray

        Temperature gradient for the fluid volume.

    wall_grad_temperature: numpy.ndarray

        Temperature gradient for the wall volume.

    fluid_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the fluid subdomain.

    wall_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the wall subdomain.

    interface_noslip: bool

        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    interface_radiation: bool

        If `True`, interface includes a radiation sink term in the heat flux. See
        :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall.InterfaceWallRadiationBoundary`
        for details. Additional arguments *wall_emissivity*, *sigma*, and
        *ambient_temperature* are required if enabled.

    wall_emissivity: float or :class:`meshmode.dof_array.DOFArray`

        Emissivity of the wall material.

    sigma: float

        Stefan-Boltzmann constant.

    ambient_temperature: :class:`meshmode.dof_array.DOFArray`

        Ambient temperature of the environment.

    wall_penalty_amount: float

        Coefficient $c$ for the interior penalty on the heat flux. See
        :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall.InterfaceFluidBoundary`
        for details.

    quadrature_tag

        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    comm_tag: Hashable
        Tag for distributed communication
    """
    fluid_interface_boundaries, wall_interface_boundaries = \
        _get_interface_boundaries(
            dcoll,
            fluid_dd, wall_dd,
            fluid_state.tv.thermal_conductivity, wall_kappa,
            fluid_state.temperature, wall_temperature,
            fluid_grad_temperature, wall_grad_temperature,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            wall_emissivity=wall_emissivity,
            sigma=sigma,
            ambient_temperature=ambient_temperature,
            wall_penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag,
            comm_tag=comm_tag)

    fluid_all_boundaries = {}
    fluid_all_boundaries.update(fluid_boundaries)
    fluid_all_boundaries.update(fluid_interface_boundaries)

    wall_all_boundaries = {}
    wall_all_boundaries.update(wall_boundaries)
    wall_all_boundaries.update(wall_interface_boundaries)

    return fluid_all_boundaries, wall_all_boundaries


def coupled_grad_t_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        *,
        time=0.,
        interface_noslip=True,
        interface_radiation=False,
        use_kappa_weighted_grad_flux_in_fluid=None,
        quadrature_tag=DISCR_TAG_BASE,
        fluid_numerical_flux_func=num_flux_central,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _fluid_operator_states_quad=None,
        _fluid_all_boundaries_no_grad=None,
        _wall_all_boundaries_no_grad=None):
    r"""
    Compute $\nabla T$ on the fluid and wall subdomains.

    Deprecated; set up interface boundaries explicitly via
    :func:`add_interface_boundaries_no_grad` and include them when calling the
    individual operators instead.

    Parameters
    ----------

    dcoll: class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    fluid_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the fluid volume.

    wall_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the wall volume.

    fluid_boundaries:

        Dictionary of boundary objects for the fluid subdomain, one for each
        :class:`~grudge.dof_desc.BoundaryDomainTag` that represents a domain
        boundary.

    wall_boundaries:

        Dictionary of boundary objects for the wall subdomain, one for each
        :class:`~grudge.dof_desc.BoundaryDomainTag` that represents a domain
        boundary.

    fluid_state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state and dependent
        quantities for the fluid volume.

    wall_kappa: float or :class:`meshmode.dof_array.DOFArray`

        Thermal conductivity for the wall volume.

    wall_temperature: :class:`meshmode.dof_array.DOFArray`

        Temperature for the wall volume.

    time:

        Time

    interface_noslip: bool

        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    use_kappa_weighted_grad_flux_in_fluid: bool

        Indicates whether the temperature gradient flux on the fluid side of the
        interface should be computed using a simple average of temperatures or by
        weighting the temperature from each side by its respective thermal
        conductivity.

    quadrature_tag:

        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    fluid_numerical_flux_func:

        Callable function to return the numerical flux to be used when computing
        the temperature gradient in the fluid subdomain. Defaults to
        :class:`~mirgecom.flux.num_flux_central`.

    Returns
    -------

        The tuple `(fluid_grad_temperature, wall_grad_temperature)`.
    """
    from warnings import warn
    warn(
        "coupled_grad_t_operator is deprecated and will disappear in Q3 2023. "
        "Set up interface boundaries explicitly via "
        ":func:`add_interface_boundaries_no_grad` and include them when calling the "
        "individual operators instead.", DeprecationWarning, stacklevel=2)

    if use_kappa_weighted_grad_flux_in_fluid is None:
        warn(
            "Default value of use_kappa_weighted_grad_flux_in_fluid has changed "
            "from False to True as False is no longer allowed. Explicitly set it "
            "to True to suppress this warning.", UserWarning, stacklevel=2)
        use_kappa_weighted_grad_flux_in_fluid = True
    elif not use_kappa_weighted_grad_flux_in_fluid:
        warn(
            "Setting use_kappa_weighted_grad_flux_in_fluid to True; False is no "
            "longer allowed. Explicitly set it to True to suppress this warning.",
            UserWarning, stacklevel=2)

    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    # Include boundaries for the fluid-wall interface; no temperature gradient
    # yet because that's what we're trying to compute

    assert (
        (_fluid_all_boundaries_no_grad is None)
        == (_wall_all_boundaries_no_grad is None)), (
        "Expected both _fluid_all_boundaries_no_grad and "
        "_wall_all_boundaries_no_grad or neither")

    if _fluid_all_boundaries_no_grad is None:
        fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad = \
            add_interface_boundaries_no_grad(
                dcoll,
                gas_model,
                fluid_dd, wall_dd,
                fluid_state, wall_kappa, wall_temperature,
                fluid_boundaries, wall_boundaries,
                interface_noslip=interface_noslip,
                quadrature_tag=quadrature_tag)
    else:
        fluid_all_boundaries_no_grad = _fluid_all_boundaries_no_grad
        wall_all_boundaries_no_grad = _wall_all_boundaries_no_grad

    # Compute the subdomain gradient operators using the augmented boundaries

    return (
        fluid_grad_t_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=fluid_dd,
            operator_states_quad=_fluid_operator_states_quad,
            comm_tag=_FluidGradTag),
        wall_grad_t_operator(
            dcoll, wall_kappa, wall_all_boundaries_no_grad, wall_temperature,
            quadrature_tag=quadrature_tag, dd=wall_dd, comm_tag=_WallGradTag))


def coupled_ns_heat_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        *,
        time=0.,
        interface_noslip=True,
        use_kappa_weighted_grad_flux_in_fluid=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        limiter_func=None,
        fluid_gradient_numerical_flux_func=num_flux_central,
        return_gradients=False,
        ns_operator=ns_operator):
    r"""
    Compute the RHS of the fluid and wall subdomains.

    Augments *fluid_boundaries* and *wall_boundaries* with the boundaries for the
    fluid-wall interface that are needed to enforce continuity of temperature and
    heat flux.

    Deprecated; set up interface boundaries explicitly via
    :func:`add_interface_boundaries` and include them when calling the individual
    operators instead.

    Parameters
    ----------

    dcoll: class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    fluid_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the fluid volume.

    wall_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the wall volume.

    fluid_boundaries:

        Dictionary of boundary objects for the fluid subdomain, one for each
        :class:`~grudge.dof_desc.BoundaryDomainTag` that represents a domain
        boundary.

    wall_boundaries:

        Dictionary of boundary objects for the wall subdomain, one for each
        :class:`~grudge.dof_desc.BoundaryDomainTag` that represents a domain
        boundary.

    fluid_state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state and dependent
        quantities for the fluid volume.

    wall_kappa: float or :class:`meshmode.dof_array.DOFArray`

        Thermal conductivity for the wall volume.

    wall_temperature: :class:`meshmode.dof_array.DOFArray`

        Temperature for the wall volume.

    time:

        Time

    interface_noslip: bool

        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    use_kappa_weighted_grad_flux_in_fluid: bool

        Indicates whether the temperature gradient flux on the fluid side of the
        interface should be computed using a simple average of temperatures or by
        weighting the temperature from each side by its respective thermal
        conductivity.

    wall_penalty_amount: float

        Coefficient $c$ for the interior penalty on the heat flux. See
        :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall.InterfaceFluidBoundary`
        for details.

    quadrature_tag:

        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    fluid_gradient_numerical_flux_func:

        Callable function to return the numerical flux to be used when computing
        the temperature gradient in the fluid subdomain. Defaults to
        :class:`~mirgecom.flux.num_flux_central`.

    limiter_func:

        Callable function to be passed to
        :func:`~mirgecom.gas_model.make_operator_fluid_states`
        that filters or limits the produced fluid states.  This is used to keep
        species mass fractions in physical and realizable states, for example.

    Returns
    -------

        The tuple `(fluid_rhs, wall_rhs)`.
    """
    from warnings import warn
    warn(
        "coupled_ns_heat_operator is deprecated and will disappear in Q3 2023. "
        "Set up interface boundaries explicitly via "
        ":func:`add_interface_boundaries` and include them when calling the "
        "individual operators instead.", DeprecationWarning, stacklevel=2)

    if use_kappa_weighted_grad_flux_in_fluid is None:
        warn(
            "Default value of use_kappa_weighted_grad_flux_in_fluid has changed "
            "from False to True as False is no longer allowed. Explicitly set it "
            "to True to suppress this warning.", UserWarning, stacklevel=2)
        use_kappa_weighted_grad_flux_in_fluid = True
    elif not use_kappa_weighted_grad_flux_in_fluid:
        warn(
            "Setting use_kappa_weighted_grad_flux_in_fluid to True; False is no "
            "longer allowed. Explicitly set it to True to suppress this warning.",
            UserWarning, stacklevel=2)

    if wall_penalty_amount is None:
        # FIXME: After verifying the form of the penalty term, figure out what value
        # makes sense to use as a default here
        wall_penalty_amount = 0.05

    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    # Include boundaries for the fluid-wall interface; no temperature gradient
    # yet because we need to compute it
    fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad = \
        add_interface_boundaries_no_grad(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_kappa, wall_temperature,
            fluid_boundaries, wall_boundaries,
            interface_noslip=interface_noslip,
            quadrature_tag=quadrature_tag)

    # Get the operator fluid states
    fluid_operator_states_quad = make_operator_fluid_states(
        dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=fluid_dd, comm_tag=_FluidOpStatesTag,
        limiter_func=limiter_func)

    # Compute the temperature gradient for both subdomains
    fluid_grad_temperature, wall_grad_temperature = coupled_grad_t_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        time=time,
        interface_noslip=interface_noslip,
        quadrature_tag=quadrature_tag,
        fluid_numerical_flux_func=fluid_gradient_numerical_flux_func,
        _fluid_operator_states_quad=fluid_operator_states_quad,
        _fluid_all_boundaries_no_grad=fluid_all_boundaries_no_grad,
        _wall_all_boundaries_no_grad=wall_all_boundaries_no_grad)

    # Include boundaries for the fluid-wall interface, now with the temperature
    # gradient
    fluid_all_boundaries, wall_all_boundaries = \
        add_interface_boundaries(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_kappa, wall_temperature,
            fluid_grad_temperature, wall_grad_temperature,
            fluid_boundaries, wall_boundaries,
            interface_noslip=interface_noslip,
            wall_penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag)

    # Compute the subdomain NS/diffusion operators using the augmented boundaries
    ns_result = ns_operator(
        dcoll, gas_model, fluid_state, fluid_all_boundaries,
        time=time, quadrature_tag=quadrature_tag, dd=fluid_dd,
        return_gradients=return_gradients,
        operator_states_quad=fluid_operator_states_quad,
        grad_t=fluid_grad_temperature, comm_tag=_FluidOperatorTag)

    diffusion_result = diffusion_operator(
        dcoll, wall_kappa, wall_all_boundaries, wall_temperature,
        penalty_amount=wall_penalty_amount, quadrature_tag=quadrature_tag,
        return_grad_u=return_gradients, dd=wall_dd, grad_u=wall_grad_temperature,
        comm_tag=_WallOperatorTag)

    if return_gradients:
        fluid_rhs, fluid_grad_cv, fluid_grad_temperature = ns_result
        wall_rhs, wall_grad_temperature = diffusion_result
        return (
            fluid_rhs, wall_rhs, fluid_grad_cv, fluid_grad_temperature,
            wall_grad_temperature)
    else:
        return ns_result, diffusion_result


def basic_coupled_ns_heat_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        *,
        time=0.,
        interface_noslip=True,
        interface_radiation=False,
        wall_emissivity=None,
        sigma=None,
        ambient_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        limiter_func=None,
        return_gradients=False,
        use_esdg=False,
        inviscid_terms_on=True):
    r"""
    Simple implementation of a thermally-coupled fluid/wall operator.

    Computes the RHS for a two-volume domain coupled by temperature and heat flux,
    by augmenting *fluid_boundaries* and *wall_boundaries* with the boundaries for
    the fluid-wall interface and calling the respective NS/diffusion operators.

    Parameters
    ----------

    dcoll: class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    fluid_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the fluid volume.

    wall_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the wall volume.

    fluid_boundaries:

        Dictionary of boundary objects for the fluid subdomain, one for each
        :class:`~grudge.dof_desc.BoundaryDomainTag` that represents a domain
        boundary.

    wall_boundaries:

        Dictionary of boundary objects for the wall subdomain, one for each
        :class:`~grudge.dof_desc.BoundaryDomainTag` that represents a domain
        boundary.

    fluid_state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state and dependent
        quantities for the fluid volume.

    wall_kappa: float or :class:`meshmode.dof_array.DOFArray`

        Thermal conductivity for the wall volume.

    wall_temperature: :class:`meshmode.dof_array.DOFArray`

        Temperature for the wall volume.

    time:

        Time

    interface_noslip: bool

        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    interface_radiation: bool

        If `True`, interface includes a radiation sink term in the heat flux. See
        :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall.InterfaceWallRadiationBoundary`
        for details. Additional arguments *wall_emissivity*, *sigma*, and
        *ambient_temperature* are required if enabled.

    wall_emissivity: float or :class:`meshmode.dof_array.DOFArray`

        Emissivity of the wall material.

    sigma: float

        Stefan-Boltzmann constant.

    ambient_temperature: :class:`meshmode.dof_array.DOFArray`

        Ambient temperature of the environment.

    wall_penalty_amount: float

        Coefficient $c$ for the interior penalty on the heat flux. See
        :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall.InterfaceFluidBoundary`
        for details. Not used if *interface_radiation* is `True`.

    quadrature_tag:

        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    limiter_func:

        Callable function to be passed to
        :func:`~mirgecom.gas_model.make_operator_fluid_states`
        that filters or limits the produced fluid states.  This is used to keep
        species mass fractions in physical and realizable states, for example.

    use_esdg: bool

        If `True`, use the entropy-stable version of the Navier-Stokes operator.

    Returns
    -------

        The tuple `(fluid_rhs, wall_rhs)`.
    """
    if wall_penalty_amount is None:
        # FIXME: After verifying the form of the penalty term, figure out what value
        # makes sense to use as a default here
        wall_penalty_amount = 0.05

    if interface_radiation:
        if (
                wall_emissivity is None
                or sigma is None
                or ambient_temperature is None):
            raise TypeError(
                "Arguments 'wall_emissivity', 'sigma' and 'ambient_temperature'"
                "are required if using surface radiation.")

    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    # Include boundaries for the fluid-wall interface; no temperature gradient
    # yet because we need to compute it
    fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad = \
        add_interface_boundaries_no_grad(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_kappa, wall_temperature,
            fluid_boundaries, wall_boundaries,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            quadrature_tag=quadrature_tag)

    # Get the operator fluid states
    fluid_operator_states_quad = make_operator_fluid_states(
        dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=fluid_dd, comm_tag=_FluidOpStatesTag,
        limiter_func=limiter_func)

    # Compute the temperature gradient for both subdomains
    fluid_grad_temperature = fluid_grad_t_operator(
        dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
        time=time, quadrature_tag=quadrature_tag,
        dd=fluid_dd, operator_states_quad=fluid_operator_states_quad,
        comm_tag=_FluidGradTag)
    wall_grad_temperature = wall_grad_t_operator(
        dcoll, wall_kappa, wall_all_boundaries_no_grad, wall_temperature,
        quadrature_tag=quadrature_tag, dd=wall_dd, comm_tag=_WallGradTag)

    # Include boundaries for the fluid-wall interface, now with the temperature
    # gradient
    fluid_all_boundaries, wall_all_boundaries = \
        add_interface_boundaries(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_kappa, wall_temperature,
            fluid_grad_temperature, wall_grad_temperature,
            fluid_boundaries, wall_boundaries,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            wall_emissivity=wall_emissivity,
            sigma=sigma,
            ambient_temperature=ambient_temperature,
            wall_penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag)

    # Compute the subdomain NS/diffusion operators using the augmented boundaries

    my_ns_operator = partial(ns_operator,
        viscous_numerical_flux_func=viscous_facial_flux_harmonic,
        use_esdg=use_esdg, inviscid_terms_on=inviscid_terms_on)
    ns_result = my_ns_operator(
        dcoll, gas_model, fluid_state, fluid_all_boundaries,
        time=time, quadrature_tag=quadrature_tag, dd=fluid_dd,
        return_gradients=return_gradients,
        operator_states_quad=fluid_operator_states_quad,
        grad_t=fluid_grad_temperature, comm_tag=_FluidOperatorTag)

    wall_rhs = diffusion_operator(
        dcoll, wall_kappa, wall_all_boundaries, wall_temperature,
        penalty_amount=wall_penalty_amount, quadrature_tag=quadrature_tag,
        dd=wall_dd, grad_u=wall_grad_temperature, comm_tag=_WallOperatorTag)

    if return_gradients:
        # Ignore fluid_grad_temperature result here because we already have it
        fluid_rhs, fluid_grad_cv, _fluid_grad_temperature = ns_result
        return (
            fluid_rhs, wall_rhs, fluid_grad_cv, fluid_grad_temperature,
            wall_grad_temperature)
    else:
        fluid_rhs = ns_result
        return fluid_rhs, wall_rhs
