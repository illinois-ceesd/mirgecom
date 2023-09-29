r"""TODO."""

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

from arraycontext import dataclass_array_container
from meshmode.dof_array import DOFArray
from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    as_dofdesc,
)
import grudge.op as op

from mirgecom.math import harmonic_mean, weighted_arithmetic_mean
from mirgecom.boundary import MengaldoBoundaryCondition
from mirgecom.gas_model import (
    make_fluid_state,
    replace_fluid_state,
)
from mirgecom.diffusion import (
    grad_facial_flux_weighted,
    diffusion_flux,
    diffusion_facial_flux_harmonic,
    DiffusionBoundary
)
from mirgecom.inviscid import inviscid_facial_flux_rusanov
from mirgecom.multiphysics import make_interface_boundaries
from mirgecom.utils import project_from_base
from mirgecom.fluid import (
    species_mass_fraction_gradient,
    make_conserved
)


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


class _ThermallyCoupledHarmonicMeanBoundaryComponent:
    def __init__(
            self, kappa_plus, t_plus, grad_t_plus=None):
        self._kappa_plus = kappa_plus
        self._t_plus = t_plus
        self._grad_t_plus = grad_t_plus

    def kappa_plus(self, dcoll, dd_bdry):
        """."""
        return project_from_base(dcoll, dd_bdry, self._kappa_plus)

    def kappa_bc(self, dcoll, dd_bdry, kappa_minus):
        """."""
        kappa_plus = project_from_base(dcoll, dd_bdry, self._kappa_plus)
        return harmonic_mean(kappa_minus, kappa_plus)

    def temperature_plus(self, dcoll, dd_bdry):
        """."""
        return project_from_base(dcoll, dd_bdry, self._t_plus)

    def temperature_bc(self, dcoll, dd_bdry, kappa_minus, t_minus):
        """."""
        t_plus = project_from_base(dcoll, dd_bdry, self._t_plus)
        kappa_plus = project_from_base(dcoll, dd_bdry, self._kappa_plus)
        return weighted_arithmetic_mean(t_plus, t_minus, kappa_plus, kappa_minus)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus):
        """."""
        if self._grad_t_plus is None:
            raise TypeError(
                "Boundary does not have external temperature gradient data.")
        grad_t_plus = project_from_base(dcoll, dd_bdry, self._grad_t_plus)
        return (grad_t_plus + grad_t_minus)/2


def _replace_kappa(state, kappa):
    """Replace the thermal conductivity in fluid state *state* with *kappa*."""
    new_tv = replace(state.tv, thermal_conductivity=kappa)
    return replace(state, tv=new_tv)


# class InterfaceFluidNoslipBoundary(InterfaceFluidBoundary):
#    """No-slip boundary for the fluid side of the fluid-wall interface.

#    .. automethod:: __init__
#    .. automethod:: state_plus
#    .. automethod:: state_bc
#    .. automethod:: grad_cv_bc
#    .. automethod:: temperature_plus
#    .. automethod:: temperature_bc
#    .. automethod:: grad_temperature_bc
#    """

#    def __init__(
#            self, kappa_plus, t_plus, boundary_momentum, grad_t_plus=None,
#            heat_flux_penalty_amount=None, lengthscales_minus=None,
#            porous_wall=None):
#        r"""
#        Initialize InterfaceFluidNoslipBoundary.

#        Arguments *grad_t_plus*, *heat_flux_penalty_amount*, and
#        *lengthscales_minus* are only required if the boundary will be used to
#        compute the viscous flux.

#        Parameters
#        ----------
#        kappa_plus: float or :class:`meshmode.dof_array.DOFArray`

#            Thermal conductivity from the wall side.

#        t_plus: :class:`meshmode.dof_array.DOFArray`

#            Temperature from the wall side.

#        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None

#            Temperature gradient from the wall side.

#        heat_flux_penalty_amount: float or None

#            Coefficient $c$ for the interior penalty on the heat flux.

#        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None

#            Characteristic mesh spacing $h^-$.

#        porous_wall: :class:`meshmode.dof_array.DOFArray` or None

#            XXX
#        """
#        InterfaceFluidBoundary.__init__(
#            self,
#            heat_flux_penalty_amount=heat_flux_penalty_amount,
#            lengthscales_minus=lengthscales_minus)

#        self._thermally_coupled = _ThermallyCoupledHarmonicMeanBoundaryComponent(
#            kappa_plus=kappa_plus,
#            t_plus=t_plus,
#            grad_t_plus=grad_t_plus)

#        self._t_plus = t_plus
#        self._boundary_momentum = boundary_momentum
#        self._porous_wall = porous_wall

#    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
#        """Return state to enforce inviscid fluxes."""
#        dd_bdry = as_dofdesc(dd_bdry)

#        mom_bc = self.momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)
#        momentum_plus = 2.0*mom_bc - state_minus.cv.momentum

#        mass_plus = state_minus.cv.mass

#        int_energy_plus = mass_plus*gas_model.eos.get_internal_energy(
#            temperature=self._t_plus,
#            species_mass_fractions=state_minus.species_mass_fractions)
#        kin_energy_plus = 0.5*np.dot(momentum_plus, momentum_plus)/mass_plus
#        energy_plus = int_energy_plus + kin_energy_plus

#        species_plus = state_minus.cv.species_mass*1.0

#        from mirgecom.fluid import make_conserved
#        cv_plus = make_conserved(dim=state_minus.dim,
#                                 mass=mass_plus,
#                                 energy=energy_plus,
#                                 momentum=momentum_plus,
#                                 species_mass=species_plus)

#        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
#                                temperature_seed=self._t_plus,
#                                smoothness_mu=state_minus.smoothness_mu,
#                                smoothness_kappa=state_minus.smoothness_kappa,
#                                smoothness_beta=state_minus.smoothness_beta)

#    def momentum_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
#        """Enforce the velocity due to outgasing."""
#        actx = state_minus.cv.mass.array_context
#        normal = actx.thaw(dcoll.normal(dd_bdry))
#        return self._boundary_momentum*normal

#    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
#        """Get temperature value used in grad(T)."""
#        actx = state_minus.array_context
#        wall_temp = project_from_base(dcoll, dd_bdry, self._t_plus)
#        return actx.np.zeros_like(state_minus.temperature) + wall_temp

#    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
#        """Return state to enforce viscous fluxes."""
#        dd_bdry = as_dofdesc(dd_bdry)

#        kappa_minus = (
#            # Make sure it has an array context
#            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)

#        momentum_bc = self.momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)

#        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus, **kwargs)

#        internal_energy_bc = gas_model.eos.get_internal_energy(
#            temperature=t_bc,
#            species_mass_fractions=state_minus.species_mass_fractions)

#        total_energy_bc = state_minus.mass_density*internal_energy_bc + \
#            0.5*np.dot(momentum_bc, momentum_bc)/state_minus.mass_density

#        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

#        return _replace_kappa(
#            replace_fluid_state(
#                state_minus, gas_model,
#                energy=total_energy_bc,
#                momentum=momentum_bc),
#            kappa_bc)

#    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
#                   normal, **kwargs):
#        """Return BC on grad(CV).

#        Zero gradient on the solid wall region; do nothing in the
#        porous wall interface."""
#        actx = state_minus.array_context
#        nspecies = len(state_minus.species_mass_density)

#        grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
#                                                      grad_cv_minus)
#        grad_y_bc = grad_y_minus - np.outer(grad_y_minus@normal, normal)
#        zero_grad_species_mass = 0.*grad_y_bc
#        for i in range(nspecies):
#            zero_grad_species_mass[i] = \
#                (state_minus.mass_density*grad_y_bc[i]
#                 + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

#        grad_species_mass_bc = (
#            grad_cv_minus.species_mass*(self._porous_wall)
#            + zero_grad_species_mass*(1.0 - self._porous_wall))

#        return grad_cv_minus.replace(species_mass=grad_species_mass_bc)

#    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, **kwargs):
#        """Return BC on grad(temperature)."""
#        # Mengaldo Eqns (50-51)
#        return grad_t_minus

#    def inviscid_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
#                                 numerical_flux_func=inviscid_facial_flux_rusanov,
#                                 **kwargs):
#        """Get the inviscid boundary flux for the divergence operator."""
#        dd_bdry = as_dofdesc(dd_bdry)
#        actx = state_minus.array_context
#        state_plus = self.state_plus(
#            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
#            state_minus=state_minus, **kwargs)
#        boundary_state_pair = TracePair(dd=dd_bdry,
#                                        interior=state_minus,
#                                        exterior=state_plus)
#        normal = actx.thaw(dcoll.normal(dd_bdry))

#        base_flux = numerical_flux_func(boundary_state_pair, gas_model, normal)

#        presc_flux = self._boundary_momentum*normal@normal
#        mass_flux = 0.5*(presc_flux + base_flux.mass)

#        species_flux = base_flux.species_mass*1.0
#        species_flux[-1] = 0.5*(presc_flux + base_flux.species_mass[-1])

#        return base_flux.replace(mass=mass_flux, species_mass=species_flux)


class InterfaceFluidNoslipBoundary(MengaldoBoundaryCondition):
    """No-slip boundary for the fluid side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_plus
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(
            self, kappa_plus, t_plus, boundary_momentum, grad_t_plus=None,
            heat_flux_penalty_amount=None, lengthscales_minus=None,
            porous_wall=None):
        r"""Initialize InterfaceFluidNoslipBoundary.

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

        porous_wall: :class:`meshmode.dof_array.DOFArray` or None

            XXX
        """
        self._thermally_coupled = _ThermallyCoupledHarmonicMeanBoundaryComponent(
            kappa_plus=kappa_plus,
            t_plus=t_plus,
            grad_t_plus=grad_t_plus)

        self._penalty_amount = heat_flux_penalty_amount
        self._lengthscales_minus = lengthscales_minus

        self._t_plus = t_plus
        self._boundary_momentum = boundary_momentum
        self._porous_wall = porous_wall

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state to enforce inviscid fluxes."""
        dd_bdry = as_dofdesc(dd_bdry)

        mom_bc = self.momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)
        momentum_plus = 2.0*mom_bc - state_minus.cv.momentum

        mass_plus = state_minus.cv.mass

        int_energy_plus = mass_plus*gas_model.eos.get_internal_energy(
            temperature=self._t_plus,
            species_mass_fractions=state_minus.species_mass_fractions)
        kin_energy_plus = 0.5*np.dot(momentum_plus, momentum_plus)/mass_plus
        energy_plus = int_energy_plus + kin_energy_plus

        species_plus = state_minus.cv.species_mass*1.0

        cv_plus = make_conserved(dim=state_minus.dim,
                                 mass=mass_plus,
                                 energy=energy_plus,
                                 momentum=momentum_plus,
                                 species_mass=species_plus)

        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=self._t_plus,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def momentum_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Enforce the velocity due to outgasing."""
        actx = state_minus.cv.mass.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return self._boundary_momentum*normal*state_minus.species_mass_fractions[-1]

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        actx = state_minus.array_context
        wall_temp = project_from_base(dcoll, dd_bdry, self._t_plus)
        return actx.np.zeros_like(state_minus.temperature) + wall_temp

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state to enforce viscous fluxes."""
        dd_bdry = as_dofdesc(dd_bdry)

        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)

        momentum_bc = self.momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)

        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus, **kwargs)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc,
            species_mass_fractions=state_minus.species_mass_fractions)

        total_energy_bc = state_minus.mass_density*internal_energy_bc + \
            0.5*np.dot(momentum_bc, momentum_bc)/state_minus.mass_density

        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

        return _replace_kappa(
            replace_fluid_state(
                state_minus, gas_model,
                energy=total_energy_bc,
                momentum=momentum_bc),
            kappa_bc)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return BC on grad(CV).

        Zero gradient on the solid wall region; do nothing in the
        porous wall interface.
        """
        nspecies = len(state_minus.species_mass_density)

        gradient_at_the_porous_material = grad_cv_minus.species_mass

        grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                      grad_cv_minus)
        grad_y_bc = grad_y_minus - np.outer(grad_y_minus@normal, normal)
        zero_grad_species_mass = 0.*grad_y_bc
        for i in range(nspecies):
            zero_grad_species_mass[i] = \
                (state_minus.mass_density*grad_y_bc[i]
                 + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        grad_species_mass_bc = (
            gradient_at_the_porous_material*(self._porous_wall)
            + zero_grad_species_mass*(1.0 - self._porous_wall))

        return grad_cv_minus.replace(species_mass=grad_species_mass_bc)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return BC on grad(temperature)."""
        # Mengaldo Eqns (50-51)
        return grad_t_minus

    def inviscid_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                 numerical_flux_func=inviscid_facial_flux_rusanov,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        dd_bdry = as_dofdesc(dd_bdry)
        actx = state_minus.array_context
        state_plus = self.state_plus(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)
        boundary_state_pair = TracePair(dd=dd_bdry,
                                        interior=state_minus,
                                        exterior=state_plus)
        normal = actx.thaw(dcoll.normal(dd_bdry))

        inviscid_flux = numerical_flux_func(boundary_state_pair, gas_model, normal)

        presc_flux = self._boundary_momentum*normal@normal
        mass_flux = 0.5*(presc_flux + inviscid_flux.mass)

        species_flux = mass_flux*state_minus.cv.species_mass_fractions

        return inviscid_flux.replace(mass=mass_flux, species_mass=species_flux)


class InterfaceWallRadiationBoundary(DiffusionBoundary):
    r"""Boundary for the wall side of the fluid-wall interface (radiating).

    Enforces the heat flux based on the fluid side plus a radiation sink term:

    .. math::
        -\kappa_\text{wall} \nabla T_\text{wall} \cdot \hat{n} =
            -\sum \rho D_i \nabla Y_i h_i
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
        r"""Initialize InterfaceWallRadiationBoundary.

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
            numerical_flux_func=grad_facial_flux_weighted):
        """Return the numerical fluxes for gradient evaluation."""
        actx = u_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_minus)

        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):
        """Return the prescribed fluxes for diffusion.

        The flux is prescribed based on fluid-only properties with a
        non-conservative sink term to account for radiation.
        """
        if self.grad_u_plus is None:
            raise TypeError("External temperature gradient is not specified.")

        actx = u_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        kappa_plus = project_from_base(dcoll, dd_bdry, self.kappa_plus)
        grad_u_plus = project_from_base(dcoll, dd_bdry, self.grad_u_plus)
        emissivity = project_from_base(dcoll, dd_bdry, self.emissivity)
        u_ambient = project_from_base(dcoll, dd_bdry, self.u_ambient)

        # Note: numerical_flux_func is ignored
        return (
            # FIXME add species diffusion flux
            np.dot(diffusion_flux(kappa_plus, grad_u_plus), normal)
            + emissivity * self.sigma * (u_minus**4 - u_ambient**4))

#        # species flux
#        from arraycontext import outer
#        grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
#                                                      grad_cv_minus)
#        grad_y_plus = species_mass_fraction_gradient(state_plus.cv,
#                                                     self._grad_cv_plus)
#        rho_bc = 0.5*(state_minus.cv.mass + state_plus.cv.mass)
#        grad_y_bc = 1.0/(state_minus.wv.void_fraction + 1.0)*(
#            state_minus.wv.void_fraction*grad_y_minus + grad_y_plus)
#        d_bc = state_bc.species_diffusivity
#        y_bc = state_bc.species_mass_fractions

#        species_flux = -rho_bc*(
#            d_bc.reshape(-1, 1)*grad_y_bc
#            - outer(y_bc, sum(d_bc.reshape(-1, 1)*grad_y_bc)))
#        species_mass_flux = -species_flux@normal

#        # heat flux due to shear, species and thermal conduction
#        # tau = viscous_stress_tensor(state_bc, grad_cv_bc)
#        h_alpha = state_bc.species_enthalpies
#        heat_flux = (
#            # np.dot(tau, state_bc.velocity)
#            - sum(h_alpha.reshape(-1, 1) * species_flux)
#            - diffusion_flux(kappa_plus, grad_t_plus))@normal

#        # radiation sink term
#        wall_emissivity = project_from_base(dcoll, dd_bdry, self._emissivity)
#        radiation = wall_emissivity * self._sigma * (
#            state_minus.temperature**4 - self._u_ambient**4)

#        return heat_flux - radiation


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

    return inter_volume_trace_pairs(
        dcoll, pairwise_thermal_data,
        comm_tag=(_ThermalDataInterVolTag, comm_tag))


def _get_interface_boundaries_no_grad(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        boundary_momentum,
        *,
        quadrature_tag=DISCR_TAG_BASE,  # XXX dummy
        comm_tag=None):
    interface_tpairs = _get_interface_trace_pairs_no_grad(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        comm_tag=comm_tag)

    def make_fluid_boundary(interface_tpair):
        return InterfaceFluidNoslipBoundary(
            interface_tpair.ext.kappa,
            interface_tpair.ext.temperature,
            boundary_momentum)

    def make_wall_boundary(interface_tpair):
        return InterfaceWallRadiationBoundary(
            interface_tpair.ext.kappa)

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
        boundary_momentum,
        fluid_grad_temperature, wall_grad_temperature,
        interface_radiation,  # XXX dummy
        porous_wall,
        *,
        interface_noslip=None,  # XXX dummy
        wall_emissivity=None,
        sigma=None,
        ambient_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,  # XXX dummy
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

    if (wall_emissivity is None or sigma is None or ambient_temperature is None):
        raise TypeError(
            "Arguments 'wall_emissivity', 'sigma' and 'ambient_temperature'"
            "are required if using surface radiation.")

    from grudge.dt_utils import characteristic_lengthscales
    fluid_lengthscales = (
        characteristic_lengthscales(
            fluid_temperature.array_context, dcoll, fluid_dd)
        * (0*fluid_temperature+1))

    def make_fluid_boundary(interface_tpair):
        return InterfaceFluidNoslipBoundary(
            interface_tpair.ext.kappa,
            interface_tpair.ext.temperature,
            boundary_momentum,
            interface_tpair.ext.grad_temperature,
            heat_flux_penalty_amount=wall_penalty_amount,
            lengthscales_minus=op.project(dcoll,
                fluid_dd, interface_tpair.dd, fluid_lengthscales),
            porous_wall=porous_wall)

    def make_wall_boundary(interface_tpair):
        emissivity_minus = op.project(dcoll, wall_dd, interface_tpair.dd,
                                      wall_emissivity)
        return InterfaceWallRadiationBoundary(
            interface_tpair.ext.kappa,
            interface_tpair.ext.grad_temperature,
            emissivity_minus, sigma,
            ambient_temperature)

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
        gas_model,  # XXX dummy
        fluid_dd, wall_dd,
        fluid_state, wall_kappa, wall_temperature,
        boundary_momentum,
        fluid_boundaries, wall_boundaries,
        *,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    """Include the fluid-wall interface boundaries (without temperature gradient).

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
            boundary_momentum,
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
        gas_model,  # XXX dummy
        fluid_dd, wall_dd,
        fluid_state, wall_kappa, wall_temperature,
        boundary_momentum,
        fluid_grad_temperature, wall_grad_temperature,
        fluid_boundaries, wall_boundaries,
        *,
        porous_wall=None,
        interface_noslip=True,
        interface_radiation=False,
        wall_emissivity=None,
        sigma=None,
        ambient_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    """Include the fluid-wall interface boundaries.

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

    porous_wall: float or :class:`meshmode.dof_array.DOFArray`

        XXX

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
            boundary_momentum,
            fluid_grad_temperature, wall_grad_temperature,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            porous_wall=porous_wall,
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
