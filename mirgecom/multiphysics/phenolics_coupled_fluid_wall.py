r""":mod:`mirgecom.multiphysics.phenolics_coupled_fluid_wall` for thermally-coupled
fluid and composite-wall (simplified model).

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
(:mod:`mirgecom.navierstokes`) with a wall subdomain governed by the heat
equation (:mod:`mirgecom.diffusion`) through temperature and heat flux.
The coupling enforces continuity of temperature and heat flux.

Boundary Setup Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: add_interface_boundaries_no_grad
.. autofunction:: add_interface_boundaries

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: InterfaceFluidNoslipBoundary
.. autoclass:: InterfaceWallRadiationBoundary
"""

__copyright__ = """
Copyright (C) 2024 University of Illinois Board of Trustees
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
from grudge.trace_pair import TracePair, inter_volume_trace_pairs
from grudge.dof_desc import as_dofdesc
from grudge import op
from grudge.dt_utils import characteristic_lengthscales

from mirgecom.inviscid import inviscid_facial_flux_rusanov
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
from mirgecom.multiphysics import make_interface_boundaries
from mirgecom.utils import project_from_base
from mirgecom.fluid import species_mass_fraction_gradient, make_conserved


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


class _SampleMaskTag:
    pass


def get_porous_domain_interface(actx, dcoll, fluid_dd, wall_dd, wall_sample_mask):
    """Compute the porous region of the fluid-wall interface."""

    fluid_zeros = dcoll.zeros(actx, dd=fluid_dd)

    pairwise_mask = {(fluid_dd, wall_dd): (fluid_zeros, wall_sample_mask)}
    mask_pairs = inter_volume_trace_pairs(dcoll, pairwise_mask,
                                          comm_tag=_SampleMaskTag)

    fluid_dd_list = [tpair.dd for tpair in mask_pairs[wall_dd, fluid_dd]]
    solid_dd_list = [tpair.dd for tpair in mask_pairs[fluid_dd, wall_dd]]

    porous_interfaces_fluid = [
        tpair.ext + 0*tpair.int  # Need to include int value to avoid dropping sends
        for tpair in mask_pairs[wall_dd, fluid_dd]]

    porous_interfaces_solid = [
        tpair.int + 0*tpair.ext  # Need to include int value to avoid dropping sends
        for tpair in mask_pairs[fluid_dd, wall_dd]]

    # dummy
    dummy_fluid = [
        0*tpair.int + 0*tpair.ext
        for tpair in mask_pairs[wall_dd, fluid_dd]]

    # dummy
    dummy_solid = [
        0*tpair.ext + 0*tpair.int
        for tpair in mask_pairs[fluid_dd, wall_dd]]

    return (
        porous_interfaces_fluid + dummy_fluid,
        porous_interfaces_solid + dummy_solid,
        fluid_dd_list, solid_dd_list)


def _replace_kappa(state, kappa):
    """Replace the thermal conductivity in fluid state *state* with *kappa*."""
    new_tv = replace(state.tv, thermal_conductivity=kappa)
    return replace(state, tv=new_tv)


class InterfaceFluidNoslipBoundary(MengaldoBoundaryCondition):
    """No-slip boundary for the fluid side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: temperature_bc
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(
            self, kappa_plus, t_plus, boundary_momentum, porous_wall,
            grad_t_plus=None, heat_flux_penalty_amount=None):
        r"""Initialize InterfaceFluidNoslipBoundary.

        Arguments *grad_t_plus* and *heat_flux_penalty_amount* are only
        required if the boundary will be used to compute the viscous flux.

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

        porous_wall: float or :class:`meshmode.dof_array.DOFArray`

            Flag indicating where the fluid-wall interface is porous or not.
            Evaluated using :func:~`get_porous_domain_interface`
        """
        self._penalty_amount = heat_flux_penalty_amount

        self._t_plus = t_plus
        self._boundary_momentum = boundary_momentum
        self._porous_wall = porous_wall[0]

    def _momentum_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Enforce the velocity due to outgasing.

        A non-zero normal velocity is enforced only in the porous region,
        while it is zero in the impermeable region due to the no-slip condition.
        """
        actx = state_minus.cv.mass.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return self._boundary_momentum * normal * self._porous_wall

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state to enforce inviscid fluxes."""
        dd_bdry = as_dofdesc(dd_bdry)

        mom_bc = self._momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)
        momentum_plus = 2.0*mom_bc - state_minus.cv.momentum

        mass_plus = state_minus.cv.mass
        species_plus = state_minus.cv.species_mass

        int_energy_plus = mass_plus*gas_model.eos.get_internal_energy(
            temperature=state_minus.temperature,
            species_mass_fractions=state_minus.species_mass_fractions)
        kin_energy_plus = 0.5*np.dot(momentum_plus, momentum_plus)/mass_plus
        energy_plus = int_energy_plus + kin_energy_plus

        cv_plus = make_conserved(dim=state_minus.dim,
                                 mass=mass_plus,
                                 energy=energy_plus,
                                 momentum=momentum_plus,
                                 species_mass=species_plus)

        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        actx = state_minus.array_context
        wall_temp = project_from_base(dcoll, dd_bdry, self._t_plus)
        return actx.np.zeros_like(state_minus.temperature) + wall_temp

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state to enforce viscous fluxes."""
        dd_bdry = as_dofdesc(dd_bdry)

        mom_bc = self._momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)
        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus, **kwargs)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc,
            species_mass_fractions=state_minus.species_mass_fractions)

        total_energy_bc = state_minus.mass_density*internal_energy_bc + \
            0.5*np.dot(mom_bc, mom_bc)/state_minus.mass_density

        return replace_fluid_state(state_minus, gas_model, momentum=mom_bc,
                                   energy=total_energy_bc)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return BC on grad(CV).

        Zero gradient on the solid wall region;
        Robin BC for species diffusion
        """
        actx = state_minus.array_context

        pyro_mech = gas_model.eos._pyrometheus_mech
        idx_X2 = pyro_mech.species_indices["X2"]  # noqa

        y_reference = state_minus.cv.species_mass_fractions*0.0
        y_reference[idx_X2] = actx.np.ones_like(state_minus.cv.mass)

        normal_velocity = 1.0/state_minus.cv.mass*(
            self._momentum_bc(dcoll, dd_bdry, state_minus, **kwargs))

        grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                      grad_cv_minus)
        grad_y_tangential = grad_y_minus - np.outer(grad_y_minus@normal, normal)

        # prescribe directly the boundary gradient since no numerical flux is used
        diff = state_minus.tv.species_diffusivity
        delta_y = state_minus.species_mass_fractions - y_reference
        grad_y_bc = np.outer(delta_y/diff, normal_velocity) + grad_y_tangential

        grad_species_mass_bc = (
            state_minus.mass_density*grad_y_bc
            + np.outer(state_minus.species_mass_fractions, grad_cv_minus.mass)
        )

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

        base_flux = numerical_flux_func(boundary_state_pair, gas_model, normal)

        momentum_boundary = self._momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)
        presc_mass_flux = momentum_boundary@normal

        # apply Robin BC for species diffusion, so extrapolate Y from the
        # fluid and only prescribe the total mass flux from the wall
        return base_flux.replace(mass=presc_mass_flux)


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
        # TODO add species diffusion energy flux
        return (
            np.dot(diffusion_flux(kappa_plus, grad_u_plus), normal)
            + emissivity * self.sigma * (u_minus**4 - u_ambient**4))

#        from arraycontext import outer

#        nspecies = len(state_minus.species_mass_density)

#        # ~~~
#        grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
#                                                      grad_cv_minus)
#        grad_y_solid = grad_y_minus - np.outer(grad_y_minus@normal, normal)
#        zero_grad_species_mass_bc = 0.*grad_y_solid
#        for i in range(nspecies):
#            zero_grad_species_mass_bc[i] = \
#                (state_minus.mass_density*grad_y_solid[i]
#                 + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

#        # ~~~
#        pyro_mech = gas_model.eos._pyrometheus_mech
#        idx_X2 = pyro_mech.species_indices["X2"]  # noqa

#        normal_velocity = self._boundary_momentum/state_minus.cv.mass

#        y_reference = state_minus.cv.species_mass_fractions*0.0
#        y_reference[idx_X2] = 1.0

#        presc_grad_species_mass_bc = 0.*grad_cv_minus.species_mass
#        for i in range(nspecies):
#            delta_y = state_minus.cv.species_mass_fractions[i] - y_reference[i]
#            dij = state_minus.tv.species_diffusivity[i]
#            grad_y_porous = + (normal_velocity*delta_y/dij) * normal
#            presc_grad_species_mass_bc[i] = (
#                state_minus.mass_density*grad_y_porous
#                + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

#        grad_y_bc = (
#            presc_grad_species_mass_bc*(self._porous_wall)
#            + zero_grad_species_mass_bc*(1.0 - self._porous_wall))

#        rho_bc = state_minus.cv.mass
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


##class InterfaceFluidNoslipBoundary(MengaldoBoundaryCondition):
##    """
##    Boundary for the fluid side of the fluid-wall interface, without slip.

##    .. automethod:: __init__
##    .. automethod:: state_plus
##    .. automethod:: state_bc
##    .. automethod:: grad_cv_bc
##    .. automethod:: temperature_plus
##    .. automethod:: temperature_bc
##    .. automethod:: grad_temperature_bc
##    """

##    def __init__(
##            self, kappa_plus, t_plus, grad_t_plus=None,
##            heat_flux_penalty_amount=None, lengthscales_minus=None):
##        r"""
##        Initialize InterfaceFluidNoslipBoundary.

##        Arguments *grad_t_plus*, *heat_flux_penalty_amount*, and
##        *lengthscales_minus* are only required if the boundary will be used to
##        compute the viscous flux.

##        Parameters
##        ----------
##        kappa_plus: float or :class:`meshmode.dof_array.DOFArray`

##            Thermal conductivity from the wall side.

##        t_plus: :class:`meshmode.dof_array.DOFArray`

##            Temperature from the wall side.

##        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None

##            Temperature gradient from the wall side.

##        heat_flux_penalty_amount: float or None

##            Coefficient $c$ for the interior penalty on the heat flux.

##        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None

##            Characteristic mesh spacing $h^-$.
##        """
##        self._penalty_amount = heat_flux_penalty_amount
##        self._lengthscales_minus = lengthscales_minus

##        self._t_plus = t_plus
##        self._boundary_momentum = boundary_momentum
##        self._porous_wall = porous_wall[0]

##    def _momentum_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
##        """Enforce the velocity due to outgasing.

##        A non-zero normal velocity is enforced only in the porous region,
##        while it is zero in the impermeable region due to the no-slip condition.
##        """
##        actx = state_minus.cv.mass.array_context
##        normal = actx.thaw(dcoll.normal(dd_bdry))
##        return self._boundary_momentum * normal * self._porous_wall

##    def state_plus(
##            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
##        dd_bdry = as_dofdesc(dd_bdry)
##        mom_plus = self._no_slip.momentum_plus(state_minus.momentum_density)

##        # Don't bother replacing kappa since this is just for inviscid
##        return replace_fluid_state(state_minus, gas_model, momentum=mom_plus)

##    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
##        """Return state to enforce viscous fluxes."""
##        dd_bdry = as_dofdesc(dd_bdry)

##        mom_bc = self._momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)
##        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus, **kwargs)

##        internal_energy_bc = gas_model.eos.get_internal_energy(
##            temperature=t_bc,
##            species_mass_fractions=state_minus.species_mass_fractions)

##        total_energy_bc = state_minus.mass_density*internal_energy_bc + \
##            0.5*np.dot(mom_bc, mom_bc)/state_minus.mass_density

##        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

##        return _replace_kappa(
##            replace_fluid_state(
##                state_minus, gas_model,
##                energy=total_energy_bc,
##                momentum=mom_bc),
##            kappa_bc)

##    def _normal_kappa_plus(self, dcoll, dd_bdry):
##        # project kappa plus in case of overintegration
##        if isinstance(self._kappa_plus, np.ndarray):
##            # orthotropic materials
##            actx = self._t_plus.array_context
##            normal = actx.thaw(dcoll.normal(dd_bdry))
##            kappa_plus = project_from_base(dcoll, dd_bdry, self._kappa_plus)
##            return np.dot(normal, kappa_plus*normal)
##        return project_from_base(dcoll, dd_bdry, self._kappa_plus)

##    def _normal_kappa_minus(self, dcoll, dd_bdry, kappa):
##        # state minus is already in the quadrature domain
##        if isinstance(kappa, np.ndarray):
##            # orthotropic materials
##            actx = self._t_plus.array_context
##            normal = actx.thaw(dcoll.normal(dd_bdry))
##            return np.dot(normal, kappa*normal)
##        return kappa

##    def kappa_bc(self, dcoll, dd_bdry, kappa_minus):
##        return harmonic_mean(kappa_minus,
##                             project_from_base(dcoll, dd_bdry, self._kappa_plus))

##    def temperature_plus(
##            self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
##        return project_from_base(dcoll, dd_bdry, self._t_plus)

##    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
##        t_plus = project_from_base(dcoll, dd_bdry, self._t_plus)
##        actx = t_minus.array_context
##        kappa_plus = self._normal_kappa_plus(dcoll, dd_bdry)
##        kappa_minus = self._normal_kappa_minus(dcoll, dd_bdry,
##                                               kappa_minus + t_minus*0.0)
##        kappa_sum = actx.np.where(
##            actx.np.greater(kappa_minus + kappa_plus, 0*kappa_minus),
##            kappa_minus + kappa_plus,
##            0*kappa_minus + 1)
##        return (t_minus * kappa_minus + t_plus * kappa_plus)/kappa_sum

##    def grad_cv_bc(
##            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus, normal,
##            **kwargs):
##        """Return BC on grad(CV).

##        Zero gradient on the solid wall region;
##        Robin BC for species diffusion
##        """
##        actx = state_minus.array_context

##        pyro_mech = gas_model.eos._pyrometheus_mech
##        idx_X2 = pyro_mech.species_indices["X2"]  # noqa

##        y_reference = state_minus.cv.species_mass_fractions*0.0
##        y_reference[idx_X2] = actx.np.ones_like(state_minus.cv.mass)

##        normal_velocity = 1.0/state_minus.cv.mass*(
##            self._momentum_bc(dcoll, dd_bdry, state_minus, **kwargs))

##        grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
##                                                      grad_cv_minus)
##        grad_y_tangential = grad_y_minus - np.outer(grad_y_minus@normal, normal)

##        # prescribe directly the boundary gradient since no numerical flux is used
##        diff = state_minus.tv.species_diffusivity
##        delta_y = state_minus.species_mass_fractions - y_reference
##        grad_y_bc = np.outer(delta_y/diff, normal_velocity) + grad_y_tangential

##        grad_species_mass_bc = (
##            state_minus.mass_density*grad_y_bc
##            + np.outer(state_minus.species_mass_fractions, grad_cv_minus.mass)
##        )

##        return grad_cv_minus.replace(species_mass=grad_species_mass_bc)

##    def grad_temperature_bc(
##            self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
##        if self._grad_t_plus is None:
##            raise TypeError(
##                "Boundary does not have external temperature gradient data.")
##        grad_t_plus = project_from_base(dcoll, dd_bdry, self._grad_t_plus)
##        return (grad_t_plus + grad_t_minus)/2

##    def inviscid_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
##                                 numerical_flux_func=inviscid_facial_flux_rusanov,
##                                 **kwargs):
##        """Get the inviscid boundary flux for the divergence operator."""
##        dd_bdry = as_dofdesc(dd_bdry)
##        actx = state_minus.array_context
##        state_plus = self.state_plus(
##            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
##            state_minus=state_minus, **kwargs)
##        boundary_state_pair = TracePair(dd=dd_bdry,
##                                        interior=state_minus,
##                                        exterior=state_plus)
##        normal = actx.thaw(dcoll.normal(dd_bdry))

##        base_flux = numerical_flux_func(boundary_state_pair, gas_model, normal)

##        momentum_boundary = self._momentum_bc(dcoll, dd_bdry, state_minus, **kwargs)
##        presc_mass_flux = momentum_boundary@normal

##        # apply Robin BC for species diffusion, so extrapolate Y from the
##        # fluid and only prescribe the total mass flux from the wall
##        return base_flux.replace(mass=presc_mass_flux)


class InterfaceWallBoundary(DiffusionBoundary):
    """
    Boundary for the wall side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(self, kappa_plus, grad_u_plus=None):
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

        # Note: numerical_flux_func is ignored
        # TODO add species diffusion energy flux
        return np.dot(diffusion_flux(kappa_plus, grad_u_plus), normal)


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
        boundary_momentum, porous_wall,
        *,
        interface_radiation=None,
        comm_tag=None):
    interface_tpairs = _get_interface_trace_pairs_no_grad(
        dcoll,
        fluid_dd, wall_dd,
        fluid_kappa, wall_kappa,
        fluid_temperature, wall_temperature,
        comm_tag=comm_tag)

    def make_fluid_boundary(interface_tpair):
        return InterfaceFluidNoslipBoundary(
            kappa_plus=interface_tpair.ext.kappa,
            t_plus=interface_tpair.ext.temperature,
            boundary_momentum=boundary_momentum,
            porous_wall=porous_wall)

    # TODO combine the two functions in a single one
    if interface_radiation:
        def make_wall_boundary(interface_tpair):
            return InterfaceWallRadiationBoundary(
                interface_tpair.ext.kappa)
    else:
        def make_wall_boundary(interface_tpair):
            return InterfaceWallBoundary(
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
        boundary_momentum, porous_wall,
        fluid_grad_temperature, wall_grad_temperature,
        *,
        interface_radiation=None,
        wall_emissivity=None,
        sigma=None,
        ambient_temperature=None,
        wall_penalty_amount=None,
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
        if (wall_emissivity is None or sigma is None or ambient_temperature is None):
            raise TypeError(
                "Arguments 'wall_emissivity', 'sigma' and 'ambient_temperature'"
                "are required if using surface radiation.")

    fluid_lengthscales = (
        characteristic_lengthscales(
            fluid_temperature.array_context, dcoll, fluid_dd)
        * (0*fluid_temperature+1))

    def make_fluid_boundary(interface_tpair):
        return InterfaceFluidNoslipBoundary(
            kappa_plus=interface_tpair.ext.kappa,
            t_plus=interface_tpair.ext.temperature,
            boundary_momentum=boundary_momentum,
            porous_wall=porous_wall,
            grad_t_plus=interface_tpair.ext.grad_temperature,
            heat_flux_penalty_amount=wall_penalty_amount,
            )

    # TODO combine the two functions in a single one
    if interface_radiation:
        def make_wall_boundary(interface_tpair):
            emissivity_minus = op.project(dcoll, wall_dd, interface_tpair.dd,
                                          wall_emissivity)
            return InterfaceWallRadiationBoundary(
                interface_tpair.ext.kappa,
                interface_tpair.ext.grad_temperature,
                emissivity_minus, sigma,
                ambient_temperature)
    else:
        def make_wall_boundary(interface_tpair):
            return InterfaceWallBoundary(
                interface_tpair.ext.kappa,
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
        boundary_momentum, porous_wall,
        fluid_boundaries, wall_boundaries,
        *,
        interface_radiation=True,
        quadrature_tag=None,
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

    boundary_momentum: float or :class:`meshmode.dof_array.DOFArray`

        Normal component of the blowing velocity along the porous wall.

    porous_wall: float or :class:`meshmode.dof_array.DOFArray`

        Flag indicating where the fluid-wall interface is porous or not.
        Evaluated using :func:~`get_porous_domain_interface`

    fluid_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the fluid subdomain.

    wall_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the wall subdomain.

    quadrature_tag

        Deprecated

    comm_tag: Hashable
        Tag for distributed communication
    """
    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        _get_interface_boundaries_no_grad(
            dcoll,
            fluid_dd, wall_dd,
            fluid_state.tv.thermal_conductivity, wall_kappa,
            fluid_state.temperature, wall_temperature,
            boundary_momentum, porous_wall,
            interface_radiation=interface_radiation,
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
        boundary_momentum, porous_wall,
        fluid_grad_temperature, wall_grad_temperature,
        fluid_boundaries, wall_boundaries,
        *,
        interface_radiation=True,
        wall_emissivity=None,
        sigma=None,
        ambient_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=None,
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

    boundary_momentum: float or :class:`meshmode.dof_array.DOFArray`

        Normal component of the blowing velocity along the porous wall.

    porous_wall: float or :class:`meshmode.dof_array.DOFArray`

        Flag indicating where the fluid-wall interface is porous or not.
        Evaluated using :func:~`get_porous_domain_interface`

    fluid_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the fluid subdomain.

    wall_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the wall subdomain.

    wall_emissivity: float or :class:`meshmode.dof_array.DOFArray`

        Emissivity of the wall material.

    sigma: float

        Stefan-Boltzmann constant.

    ambient_temperature: :class:`meshmode.dof_array.DOFArray`

        Ambient temperature of the environment.

    wall_penalty_amount: float

        Coefficient $c$ for the interior penalty on the heat flux.

    quadrature_tag

        Deprecated

    comm_tag: Hashable
        Tag for distributed communication
    """
    fluid_interface_boundaries, wall_interface_boundaries = \
        _get_interface_boundaries(
            dcoll,
            fluid_dd, wall_dd,
            fluid_state.tv.thermal_conductivity, wall_kappa,
            fluid_state.temperature, wall_temperature,
            boundary_momentum, porous_wall,
            fluid_grad_temperature, wall_grad_temperature,
            interface_radiation=interface_radiation,
            wall_emissivity=wall_emissivity,
            sigma=sigma,
            ambient_temperature=ambient_temperature,
            wall_penalty_amount=wall_penalty_amount,
            comm_tag=comm_tag)

    fluid_all_boundaries = {}
    fluid_all_boundaries.update(fluid_boundaries)
    fluid_all_boundaries.update(fluid_interface_boundaries)

    wall_all_boundaries = {}
    wall_all_boundaries.update(wall_boundaries)
    wall_all_boundaries.update(wall_interface_boundaries)

    return fluid_all_boundaries, wall_all_boundaries
