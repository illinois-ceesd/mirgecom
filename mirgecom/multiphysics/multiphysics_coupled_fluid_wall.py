r""":mod:`mirgecom.multiphysics.multiphysics_coupled_fluid_wall` for
fully-coupled fluid and wall.

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
with a wall subdomain governed by the porous media equation by enforcing
continuity of quantities and their respective fluxes

.. math::
    q_\text{fluid} &= q_\text{wall} \\
    - D_\text{fluid} \nabla q_\text{fluid} \cdot \hat{n} &=
        - D_\text{wall} \nabla q_\text{wall} \cdot \hat{n}.

at the interface.

.. autofunction:: add_interface_boundaries_no_grad
.. autofunction:: add_interface_boundaries
.. autoclass:: InterfaceFluidBoundary
.. autoclass:: InterfaceWallBoundary
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

from grudge.trace_pair import (
    inter_volume_trace_pairs,
    TracePair
)
from grudge.dof_desc import as_dofdesc

from mirgecom.fluid import make_conserved
from mirgecom.fluid import species_mass_fraction_gradient
from mirgecom.math import harmonic_mean
from mirgecom.transport import GasTransportVars
from mirgecom.boundary import MengaldoBoundaryCondition
# from mirgecom.viscous import (
#     viscous_stress_tensor,
# )
from mirgecom.gas_model import (
    make_fluid_state,
    ViscousFluidState
)
from mirgecom.diffusion import diffusion_flux
from mirgecom.utils import project_from_base
from mirgecom.wall_model import PorousFlowModel


class _CVInterVolTag:
    pass


class _TemperatureInterVolTag:
    pass


class _MatDensityInterVolTag:
    pass


class _GradCVInterVolTag:
    pass


class _GradTemperatureInterVolTag:
    pass


class _MultiphysicsCoupledHarmonicMeanBoundaryComponent:
    """Setup of the coupling between both sides of the interface."""

    def __init__(self, state_plus, interface_noslip, interface_radiation,
            boundary_velocity=None, grad_cv_plus=None, grad_t_plus=None):
        r"""Initialize coupling interface.

        Arguments *grad_cv_plus* and *grad_t_plus*, are only required if the
        boundary will be used to compute the viscous flux.

        Parameters
        ----------
        state_plus: :class:`~mirgecom.gas_model.FluidState`
            Fluid state on either wall or fluid side.

        interface_noslip: bool
            If `True`, interface boundaries on the fluid side will be treated
            as no-slip walls. If `False` they will be treated as slip walls.

        interface_radiation: bool
            If `True`, radiation is accounted for as a sink term in the coupling.
            If `False` they will be treated as slip walls.

        boundary_velocity: float
            If there is a normal velocity prescribed at the boundary.

        grad_cv_plus: :class:`meshmode.dof_array.DOFArray` or None
            CV gradient from the wall side.

        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None
            Temperature gradient from the wall side.
        """
        self._state_plus = state_plus
        self._no_slip = interface_noslip
        self._radiation = interface_radiation
        self._boundary_velocity = boundary_velocity
        self._grad_cv_plus = grad_cv_plus
        self._grad_t_plus = grad_t_plus

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce inviscid BC at the interface."""
        # This is only used for inviscid flux, so I dont think I have to exactly
        # use the plus side but rather a state that enforces the BC. Thus,
        # a "no-slip wall"-like implementation is used.
        # For the viscous fluxes/gradients, the actual plus state is used.
        if self._boundary_velocity is not None:
            # actx = state_minus.cv.mass.array_context
            # normal = actx.thaw(dcoll.normal(dd_bdry))
            # momentum_plus = (2.0*state_minus.cv.mass*self._boundary_velocity*normal
            #                  - state_minus.cv.momentum)

            # cv_plus = make_conserved(dim=dcoll.dim,
            #                          mass=state_minus.cv.mass,
            #                          energy=state_minus.cv.energy,
            #                          momentum=momentum_plus,
            #                          species_mass=state_minus.cv.species_mass)

            # return ViscousFluidState(cv=cv_plus, dv=state_minus.dv,
            #                          tv=state_minus.tv)
            raise NotImplementedError

        if self._no_slip is True:
            # use the same implementation from no-slip walls
            cv_plus = make_conserved(dim=dcoll.dim,
                                     mass=state_minus.cv.mass,
                                     energy=state_minus.cv.energy,
                                     momentum=-state_minus.cv.momentum,
                                     species_mass=state_minus.cv.species_mass)

            return ViscousFluidState(cv=cv_plus, dv=state_minus.dv,
                                     tv=state_minus.tv)

        raise NotImplementedError

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce viscous BC at the interface."""
        actx = state_minus.array_context

        state_plus = project_from_base(dcoll, dd_bdry, self._state_plus)

        u_bc = self.velocity_bc(dcoll, dd_bdry, state_minus)
        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus)
        y_bc = self.species_mass_fractions_bc(dcoll, dd_bdry, state_minus)

        material_densities = state_minus.wv.material_densities \
            if isinstance(gas_model, PorousFlowModel) else None

        # the gradient that matters is the intrinsic density, not the bulk one.
        # thus, has to consider the presence of 'epsilon' to avoid jump in mass.
        if isinstance(gas_model, PorousFlowModel):
            # wall side, where the plus is the fluid
            epsilon_plus = state_minus.wv.void_fraction
            mass_bc = 0.5*(state_minus.mass_density
                           + state_plus.mass_density*epsilon_plus)

            tau = gas_model.decomposition_progress(material_densities)
            from mirgecom.wall_model import PorousWallVars
            from mirgecom.gas_model import PorousFlowFluidState
            wv = PorousWallVars(
                material_densities=material_densities,
                tau=tau,
                density=gas_model.solid_density(material_densities),
                void_fraction=gas_model.wall_eos.void_fraction(tau),
                emissivity=gas_model.wall_eos.emissivity(tau),
                permeability=gas_model.wall_eos.permeability(tau),
                tortuosity=gas_model.wall_eos.tortuosity(tau))

            total_energy_bc = mass_bc*gas_model.eos.get_internal_energy(t_bc, y_bc) \
                                + wv.density*gas_model.wall_eos.enthalpy(t_bc, tau)

            smoothness_mu = actx.np.zeros_like(state_minus.cv.mass)
            smoothness_kappa = actx.np.zeros_like(state_minus.cv.mass)
            smoothness_beta = actx.np.zeros_like(state_minus.cv.mass)

        else:
            # fluid side, where the plus is the wall
            epsilon_plus = state_plus.wv.void_fraction
            mass_bc = 0.5*(state_minus.mass_density
                           + state_plus.mass_density/epsilon_plus)

            internal_energy_bc = gas_model.eos.get_internal_energy(
                temperature=t_bc, species_mass_fractions=y_bc)
            total_energy_bc = mass_bc*(internal_energy_bc + 0.5*np.dot(u_bc, u_bc))

            smoothness_mu = state_minus.dv.smoothness_mu
            smoothness_kappa = state_minus.dv.smoothness_kappa
            smoothness_beta = state_minus.dv.smoothness_beta

        cv_bc = make_conserved(dim=dcoll.dim,
                               mass=mass_bc,
                               momentum=mass_bc*u_bc,
                               energy=total_energy_bc,
                               species_mass=mass_bc*y_bc)

        state_bc = make_fluid_state(cv=cv_bc, gas_model=gas_model,
                                    temperature_seed=t_bc,
                                    smoothness_mu=smoothness_mu,
                                    smoothness_kappa=smoothness_kappa,
                                    smoothness_beta=smoothness_beta,
                                    material_densities=material_densities)

        new_mu = state_minus.tv.viscosity

        new_kappa = state_minus.tv.thermal_conductivity if self._radiation else \
            harmonic_mean(state_minus.tv.thermal_conductivity,
                          state_plus.tv.thermal_conductivity)

        new_diff = harmonic_mean(state_minus.tv.species_diffusivity,
                                 state_plus.tv.species_diffusivity)

        new_tv = GasTransportVars(
            bulk_viscosity=state_bc.tv.bulk_viscosity,
            viscosity=new_mu,
            thermal_conductivity=new_kappa,
            species_diffusivity=new_diff)

        if isinstance(gas_model, PorousFlowModel):
            return PorousFlowFluidState(cv=state_bc.cv, dv=state_bc.dv,
                                        tv=new_tv, wv=wv)

        return ViscousFluidState(cv=state_bc.cv, dv=state_bc.dv, tv=new_tv)

    def velocity_bc(self, dcoll, dd_bdry, state_minus):
        """Velocity at the interface.

        The velocity can be non-zero due to the blowing or zero if no-slip.
        """
        u_minus = state_minus.cv.velocity

        # if there is mass blowing normal to the surface
        if self._boundary_velocity is not None:
            # actx = state_minus.cv.mass.array_context
            # normal = actx.thaw(dcoll.normal(dd_bdry))
            # return self._boundary_velocity*normal

            raise NotImplementedError

        # if the coupling involves a no-slip wall:
        if self._no_slip:
            return u_minus*0.0

        raise NotImplementedError

    def species_mass_fractions_bc(self, dcoll, dd_bdry, state_minus):
        """Species mass fractions at the interface."""
        y_minus = state_minus.species_mass_fractions
        y_plus = project_from_base(dcoll, dd_bdry,
                                   self._state_plus.species_mass_fractions)

        actx = state_minus.array_context
        diff_minus = state_minus.tv.species_diffusivity
        diff_plus = project_from_base(dcoll, dd_bdry,
                                      self._state_plus.tv.species_diffusivity)
        diff_sum = diff_minus + diff_plus
        # for cases with zero species diffusion
        return actx.np.where(
            actx.np.greater(diff_sum, 0.0),
            (y_minus * diff_minus + y_plus * diff_plus)/diff_sum,
            y_plus + y_minus)

    def temperature_bc(self, dcoll, dd_bdry, state_minus):
        """Temperature at the interface."""
        t_minus = state_minus.temperature
        t_plus = project_from_base(dcoll, dd_bdry, self._state_plus.temperature)

        kappa_minus = state_minus.tv.thermal_conductivity
        kappa_plus = project_from_base(dcoll, dd_bdry,
                                       self._state_plus.tv.thermal_conductivity)
        kappa_sum = kappa_minus + kappa_plus
        return (t_minus * kappa_minus + t_plus * kappa_plus)/kappa_sum

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Gradient averaging for viscous flux."""
        if self._grad_cv_plus is None:
            raise ValueError(
                "Boundary does not have external CV gradient data.")

        grad_cv_plus = project_from_base(dcoll, dd_bdry, self._grad_cv_plus)

        # if the coupling involves a no-slip wall:
        if self._no_slip:
            grad_cv_bc = (grad_cv_plus + grad_cv_minus)/2
            return make_conserved(dim=dcoll.dim,
                                  mass=grad_cv_bc.mass,
                                  momentum=grad_cv_minus.momentum,
                                  energy=grad_cv_bc.energy,
                                  species_mass=grad_cv_bc.species_mass)

        raise NotImplementedError

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus):
        """Gradient averaging for viscous flux."""
        if self._grad_t_plus is None:
            raise ValueError(
                "Boundary does not have external temperature gradient data.")

        grad_t_plus = project_from_base(dcoll, dd_bdry, self._grad_t_plus)
        return (grad_t_plus + grad_t_minus)/2


class InterfaceFluidBoundary(MengaldoBoundaryCondition):
    """Boundary for the fluid side on the interface between fluid and wall.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: viscous_divergence_flux
    """

    def __init__(self, state_plus, interface_noslip, interface_radiation,
                 boundary_velocity=None,
                 grad_cv_plus=None, grad_t_plus=None,
                 flux_penalty_amount=None, lengthscales_minus=None):
        r"""Initialize InterfaceFluidBoundary.

        Arguments *grad_cv_plus*, *grad_t_plus*, *flux_penalty_amount*, and
        *lengthscales_minus* are only required if the boundary will be used to
        compute the viscous flux.

        Parameters
        ----------
        state_plus: :class:`~mirgecom.gas_model.FluidState`
            Fluid state from the wall side, i.e., porous flow.

        interface_noslip: bool
            If `True`, interface boundaries on the fluid side will be treated
            as no-slip walls. If `False` they will be treated as slip walls.

        interface_radiation: bool
            If `True`, radiation is accounted for as a sink term in the coupling.
            If `False` they will be treated as slip walls.

        boundary_velocity: float
            If there is a prescribed velocity at the boundary.

        grad_cv_plus: :class:`meshmode.dof_array.DOFArray` or None
            CV gradient from the wall side.

        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None
            Temperature gradient from the wall side.

        flux_penalty_amount: float or None
            Coefficient $c$ for the interior penalty on the heat flux.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None
            Characteristic mesh spacing $h^-$.
        """
        self._state_plus = state_plus
        self._radiation = interface_radiation
        self._grad_cv_plus = grad_cv_plus
        self._grad_t_plus = grad_t_plus
        self._flux_penalty_amount = flux_penalty_amount
        self._lengthscales_minus = lengthscales_minus

        self._coupled = _MultiphysicsCoupledHarmonicMeanBoundaryComponent(
            state_plus=state_plus,
            boundary_velocity=boundary_velocity,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            grad_cv_plus=grad_cv_plus,
            grad_t_plus=grad_t_plus)

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce inviscid BC at the interface."""
        # Don't bother replacing anything since this is just for inviscid
        return self._coupled.state_plus(dcoll, dd_bdry, gas_model, state_minus, **kwargs)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce viscous BC at the interface."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._coupled.state_bc(dcoll, dd_bdry, gas_model, state_minus)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        r"""Interface temperature to enforce viscous BC."""
        if self._radiation:
            t_plus = project_from_base(dcoll, dd_bdry, self._state_plus.temperature)
            return 0.5*(t_plus + state_minus.temperature)
        return self._coupled.temperature_bc(dcoll, dd_bdry, state_minus)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            normal, **kwargs):
        """Gradient of CV to enforce viscous BC."""
        return self._coupled.grad_cv_bc(dcoll, dd_bdry, gas_model, state_minus,
                                        grad_cv_minus, normal, **kwargs)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        r"""Gradient of temperature to enforce viscous BC.

        If using radiation, uses $\nabla T_{bc} = \nabla T^{-}$.
        Else, the simple averaging of gradient at both sides is used instead.
        """
        if self._radiation:
            return grad_t_minus
        return self._coupled.grad_temperature_bc(dcoll, dd_bdry, grad_t_minus)

    def viscous_divergence_flux(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, numerical_flux_func=None, **kwargs):
        r"""Return the viscous flux at the interface boundaries.

        It is defined by
        :meth:`mirgecom.boundary.MengaldoBoundaryCondition.viscous_divergence_flux`

        For radiation cases:
        ..math::

             \nabla T_{bc} = \nabla T^-
             \kappa_{bc} = \kappa^-
        """
        dd_bdry = as_dofdesc(dd_bdry)

        base_viscous_flux = super().viscous_divergence_flux(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, numerical_flux_func=numerical_flux_func,
            grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus, **kwargs)

        penalization = 0.0

        # state_plus = project_from_base(dcoll, dd_bdry, self._state_plus)

        # state_bc = self.state_bc(dcoll=dcoll, dd_bdry=dd_bdry,
        #     gas_model=gas_model, state_minus=state_minus, **kwargs)

        # lengthscales_minus = project_from_base(
        #     dcoll, dd_bdry, self._lengthscales_minus)

        # penalty = self._flux_penalty_amount/lengthscales_minus
        # tau_momentum = penalty * state_bc.tv.viscosity
        # tau_energy = penalty * state_bc.tv.thermal_conductivity
        # tau_species = penalty * state_bc.tv.species_diffusivity

        # penalization = make_conserved(dim=dcoll.dim,
        #     mass=state_minus.cv.mass*0.0,
        #     energy=tau_energy*(
        #         state_plus.temperature - state_minus.temperature),
        #     momentum=tau_momentum*(
        #         state_plus.cv.momentum - state_minus.cv.momentum),
        #     species_mass=tau_species*(
        #         state_plus.cv.species_mass - state_minus.cv.species_mass)
        # )

        return base_viscous_flux + penalization


class InterfaceWallBoundary(MengaldoBoundaryCondition):
    """Boundary for the wall side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: viscous_divergence_flux
    """

    def __init__(self, state_plus, interface_noslip, interface_radiation,
                 wall_emissivity=None, sigma=None, u_ambient=None,
                 grad_cv_plus=None, grad_t_plus=None,
                 flux_penalty_amount=None, lengthscales_minus=None):
        r"""Initialize InterfaceWallBoundary.

        Arguments *grad_cv_plus*, *grad_t_plus*, *flux_penalty_amount*, and
        *lengthscales_minus* are only required if the boundary will be used to
        compute the viscous flux.

        Parameters
        ----------
        state_plus: :class:`~mirgecom.gas_model.FluidState`
            Fluid state from the fluid side.

        interface_noslip: bool
            If `True`, interface boundaries on the fluid side will be treated
            as no-slip walls. If `False` they will be treated as slip walls.

        grad_cv_plus: :class:`meshmode.dof_array.DOFArray` or None
            CV gradient from the fluid side.

        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None
            Temperature gradient from the fluid side.

        flux_penalty_amount: float or None
            Coefficient $c$ for the interior penalty on the viscous fluxes.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None
            Characteristic mesh spacing $h^-$.
        """
        self._state_plus = state_plus
        self._radiation = interface_radiation
        self._grad_cv_plus = grad_cv_plus
        self._grad_t_plus = grad_t_plus
        self._emissivity = wall_emissivity
        self._sigma = sigma
        self._u_ambient = u_ambient
        self._flux_penalty_amount = flux_penalty_amount
        self._lengthscales_minus = lengthscales_minus

        self._coupled = _MultiphysicsCoupledHarmonicMeanBoundaryComponent(
            state_plus=state_plus,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            grad_cv_plus=grad_cv_plus,
            grad_t_plus=grad_t_plus)

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce inviscid BC at the interface."""
        # Don't bother replacing anything since this is just for inviscid
        return self._coupled.state_plus(dcoll, dd_bdry, state_minus, **kwargs)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce viscous BC at the interface."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._coupled.state_bc(dcoll, dd_bdry, gas_model, state_minus)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Interface temperature to enforce viscous BC.

        If using radiation, uses $T_{bc} = T^{+}$. Else, the simple averaging
        of temperature at both sides is used instead.
        """
        if self._radiation:
            return project_from_base(dcoll, dd_bdry, self._state_plus.temperature)
        return self._coupled.temperature_bc(dcoll, dd_bdry, state_minus)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Gradient of CV to enforce viscous BC."""
        return self._coupled.grad_cv_bc(dcoll, dd_bdry, gas_model, state_minus,
                                        grad_cv_minus, normal, **kwargs)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Gradient of temperature to enforce viscous BC."""
        if self._radiation:
            return grad_t_minus
        return self._coupled.grad_temperature_bc(dcoll, dd_bdry, grad_t_minus)

    def viscous_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
            grad_cv_minus, grad_t_minus, numerical_flux_func=None, **kwargs):
        """Return the viscous flux at the interface boundaries.

        It is defined by
        :meth:`mirgecom.boundary.MengaldoBoundaryCondition.viscous_divergence_flux`
        """
        dd_bdry = as_dofdesc(dd_bdry)

        state_plus = project_from_base(dcoll, dd_bdry, self._state_plus)

        state_bc = self.state_bc(dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                                 state_minus=state_minus, **kwargs)

        base_viscous_flux = super().viscous_divergence_flux(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, numerical_flux_func=numerical_flux_func,
            grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus, **kwargs)

        # lengthscales_minus = project_from_base(dcoll, dd_bdry,
        #                                        self._lengthscales_minus)

        # penalty = self._flux_penalty_amount/lengthscales_minus
        # tau_momentum = penalty * state_bc.tv.viscosity
        # tau_energy = penalty * state_bc.tv.thermal_conductivity
        # tau_species = penalty * state_bc.tv.species_diffusivity

        if self._radiation:
            radiation_spec = [self._emissivity is None,
                              self._sigma is None,
                              self._u_ambient is None]
            if sum(radiation_spec) != 0:
                raise TypeError(
                    "Arguments 'wall_emissivity', 'sigma' and 'ambient_temperature'"
                    "are required if using surface radiation.")

            actx = state_minus.cv.mass.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            kappa_plus = state_plus.thermal_conductivity
            grad_t_plus = project_from_base(dcoll, dd_bdry, self._grad_t_plus)

            # species flux
            from arraycontext import outer
            grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                          grad_cv_minus)
            grad_y_plus = species_mass_fraction_gradient(state_plus.cv,
                                                         self._grad_cv_plus)
            rho_bc = 0.5*(state_minus.cv.mass + state_plus.cv.mass)
            grad_y_bc = 1.0/(state_minus.wv.void_fraction + 1.0)*(
                state_minus.wv.void_fraction*grad_y_minus + grad_y_plus)
            d_bc = state_bc.species_diffusivity
            y_bc = state_bc.species_mass_fractions

            species_flux = -rho_bc*(
                d_bc.reshape(-1, 1)*grad_y_bc
                - outer(y_bc, sum(d_bc.reshape(-1, 1)*grad_y_bc)))
            species_mass_flux = -species_flux@normal

            # heat flux due to shear, species and thermal conduction
            # tau = viscous_stress_tensor(state_bc, grad_cv_bc)
            h_alpha = state_bc.species_enthalpies
            heat_flux = (
                # np.dot(tau, state_bc.velocity)
                - sum(h_alpha.reshape(-1, 1) * species_flux)
                - diffusion_flux(kappa_plus, grad_t_plus))@normal

            # radiation sink term
            wall_emissivity = project_from_base(dcoll, dd_bdry, self._emissivity)
            radiation = wall_emissivity * self._sigma * (
                state_minus.temperature**4 - self._u_ambient**4)

            penalization = 0.0
            # penalization = make_conserved(dim=dcoll.dim,
            #     mass=state_minus.cv.mass*0.0,
            #     energy=tau_energy*(
            #         state_plus.temperature - state_minus.temperature),
            #     momentum=tau_momentum*(
            #         state_plus.cv.momentum - state_minus.cv.momentum),
            #     species_mass=tau_species*(
            #         state_plus.cv.species_mass - state_minus.cv.species_mass)
            # )

            return (base_viscous_flux.replace(energy=heat_flux - radiation,
                                              species_mass=species_mass_flux)
                    + penalization)

        else:
            penalization = 0.0
            # penalization = make_conserved(dim=dcoll.dim,
            #     mass=state_minus.cv.mass*0.0,
            #     energy=tau_energy*(
            #         state_plus.temperature - state_minus.temperature),
            #     momentum=tau_momentum*(
            #         state_plus.cv.momentum - state_minus.cv.momentum),
            #     species_mass=tau_species*(
            #         state_plus.cv.species_mass - state_minus.cv.species_mass)
            # )

            return base_viscous_flux + penalization


def _getattr_ish(obj, name):
    if obj is None:
        return None
    else:
        return getattr(obj, name)


def _state_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, gas_model_fluid, gas_model_wall,
        fluid_state, wall_state, limiter_func_fluid, limiter_func_wall):
    """Exchange state across the fluid-wall interface."""
    actx = fluid_state.cv.mass.array_context

    # exchange CV
    pairwise_cv = {(fluid_dd, wall_dd):
                   (fluid_state.cv, wall_state.cv)}
    cv_pairs = inter_volume_trace_pairs(
        dcoll, pairwise_cv, comm_tag=_CVInterVolTag)

    fluid_to_wall_cv_tpairs = cv_pairs[fluid_dd, wall_dd]
    wall_to_fluid_cv_tpairs = cv_pairs[wall_dd, fluid_dd]

    # exchange temperature
    pairwise_temp = {(fluid_dd, wall_dd):
                     (fluid_state.temperature, wall_state.temperature)}
    temperature_seed_pairs = inter_volume_trace_pairs(
        dcoll, pairwise_temp, comm_tag=_TemperatureInterVolTag)

    fluid_to_wall_tseed_tpairs = temperature_seed_pairs[fluid_dd, wall_dd]
    wall_to_fluid_tseed_tpairs = temperature_seed_pairs[wall_dd, fluid_dd]

    # exchange material densities. It is zero on the fluid side...
    from pytools.obj_array import make_obj_array
    ncomponents = len(wall_state.wv.material_densities)
    if ncomponents == 1:
        zeros = actx.np.zeros_like(fluid_state.cv.mass)
    else:
        zeros = make_obj_array([actx.np.zeros_like(fluid_state.cv.mass)
                                for i in range(ncomponents)])
    pairwise_dens = {(fluid_dd, wall_dd):
                     (zeros, wall_state.wv.material_densities)}
    material_densities_pairs = inter_volume_trace_pairs(
        dcoll, pairwise_dens, comm_tag=_MatDensityInterVolTag)

    fluid_to_wall_mass_tpairs = material_densities_pairs[fluid_dd, wall_dd]
    wall_to_fluid_mass_tpairs = material_densities_pairs[wall_dd, fluid_dd]

    return {
        (fluid_dd, wall_dd): [TracePair(
            cv_pair.dd,
            interior=make_fluid_state(
                cv_pair.int, gas_model_wall,
                temperature_seed=_getattr_ish(tseed_pair, "int"),
                material_densities=_getattr_ish(material_densities_pair, "int"),
                limiter_func=limiter_func_wall, limiter_dd=cv_pair.dd),
            exterior=make_fluid_state(
                cv_pair.ext, gas_model_fluid,
                temperature_seed=_getattr_ish(tseed_pair, "ext"),
                material_densities=_getattr_ish(material_densities_pair, "ext"),
                limiter_func=limiter_func_fluid, limiter_dd=cv_pair.dd))
            for cv_pair, tseed_pair, material_densities_pair in zip(
                fluid_to_wall_cv_tpairs,
                fluid_to_wall_tseed_tpairs,
                fluid_to_wall_mass_tpairs)],
        (wall_dd, fluid_dd): [TracePair(
            cv_pair.dd,
            interior=make_fluid_state(
                cv_pair.int, gas_model_fluid,
                temperature_seed=_getattr_ish(tseed_pair, "int"),
                material_densities=_getattr_ish(material_densities_pair, "int"),
                limiter_func=limiter_func_fluid, limiter_dd=cv_pair.dd),
            exterior=make_fluid_state(
                cv_pair.ext, gas_model_wall,
                temperature_seed=_getattr_ish(tseed_pair, "ext"),
                material_densities=_getattr_ish(material_densities_pair, "ext"),
                limiter_func=limiter_func_wall, limiter_dd=cv_pair.dd))
            for cv_pair, tseed_pair, material_densities_pair in zip(
                wall_to_fluid_cv_tpairs,
                wall_to_fluid_tseed_tpairs,
                wall_to_fluid_mass_tpairs)]}


def _grad_cv_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, fluid_grad_cv, wall_grad_cv):
    """Exchange CV gradients across the fluid-wall interface."""
    pairwise_grad_cv = {(fluid_dd, wall_dd): (fluid_grad_cv, wall_grad_cv)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_grad_cv, comm_tag=_GradCVInterVolTag)


def _grad_temperature_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, fluid_grad_temperature, wall_grad_temperature):
    """Exchange temperature gradient across the fluid-wall interface."""
    pairwise_grad_temperature = {
        (fluid_dd, wall_dd):
            (fluid_grad_temperature, wall_grad_temperature)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_grad_temperature, comm_tag=_GradTemperatureInterVolTag)


def add_interface_boundaries_no_grad(
        dcoll, fluid_dd, wall_dd,
        gas_model_fluid, gas_model_wall,
        fluid_state, wall_state,
        fluid_boundaries, wall_boundaries,
        interface_noslip, interface_radiation,
        *,
        limiter_func_fluid=None, limiter_func_wall=None,
        boundary_velocity=None):
    r"""Return the interface of the subdomains for gradient calculation.

    Used to apply the boundary fluxes at the interface between fluid and
    wall domains.

    Parameters
    ----------
    dcoll: class:`~grudge.discretization.DiscretizationCollection`
        A discretization collection encapsulating the DG elements

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

    wall_state: :class:`~mirgecom.gas_model.FluidState`
        Wall state object with the conserved state and dependent
        quantities for the wall volume.

    interface_noslip: bool
        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    interface_radiation: bool
        If `True`, interface includes a radiation sink term in the heat flux
        on the wall side and prescribes the temperature on the fluid side.
        Additional arguments *wall_emissivity*, *sigma*, and
        *ambient_temperature* are required if enabled.

    boundary_velocity: float or :class:`meshmode.dof_array.DOFArray`
        Normal velocity due to pyrolysis outgas. Only required for simplified
        analysis of composite material.

    Returns
    -------
        The tuple `(fluid_interface_boundaries, wall_interface_boundaries)`.
    """
    if interface_noslip is False:
        from warnings import warn
        warn("Only no-slip coupling is implemented", UserWarning, stacklevel=2)
        raise NotImplementedError

    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    # Construct boundaries for the fluid-wall interface; no gradients
    # yet because that's what we're trying to compute

    state_inter_volume_trace_pairs = \
        _state_inter_volume_trace_pairs(dcoll, fluid_dd, wall_dd,
                                        gas_model_fluid, gas_model_wall,
                                        fluid_state, wall_state,
                                        limiter_func_fluid, limiter_func_wall)

    # Construct interface boundaries without gradient

    fluid_interface_boundaries_no_grad = {
        state_tpair.dd.domain_tag: InterfaceFluidBoundary(
            state_plus=state_tpair.ext,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            boundary_velocity=boundary_velocity)
        for state_tpair in state_inter_volume_trace_pairs[wall_dd, fluid_dd]}

    wall_interface_boundaries_no_grad = {
        state_tpair.dd.domain_tag: InterfaceWallBoundary(
            state_plus=state_tpair.ext,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation)
        for state_tpair in state_inter_volume_trace_pairs[fluid_dd, wall_dd]}

    # Augment the domain boundaries with the interface boundaries

    fluid_all_boundaries_no_grad = {}
    fluid_all_boundaries_no_grad.update(fluid_boundaries)
    fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    wall_all_boundaries_no_grad = {}
    wall_all_boundaries_no_grad.update(wall_boundaries)
    wall_all_boundaries_no_grad.update(wall_interface_boundaries_no_grad)

    return fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad


def add_interface_boundaries(
        dcoll,
        fluid_dd, wall_dd,
        fluid_gas_model, wall_gas_model,
        fluid_state, wall_state,
        fluid_grad_cv, wall_grad_cv,
        fluid_grad_temperature, wall_grad_temperature,
        fluid_boundaries, wall_boundaries,
        interface_noslip, interface_radiation,
        *,
        limiter_func_fluid=None, limiter_func_wall=None,
        boundary_velocity=None,
        wall_emissivity=None, sigma=None, ambient_temperature=None,
        wall_penalty_amount=None):
    r"""Return the interface of the subdomains for viscous fluxes.

    Used to apply the boundary fluxes at the interface between fluid and
    wall domains.

    Parameters
    ----------
    dcoll: class:`~grudge.discretization.DiscretizationCollection`
        A discretization collection encapsulating the DG elements

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

    wall_state: :class:`~mirgecom.gas_model.FluidState`
        Wall state object with the conserved state and dependent
        quantities for the wall volume.

    interface_noslip: bool
        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    interface_radiation: bool
        If `True`, interface includes a radiation sink term in the heat flux
        on the wall side and prescribes the temperature on the fluid side. See
        :class:`InterfaceWallBoundary`
        for details. Additional arguments *wall_emissivity*, *sigma*, and
        *ambient_temperature* are required if enabled.

    wall_emissivity: float or :class:`meshmode.dof_array.DOFArray`
        Emissivity of the wall material.

    sigma: float
        Stefan-Boltzmann constant.

    ambient_temperature: :class:`meshmode.dof_array.DOFArray`
        Ambient temperature of the environment.

    boundary_velocity: float or :class:`meshmode.dof_array.DOFArray`
        Normal velocity due to pyrolysis outgas. Only required for simplified
        analysis of composite material.

    wall_penalty_amount: float
        Coefficient $c$ for the interior penalty on the heat flux. See
        :class:`InterfaceFluidBoundary`
        for details.

    Returns
    -------
        The tuple `(fluid_interface_boundaries, wall_interface_boundaries)`.
    """
    if wall_penalty_amount is None:
        # FIXME: After verifying the form of the penalty term, figure out what
        # value makes sense to use as a default here
        wall_penalty_amount = 0.05

    # Set up the interface boundaries

    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    # Exchange information

    state_inter_volume_trace_pairs = \
        _state_inter_volume_trace_pairs(dcoll,
                                        fluid_dd, wall_dd,
                                        fluid_gas_model, wall_gas_model,
                                        fluid_state, wall_state,
                                        limiter_func_fluid, limiter_func_wall)

    grad_cv_inter_vol_tpairs = _grad_cv_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, fluid_grad_cv, wall_grad_cv)

    grad_temperature_inter_vol_tpairs = _grad_temperature_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, fluid_grad_temperature, wall_grad_temperature)

    # Construct interface boundaries with temperature gradient

    import grudge.op as op
    from grudge.dt_utils import characteristic_lengthscales
    actx = fluid_state.cv.mass.array_context
    fluid_lengthscales = characteristic_lengthscales(actx, dcoll, fluid_dd)
    wall_lengthscales = characteristic_lengthscales(actx, dcoll, wall_dd)

    fluid_interface_boundaries = {
        state_tpair.dd.domain_tag: InterfaceFluidBoundary(
            state_plus=state_tpair.ext,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            boundary_velocity=boundary_velocity,
            grad_cv_plus=grad_cv_tpair.ext,
            grad_t_plus=grad_temperature_tpair.ext,
            # lengthscales_minus=op.project(dcoll, fluid_dd, state_tpair.dd,
            #                               fluid_lengthscales),
            flux_penalty_amount=wall_penalty_amount)
        for state_tpair, grad_cv_tpair, grad_temperature_tpair in zip(
            state_inter_volume_trace_pairs[wall_dd, fluid_dd],
            grad_cv_inter_vol_tpairs[wall_dd, fluid_dd],
            grad_temperature_inter_vol_tpairs[wall_dd, fluid_dd])}

    wall_interface_boundaries = {
        state_tpair.dd.domain_tag: InterfaceWallBoundary(
            state_plus=state_tpair.ext,
            interface_noslip=interface_noslip,
            interface_radiation=interface_radiation,
            wall_emissivity=wall_emissivity,
            sigma=sigma,
            u_ambient=ambient_temperature,
            grad_cv_plus=grad_cv_tpair.ext,
            grad_t_plus=grad_temperature_tpair.ext,
            # lengthscales_minus=op.project(dcoll, wall_dd, state_tpair.dd,
            #                               wall_lengthscales),
            flux_penalty_amount=wall_penalty_amount)
        for state_tpair, grad_cv_tpair, grad_temperature_tpair in zip(
            state_inter_volume_trace_pairs[fluid_dd, wall_dd],
            grad_cv_inter_vol_tpairs[fluid_dd, wall_dd],
            grad_temperature_inter_vol_tpairs[fluid_dd, wall_dd])}

    # Augment the domain boundaries with the interface boundaries

    fluid_all_boundaries = {}
    fluid_all_boundaries.update(fluid_boundaries)
    fluid_all_boundaries.update(fluid_interface_boundaries)

    wall_all_boundaries = {}
    wall_all_boundaries.update(wall_boundaries)
    wall_all_boundaries.update(wall_interface_boundaries)

    return fluid_all_boundaries, wall_all_boundaries
