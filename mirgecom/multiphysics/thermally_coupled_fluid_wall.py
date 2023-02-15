r"""Operator for thermally-coupled fluid and wall.

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
(:module:`mirgecom.navierstokes) with a wall subdomain governed by the heat
equation (:module:`mirgecom.diffusion`) by enforcing continuity of temperature
and heat flux across their interface.

.. autofunction:: get_interface_boundaries
.. autofunction:: coupled_grad_t_operator
.. autofunction:: coupled_ns_heat_operator

.. autoclass:: InterfaceFluidBoundary
.. autoclass:: InterfaceWallBoundary
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

from dataclasses import replace
import numpy as np
from functools import partial

from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    as_dofdesc,
)
import grudge.op as op

from mirgecom.boundary import (
    PrescribedFluidBoundary,
    _SlipBoundaryComponent,
    _NoSlipBoundaryComponent,
    _ImpermeableBoundaryComponent,
    _inviscid_flux_for_prescribed_state_mengaldo,
    _viscous_flux_for_prescribed_state_mengaldo)
from mirgecom.fluid import make_conserved
from mirgecom.flux import num_flux_central
from mirgecom.inviscid import inviscid_facial_flux_rusanov
from mirgecom.viscous import viscous_facial_flux_harmonic
from mirgecom.gas_model import (
    make_fluid_state,
    make_operator_fluid_states,
)
from mirgecom.navierstokes import (
    grad_t_operator as fluid_grad_t_operator,
    ns_operator,
)
from mirgecom.artificial_viscosity import av_laplacian_operator
from mirgecom.diffusion import (
    DiffusionBoundary,
    grad_operator as wall_grad_t_operator,
    diffusion_operator,
)


class _TemperatureInterVolTag:
    pass


class _KappaInterVolTag:
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


def _project_from_base(dcoll, dd_bdry, field):
    if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
        dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
        return op.project(dcoll, dd_bdry_base, dd_bdry, field)
    else:
        return field


# Note: callback function inputs have no default on purpose so that they can't be
# accidentally omitted
def _interface_viscous_flux(
        dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
        grad_t_minus, *, penalty_amount, lengthscales,
        state_bc_func, temperature_plus_func, grad_cv_bc_func,
        grad_temperature_bc_func, **kwargs):
    """Return the boundary flux for the divergence of the viscous flux.

    Returns the standard Mengaldo viscous flux plus a temperature interior penalty
    term on the energy.
    """
    dd_bdry = as_dofdesc(dd_bdry)

    state_bc = state_bc_func(
        dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
        state_minus=state_minus, **kwargs)

    flux_without_penalty = _viscous_flux_for_prescribed_state_mengaldo(
        dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model, state_minus=state_minus,
        grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus,
        # FIXME: Not sure if this optimization is necessary
        state_bc_func=lambda *a, **kw: state_bc,
        grad_cv_bc_func=grad_cv_bc_func,
        grad_temperature_bc_func=grad_temperature_bc_func,
        **kwargs)

    lengthscales = _project_from_base(dcoll, dd_bdry, lengthscales)

    tau = penalty_amount * state_bc.thermal_conductivity / lengthscales

    t_minus = state_minus.temperature
    t_plus = temperature_plus_func(
        dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model, state_minus=state_minus,
        **kwargs)

    return replace(
        flux_without_penalty,
        # NS and diffusion use opposite sign conventions for flux; hence penalty
        # is added here instead of subtracted
        energy=flux_without_penalty.energy + tau * (t_plus - t_minus))


class _ThermallyCoupledBoundaryComponent:
    def __init__(self, kappa_plus, t_plus, grad_t_plus=None):
        self._kappa_plus = kappa_plus
        self._t_plus = t_plus
        self._grad_t_plus = grad_t_plus

    def kappa_plus(self, dcoll, dd_bdry):
        return _project_from_base(dcoll, dd_bdry, self._kappa_plus)

    def kappa_bc(self, dcoll, dd_bdry, kappa_minus):
        actx = kappa_minus.array_context

        def harmonic_mean(x, y):
            x_plus_y = actx.np.where(actx.np.greater(x + y, 0*x), x + y, 0*x+1)
            return 2*x*y/x_plus_y

        kappa_plus = _project_from_base(dcoll, dd_bdry, self._kappa_plus)

        return harmonic_mean(kappa_minus, kappa_plus)

    def temperature_plus(self, dcoll, dd_bdry):
        return _project_from_base(dcoll, dd_bdry, self._t_plus)

    def temperature_bc(self, dcoll, dd_bdry, t_minus):
        t_plus = _project_from_base(dcoll, dd_bdry, self._t_plus)
        return (t_minus + t_plus)/2

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus):
        if self._grad_t_plus is None:
            raise ValueError(
                "Boundary does not have external temperature gradient data.")
        grad_t_plus = _project_from_base(dcoll, dd_bdry, self._grad_t_plus)
        return (grad_t_plus + grad_t_minus)/2


def _replace_kappa(state, kappa):
    new_tv = replace(state.tv, thermal_conductivity=kappa)
    return replace(state, tv=new_tv)


class InterfaceFluidSlipBoundary(PrescribedFluidBoundary):
    """Interface boundary condition for the fluid side."""

    # FIXME: Incomplete docs
    def __init__(
            self, kappa_plus, t_plus, grad_t_plus=None,
            heat_flux_penalty_amount=None, lengthscales=None):
        """Initialize InterfaceFluidBoundary."""
        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.state_bc,
            inviscid_flux_func=partial(
                _inviscid_flux_for_prescribed_state_mengaldo,
                state_plus_func=self.state_plus),
            viscous_flux_func=partial(
                _interface_viscous_flux,
                penalty_amount=heat_flux_penalty_amount,
                lengthscales=lengthscales,
                state_bc_func=self.state_bc,
                temperature_plus_func=self.temperature_plus,
                grad_cv_bc_func=self.grad_cv_bc,
                grad_temperature_bc_func=self.grad_temperature_bc),
            boundary_temperature_func=self.temperature_plus,
            boundary_gradient_cv_func=self.grad_cv_bc,
            boundary_gradient_temperature_func=self.grad_temperature_bc,
            boundary_grad_av_func=self.grad_av_plus)

        self._thermally_coupled = _ThermallyCoupledBoundaryComponent(
            kappa_plus=kappa_plus,
            t_plus=t_plus,
            grad_t_plus=grad_t_plus)
        self._slip = _SlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv

        mom_plus = self._slip.momentum_plus(cv_minus.momentum, normal)

        t_plus = self._thermally_coupled.temperature_plus(dcoll, dd_bdry)

        internal_energy_plus = (
            cv_minus.mass
            * gas_model.eos.get_internal_energy(
                temperature=t_plus,
                species_mass_fractions=cv_minus.species_mass_fractions))
        total_energy_plus = (
            internal_energy_plus
            # Kinetic energy is the same
            + gas_model.eos.kinetic_energy(cv_minus))

        cv_plus = make_conserved(
            state_minus.dim,
            mass=cv_minus.mass,
            energy=total_energy_plus,
            momentum=mom_plus,
            species_mass=cv_minus.species_mass)

        kappa_plus = self._thermally_coupled.kappa_plus(dcoll, dd_bdry)

        return _replace_kappa(
            make_fluid_state(
                cv=cv_plus, gas_model=gas_model,
                temperature_seed=state_minus.temperature),
            kappa_plus)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        dd_bdry = as_dofdesc(dd_bdry)
        normal = actx.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv

        mom_bc = self._slip.momentum_bc(cv_minus.momentum, normal)

        t_bc = self._thermally_coupled.temperature_bc(
            dcoll, dd_bdry, state_minus.temperature)

        internal_energy_bc = (
            cv_minus.mass
            * gas_model.eos.get_internal_energy(
                temperature=t_bc,
                species_mass_fractions=cv_minus.species_mass_fractions))
        total_energy_bc = (
            internal_energy_bc
            + 0.5*np.dot(mom_bc, mom_bc)/cv_minus.mass)

        cv_bc = make_conserved(
            state_minus.dim,
            mass=cv_minus.mass,
            energy=total_energy_bc,
            momentum=mom_bc,
            species_mass=cv_minus.species_mass)

        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)
        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

        return _replace_kappa(
            make_fluid_state(
                cv=cv_bc, gas_model=gas_model,
                temperature_seed=state_minus.temperature),
            kappa_bc)

    # FIXME: Remove this?
    def grad_av_plus(self, dcoll, dd_bdry, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary."""
        # Grab some boundary-relevant data
        actx = grad_av_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # Apply a Neumann condition on the energy gradient
        # Should probably compute external energy gradient using external temperature
        # gradient, but that is a can of worms
        grad_energy_plus = \
            grad_av_minus.energy - 2 * np.dot(grad_av_minus.energy, nhat) * nhat

        # uh oh - we don't have the necessary data to compute grad_y from grad_av
        # from mirgecom.fluid import species_mass_fraction_gradient
        # grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
        #                                               grad_cv_minus)
        # grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)
        # grad_species_mass_plus = 0.*grad_y_plus
        # This re-stuffs grad_y+ back into grad_cv+, skipit; we did not split AVs
        # for i in range(state_minus.nspecies):
        #    grad_species_mass_plus[i] = (state_minus.mass_density*grad_y_plus[i]
        #        + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)
        grad_species_mass_plus = (
            grad_av_minus.species_mass
            - np.outer(grad_av_minus.species_mass @ nhat, nhat))

        return make_conserved(
            grad_av_minus.dim, mass=grad_av_minus.mass, energy=grad_energy_plus,
            momentum=grad_av_minus.momentum, species_mass=grad_species_mass_plus)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, state_bc, grad_cv_minus,
            **kwargs):
        """
        Return external grad(CV) used in the boundary calculation of viscous flux.

        Specify the velocity gradients on the external state to ensure zero
        energy and momentum flux due to shear stresses.

        Gradients of species mass fractions are set to zero in the normal direction
        to ensure zero flux of species across the boundary.
        """
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        grad_mom_bc = self._slip.grad_momentum_bc(
            state_minus, state_bc, grad_cv_minus, normal)

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return make_conserved(
            grad_cv_minus.dim,
            mass=grad_cv_minus.mass,
            energy=grad_cv_minus.energy,
            momentum=grad_mom_bc,
            species_mass=grad_species_mass_bc)

    def temperature_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior T on the boundary."""
        return self._thermally_coupled.temperature_plus(dcoll, dd_bdry)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, **kwargs):
        """Get grad(T) on the boundary."""
        return self._thermally_coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)


class InterfaceFluidBoundary(PrescribedFluidBoundary):
    """Interface boundary condition for the fluid side."""

    # FIXME: Incomplete docs
    def __init__(
            self, kappa_plus, t_plus, grad_t_plus=None,
            heat_flux_penalty_amount=None, lengthscales=None):
        """Initialize InterfaceFluidBoundary."""
        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.state_bc,
            inviscid_flux_func=partial(
                _inviscid_flux_for_prescribed_state_mengaldo,
                state_plus_func=self.state_plus),
            viscous_flux_func=partial(
                _interface_viscous_flux,
                penalty_amount=heat_flux_penalty_amount,
                lengthscales=lengthscales,
                state_bc_func=self.state_bc,
                temperature_plus_func=self.temperature_plus,
                grad_cv_bc_func=self.grad_cv_bc,
                grad_temperature_bc_func=self.grad_temperature_bc),
            boundary_temperature_func=self.temperature_plus,
            boundary_gradient_cv_func=self.grad_cv_bc,
            boundary_gradient_temperature_func=self.grad_temperature_bc,
            boundary_grad_av_func=self.grad_av_plus)

        self._thermally_coupled = _ThermallyCoupledBoundaryComponent(
            kappa_plus=kappa_plus,
            t_plus=t_plus,
            grad_t_plus=grad_t_plus)
        self._no_slip = _NoSlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv

        mom_plus = self._no_slip.momentum_plus(cv_minus.momentum, normal)

        t_plus = self._thermally_coupled.temperature_plus(dcoll, dd_bdry)

        internal_energy_plus = (
            cv_minus.mass
            * gas_model.eos.get_internal_energy(
                temperature=t_plus,
                species_mass_fractions=cv_minus.species_mass_fractions))
        total_energy_plus = (
            internal_energy_plus
            # Kinetic energy is the same
            + gas_model.eos.kinetic_energy(cv_minus))

        cv_plus = make_conserved(
            state_minus.dim,
            mass=cv_minus.mass,
            energy=total_energy_plus,
            momentum=mom_plus,
            species_mass=cv_minus.species_mass)

        kappa_plus = self._thermally_coupled.kappa_plus(dcoll, dd_bdry)

        return _replace_kappa(
            make_fluid_state(
                cv=cv_plus, gas_model=gas_model,
                temperature_seed=state_minus.temperature),
            kappa_plus)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        dd_bdry = as_dofdesc(dd_bdry)
        normal = actx.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv

        mom_bc = self._no_slip.momentum_bc(cv_minus.momentum, normal)

        t_bc = self._thermally_coupled.temperature_bc(
            dcoll, dd_bdry, state_minus.temperature)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc,
            species_mass_fractions=cv_minus.species_mass_fractions)

        # Velocity is pinned to 0 here, no kinetic energy
        total_energy_bc = cv_minus.mass * internal_energy_bc

        cv_bc = make_conserved(
            state_minus.dim,
            mass=cv_minus.mass,
            energy=total_energy_bc,
            momentum=mom_bc,
            species_mass=cv_minus.species_mass)

        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)
        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

        return _replace_kappa(
            make_fluid_state(
                cv=cv_bc, gas_model=gas_model,
                temperature_seed=state_minus.temperature),
            kappa_bc)

    # FIXME: Remove this?
    def grad_av_plus(self, dcoll, dd_bdry, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary."""
        # Grab some boundary-relevant data
        actx = grad_av_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # Apply a Neumann condition on the energy gradient
        # Should probably compute external energy gradient using external temperature
        # gradient, but that is a can of worms
        grad_energy_plus = \
            grad_av_minus.energy - 2 * np.dot(grad_av_minus.energy, nhat) * nhat

        # uh oh - we don't have the necessary data to compute grad_y from grad_av
        # from mirgecom.fluid import species_mass_fraction_gradient
        # grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
        #                                               grad_cv_minus)
        # grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)
        # grad_species_mass_plus = 0.*grad_y_plus
        # This re-stuffs grad_y+ back into grad_cv+, skipit; we did not split AVs
        # for i in range(state_minus.nspecies):
        #    grad_species_mass_plus[i] = (state_minus.mass_density*grad_y_plus[i]
        #        + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)
        grad_species_mass_plus = (
            grad_av_minus.species_mass
            - np.outer(grad_av_minus.species_mass @ nhat, nhat))

        return make_conserved(
            grad_av_minus.dim, mass=grad_av_minus.mass, energy=grad_energy_plus,
            momentum=grad_av_minus.momentum, species_mass=grad_species_mass_plus)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, state_bc, grad_cv_minus,
            **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return make_conserved(
            grad_cv_minus.dim,
            mass=grad_cv_minus.mass,
            energy=grad_cv_minus.energy,
            momentum=grad_cv_minus.momentum,
            species_mass=grad_species_mass_bc)

    def temperature_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior T on the boundary."""
        return self._thermally_coupled.temperature_plus(dcoll, dd_bdry)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, **kwargs):
        """Get grad(T) on the boundary."""
        return self._thermally_coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)


class InterfaceWallBoundary(DiffusionBoundary):
    """Interface boundary condition for the wall side."""

    # FIXME: Incomplete docs
    def __init__(self, kappa_plus, u_plus, grad_u_plus=None):
        """Initialize InterfaceWallBoundary."""
        self.kappa_plus = kappa_plus
        self.u_plus = u_plus
        self.grad_u_plus = grad_u_plus

    def get_grad_flux(self, dcoll, dd_bdry, kappa_minus, u_minus):  # noqa: D102
        actx = u_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        kappa_plus = _project_from_base(dcoll, dd_bdry, self.kappa_plus)
        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_plus)

        u_plus = _project_from_base(dcoll, dd_bdry, self.u_plus)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)

        from mirgecom.diffusion import grad_facial_flux
        return grad_facial_flux(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, penalty_amount=None):  # noqa: D102
        if self.grad_u_plus is None:
            raise ValueError(
                "Boundary does not have external gradient data.")

        actx = u_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        kappa_plus = _project_from_base(dcoll, dd_bdry, self.kappa_plus)
        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_plus)

        u_plus = _project_from_base(dcoll, dd_bdry, self.u_plus)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)

        grad_u_plus = _project_from_base(dcoll, dd_bdry, self.grad_u_plus)
        grad_u_tpair = TracePair(
            dd_bdry, interior=grad_u_minus, exterior=grad_u_plus)

        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)

        from mirgecom.diffusion import diffusion_facial_flux
        return diffusion_facial_flux(
            kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal,
            penalty_amount=penalty_amount)


def _kappa_inter_volume_trace_pairs(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_kappa):
    actx = fluid_state.array_context
    fluid_kappa = fluid_state.thermal_conductivity

    from meshmode.dof_array import DOFArray
    if not isinstance(fluid_kappa, DOFArray):
        fluid_kappa = fluid_kappa * (dcoll.zeros(actx, dd=fluid_dd) + 1)
    if not isinstance(wall_kappa, DOFArray):
        wall_kappa = wall_kappa * (dcoll.zeros(actx, dd=wall_dd) + 1)
    pairwise_kappa = {
        (fluid_dd, wall_dd): (fluid_kappa, wall_kappa)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_kappa, comm_tag=_KappaInterVolTag)


def _temperature_inter_volume_trace_pairs(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_temperature):
    pairwise_temperature = {
        (fluid_dd, wall_dd):
            (fluid_state.temperature, wall_temperature)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_temperature, comm_tag=_TemperatureInterVolTag)


def _grad_temperature_inter_volume_trace_pairs(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_grad_temperature, wall_grad_temperature):
    pairwise_grad_temperature = {
        (fluid_dd, wall_dd):
            (fluid_grad_temperature, wall_grad_temperature)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_grad_temperature, comm_tag=_GradTemperatureInterVolTag)


# FIXME: Make kappa optional like the gradient?
def get_interface_boundaries(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_kappa, wall_temperature,
        interface_noslip=True,
        fluid_grad_temperature=None, wall_grad_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        *,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _kappa_inter_vol_tpairs=None,
        _temperature_inter_vol_tpairs=None,
        _grad_temperature_inter_vol_tpairs=None):
    # FIXME: Incomplete docs
    """Get the fluid-wall interface boundaries."""
    if interface_noslip:
        fluid_bc_class = InterfaceFluidBoundary
    else:
        fluid_bc_class = InterfaceFluidSlipBoundary

    include_gradient = (
        fluid_grad_temperature is not None and wall_grad_temperature is not None)

    if _kappa_inter_vol_tpairs is None:
        kappa_inter_vol_tpairs = _kappa_inter_volume_trace_pairs(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_kappa)
    else:
        kappa_inter_vol_tpairs = _kappa_inter_vol_tpairs

    if _temperature_inter_vol_tpairs is None:
        temperature_inter_vol_tpairs = _temperature_inter_volume_trace_pairs(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_temperature)
    else:
        temperature_inter_vol_tpairs = _temperature_inter_vol_tpairs

    if include_gradient:
        if _grad_temperature_inter_vol_tpairs is None:
            grad_temperature_inter_vol_tpairs = \
                _grad_temperature_inter_volume_trace_pairs(
                    dcoll,
                    gas_model,
                    fluid_dd, wall_dd,
                    fluid_grad_temperature, wall_grad_temperature)
        else:
            grad_temperature_inter_vol_tpairs = _grad_temperature_inter_vol_tpairs
    else:
        grad_temperature_inter_vol_tpairs = None

    if include_gradient:
        from grudge.dt_utils import characteristic_lengthscales
        fluid_lengthscales = (
            characteristic_lengthscales(
                fluid_state.array_context, dcoll, fluid_dd)
            * (0*fluid_state.temperature+1))

        fluid_interface_boundaries = {
            kappa_tpair.dd.domain_tag: fluid_bc_class(
                kappa_tpair.ext,
                temperature_tpair.ext,
                grad_temperature_tpair.ext,
                wall_penalty_amount,
                lengthscales=op.project(dcoll,
                    fluid_dd, temperature_tpair.dd, fluid_lengthscales))
            for kappa_tpair, temperature_tpair, grad_temperature_tpair in zip(
                kappa_inter_vol_tpairs[wall_dd, fluid_dd],
                temperature_inter_vol_tpairs[wall_dd, fluid_dd],
                grad_temperature_inter_vol_tpairs[wall_dd, fluid_dd])}

        wall_interface_boundaries = {
            kappa_tpair.dd.domain_tag: InterfaceWallBoundary(
                kappa_tpair.ext,
                temperature_tpair.ext,
                grad_temperature_tpair.ext)
            for kappa_tpair, temperature_tpair, grad_temperature_tpair in zip(
                kappa_inter_vol_tpairs[fluid_dd, wall_dd],
                temperature_inter_vol_tpairs[fluid_dd, wall_dd],
                grad_temperature_inter_vol_tpairs[fluid_dd, wall_dd])}
    else:
        fluid_interface_boundaries = {
            kappa_tpair.dd.domain_tag: fluid_bc_class(
                kappa_tpair.ext,
                temperature_tpair.ext)
            for kappa_tpair, temperature_tpair in zip(
                kappa_inter_vol_tpairs[wall_dd, fluid_dd],
                temperature_inter_vol_tpairs[wall_dd, fluid_dd])}

        wall_interface_boundaries = {
            kappa_tpair.dd.domain_tag: InterfaceWallBoundary(
                kappa_tpair.ext,
                temperature_tpair.ext)
            for kappa_tpair, temperature_tpair in zip(
                kappa_inter_vol_tpairs[fluid_dd, wall_dd],
                temperature_inter_vol_tpairs[fluid_dd, wall_dd])}

    return fluid_interface_boundaries, wall_interface_boundaries


def coupled_grad_t_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        *,
        time=0.,
        fluid_numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _kappa_inter_vol_tpairs=None,
        _temperature_inter_vol_tpairs=None,
        _fluid_operator_states_quad=None,
        _fluid_interface_boundaries_no_grad=None,
        _wall_interface_boundaries_no_grad=None):
    # FIXME: Incomplete docs
    """Compute grad(T) of the coupled fluid-wall system."""
    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    assert (
        (_fluid_interface_boundaries_no_grad is None)
        == (_wall_interface_boundaries_no_grad is None)), (
        "Expected both _fluid_interface_boundaries_no_grad and "
        "_wall_interface_boundaries_no_grad or neither")

    if _fluid_interface_boundaries_no_grad is None:
        fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
            get_interface_boundaries(
                dcoll,
                gas_model,
                fluid_dd, wall_dd,
                fluid_state, wall_kappa, wall_temperature,
                _kappa_inter_vol_tpairs=_kappa_inter_vol_tpairs,
                _temperature_inter_vol_tpairs=_temperature_inter_vol_tpairs)
    else:
        fluid_interface_boundaries_no_grad = _fluid_interface_boundaries_no_grad
        wall_interface_boundaries_no_grad = _wall_interface_boundaries_no_grad

    fluid_all_boundaries_no_grad = {}
    fluid_all_boundaries_no_grad.update(fluid_boundaries)
    fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    wall_all_boundaries_no_grad = {}
    wall_all_boundaries_no_grad.update(wall_boundaries)
    wall_all_boundaries_no_grad.update(wall_interface_boundaries_no_grad)

    return (
        fluid_grad_t_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=time, numerical_flux_func=fluid_numerical_flux_func,
            quadrature_tag=quadrature_tag, dd=fluid_dd,
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
        fluid_gradient_numerical_flux_func=num_flux_central,
        inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
        viscous_numerical_flux_func=viscous_facial_flux_harmonic,
        interface_noslip=True,
        use_av=False,
        av_kwargs=None,
        return_gradients=False,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE):
    # FIXME: Incomplete docs
    """Compute RHS of the coupled fluid-wall system."""
    if wall_penalty_amount is None:
        # *shrug*
        wall_penalty_amount = 0.05

    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    kappa_inter_vol_tpairs = _kappa_inter_volume_trace_pairs(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_kappa)

    # FIXME: Maybe better to project CV and recompute T instead?
    temperature_inter_vol_tpairs = _temperature_inter_volume_trace_pairs(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_temperature)

    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        get_interface_boundaries(
            dcoll=dcoll,
            gas_model=gas_model,
            fluid_dd=fluid_dd, wall_dd=wall_dd,
            fluid_state=fluid_state, wall_kappa=wall_kappa,
            wall_temperature=wall_temperature,
            interface_noslip=interface_noslip,
            _kappa_inter_vol_tpairs=kappa_inter_vol_tpairs,
            _temperature_inter_vol_tpairs=temperature_inter_vol_tpairs)

    fluid_all_boundaries_no_grad = {}
    fluid_all_boundaries_no_grad.update(fluid_boundaries)
    fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    fluid_operator_states_quad = make_operator_fluid_states(
        dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=fluid_dd, comm_tag=_FluidOpStatesTag)

    fluid_grad_temperature, wall_grad_temperature = coupled_grad_t_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        time=time,
        fluid_numerical_flux_func=fluid_gradient_numerical_flux_func,
        quadrature_tag=quadrature_tag,
        _kappa_inter_vol_tpairs=kappa_inter_vol_tpairs,
        _temperature_inter_vol_tpairs=temperature_inter_vol_tpairs,
        _fluid_operator_states_quad=fluid_operator_states_quad,
        _fluid_interface_boundaries_no_grad=fluid_interface_boundaries_no_grad,
        _wall_interface_boundaries_no_grad=wall_interface_boundaries_no_grad)

    fluid_interface_boundaries, wall_interface_boundaries = \
        get_interface_boundaries(
            dcoll=dcoll,
            gas_model=gas_model,
            fluid_dd=fluid_dd, wall_dd=wall_dd,
            fluid_state=fluid_state, wall_kappa=wall_kappa,
            wall_temperature=wall_temperature,
            fluid_grad_temperature=fluid_grad_temperature,
            wall_grad_temperature=wall_grad_temperature,
            interface_noslip=interface_noslip,
            wall_penalty_amount=wall_penalty_amount,
            _kappa_inter_vol_tpairs=kappa_inter_vol_tpairs,
            _temperature_inter_vol_tpairs=temperature_inter_vol_tpairs)

    fluid_all_boundaries = {}
    fluid_all_boundaries.update(fluid_boundaries)
    fluid_all_boundaries.update(fluid_interface_boundaries)

    wall_all_boundaries = {}
    wall_all_boundaries.update(wall_boundaries)
    wall_all_boundaries.update(wall_interface_boundaries)

    ns_result = ns_operator(
        dcoll, gas_model, fluid_state, fluid_all_boundaries,
        time=time, quadrature_tag=quadrature_tag, dd=fluid_dd,
        viscous_numerical_flux_func=viscous_numerical_flux_func,
        return_gradients=return_gradients,
        operator_states_quad=fluid_operator_states_quad,
        grad_t=fluid_grad_temperature, comm_tag=_FluidOperatorTag)

    if return_gradients:
        fluid_rhs, fluid_grad_cv, fluid_grad_temperature = ns_result
    else:
        fluid_rhs = ns_result

    if use_av:
        if av_kwargs is None:
            av_kwargs = {}
        fluid_rhs = fluid_rhs + av_laplacian_operator(
            dcoll, fluid_all_boundaries, fluid_state, quadrature_tag=quadrature_tag,
            dd=fluid_dd, **av_kwargs)

    diffusion_result = diffusion_operator(
        dcoll, wall_kappa, wall_all_boundaries, wall_temperature,
        return_grad_u=return_gradients, penalty_amount=wall_penalty_amount,
        quadrature_tag=quadrature_tag, dd=wall_dd, grad_u=wall_grad_temperature,
        comm_tag=_WallOperatorTag)

    if return_gradients:
        wall_rhs, wall_grad_temperature = diffusion_result
    else:
        wall_rhs = diffusion_result

    if return_gradients:
        return (
            fluid_rhs, wall_rhs, fluid_grad_cv, fluid_grad_temperature,
            wall_grad_temperature)
    else:
        return fluid_rhs, wall_rhs
