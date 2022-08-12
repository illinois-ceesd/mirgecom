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

from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    as_dofdesc,
)
import grudge.op as op

from mirgecom.boundary import PrescribedFluidBoundary
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


class InterfaceFluidBoundary(PrescribedFluidBoundary):
    """Interface boundary condition for the fluid side."""

    # FIXME: Incomplete docs
    def __init__(
            self, ext_t, ext_kappa, ext_grad_t=None, heat_flux_penalty_amount=None,
            lengthscales=None):
        """Initialize InterfaceFluidBoundary."""
        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.get_external_state,
            boundary_grad_av_func=self.get_external_grad_av,
            boundary_temperature_func=self.get_external_t,
            boundary_gradient_temperature_func=self.get_external_grad_t,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_gradient_cv_func=self.get_external_grad_cv
        )
        self.ext_t = ext_t
        self.ext_kappa = ext_kappa
        self.ext_grad_t = ext_grad_t
        self.heat_flux_penalty_amount = heat_flux_penalty_amount
        self.lengthscales = lengthscales

    # NOTE: The BC for species mass is y_+ = y_-, I think that is OK here
    #       The BC for species mass fraction gradient is set down inside the
    #       `viscous_flux` method.
    def get_external_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            ext_t = op.project(dcoll, dd_bdry_base, dd_bdry, self.ext_t)
            ext_kappa = op.project(dcoll, dd_bdry_base, dd_bdry, self.ext_kappa)
        else:
            ext_t = self.ext_t
            ext_kappa = self.ext_kappa

        # Cancel out the momentum
        cv_minus = state_minus.cv
        ext_mom = -cv_minus.momentum

        # Compute the energy
        ext_internal_energy = (
            cv_minus.mass
            * gas_model.eos.get_internal_energy(
                temperature=ext_t,
                species_mass_fractions=cv_minus.species_mass_fractions))
        ext_kinetic_energy = gas_model.eos.kinetic_energy(cv_minus)
        ext_energy = ext_internal_energy + ext_kinetic_energy

        # Form the external boundary solution with the new momentum and energy.
        ext_cv = make_conserved(
            dim=state_minus.dim, mass=cv_minus.mass, energy=ext_energy,
            momentum=ext_mom, species_mass=cv_minus.species_mass)

        def replace_thermal_conductivity(state, kappa):
            new_tv = replace(state.tv, thermal_conductivity=kappa)
            return replace(state, tv=new_tv)

        return replace_thermal_conductivity(
            make_fluid_state(
                cv=ext_cv, gas_model=gas_model,
                temperature_seed=state_minus.temperature),
            ext_kappa)

    def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """Return Riemann flux using state with mom opposite of interior state."""
        dd_bdry = as_dofdesc(dd_bdry)
        # NOTE: For the inviscid/advection part we set mom_+ = -mom_-, and
        #       use energy_+ = energy_-, per [Mengaldo_2014]_.
        wall_cv = make_conserved(dim=state_minus.dim,
                                 mass=state_minus.mass_density,
                                 momentum=-state_minus.momentum_density,
                                 energy=state_minus.energy_density,
                                 species_mass=state_minus.species_mass_density)
        wall_state = make_fluid_state(cv=wall_cv, gas_model=gas_model,
                                      temperature_seed=state_minus.temperature)
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=wall_state)

        # Grab a unit normal to the boundary
        nhat = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        return numerical_flux_func(state_pair, gas_model, nhat)

    def get_external_grad_av(self, dcoll, dd_bdry, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary."""
        # Grab some boundary-relevant data
        actx = grad_av_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # Apply a Neumann condition on the energy gradient
        # Should probably compute external energy gradient using external temperature
        # gradient, but that is a can of worms
        ext_grad_energy = \
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
        ext_grad_species_mass = (
            grad_av_minus.species_mass
            - np.outer(grad_av_minus.species_mass @ nhat, nhat))

        return make_conserved(
            grad_av_minus.dim, mass=grad_av_minus.mass, energy=ext_grad_energy,
            momentum=grad_av_minus.momentum, species_mass=ext_grad_species_mass)

    def get_external_grad_cv(self, state_minus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        from mirgecom.fluid import species_mass_fraction_gradient
        grad_y_minus = species_mass_fraction_gradient(state_minus.cv, grad_cv_minus)
        grad_y_bc = grad_y_minus - np.outer(grad_y_minus@normal, normal)
        grad_species_mass_plus = 0.*grad_y_bc

        for i in range(state_minus.nspecies):
            grad_species_mass_plus[i] = (state_minus.mass_density*grad_y_bc[i]
                + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        return make_conserved(grad_cv_minus.dim,
                              mass=grad_cv_minus.mass,
                              energy=grad_cv_minus.energy,
                              momentum=grad_cv_minus.momentum,
                              species_mass=grad_species_mass_plus)

    def viscous_wall_flux(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, numerical_flux_func=viscous_facial_flux_harmonic,
            **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        if self.heat_flux_penalty_amount is None:
            raise ValueError("Boundary does not have heat flux penalty amount.")
        if self.lengthscales is None:
            raise ValueError("Boundary does not have length scales data.")

        dd_bdry = as_dofdesc(dd_bdry)
        dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        # FIXME: Need to examine [Mengaldo_2014]_ - specifically momentum terms
        state_plus = self.get_external_state(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)
        grad_cv_bc = self.get_external_grad_cv(
            state_minus=state_minus, grad_cv_minus=grad_cv_minus, normal=normal,
            **kwargs)

        grad_t_plus = self.get_external_grad_t(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, grad_cv_minus=grad_cv_minus,
            grad_t_minus=grad_t_minus)

        def harmonic_mean(x, y):
            x_plus_y = actx.np.where(actx.np.greater(x + y, 0*x), x + y, 0*x+1)
            return 2*x*y/x_plus_y

        def replace_kappa(state, kappa):
            from dataclasses import replace
            new_tv = replace(state.tv, thermal_conductivity=kappa)
            return replace(state, tv=new_tv)

        kappa_harmonic_mean = harmonic_mean(
            state_minus.tv.thermal_conductivity,
            state_plus.tv.thermal_conductivity)

        state_pair_with_harmonic_mean_coefs = TracePair(
            dd_bdry,
            interior=replace_kappa(state_minus, kappa_harmonic_mean),
            exterior=replace_kappa(state_plus, kappa_harmonic_mean))

        f_int = viscous_flux(
            state_pair_with_harmonic_mean_coefs.int, grad_cv_bc, grad_t_minus)
        f_ext = viscous_flux(
            state_pair_with_harmonic_mean_coefs.ext, grad_cv_bc, grad_t_plus)
        f_pair = TracePair(dd_bdry, interior=f_int, exterior=f_ext)

        lengthscales = op.project(dcoll, dd_bdry_base, dd_bdry, self.lengthscales)

        tau = (
            self.heat_flux_penalty_amount * kappa_harmonic_mean / lengthscales)

        # NS and diffusion use opposite sign conventions for flux; hence penalty
        # is added here instead of subtracted
        flux_without_penalty = f_pair.avg @ normal
        return replace(
            flux_without_penalty,
            energy=(
                flux_without_penalty.energy
                + tau * (state_plus.temperature - state_minus.temperature)))

    def get_external_t(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior T on the boundary."""
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            return op.project(dcoll, dd_bdry_base, dd_bdry, self.ext_t)
        else:
            return self.ext_t

    def get_external_grad_t(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, **kwargs):
        """Get the exterior grad(T) on the boundary."""
        if self.ext_grad_t is None:
            raise ValueError(
                "Boundary does not have external temperature gradient data.")
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            return op.project(dcoll, dd_bdry_base, dd_bdry, self.ext_grad_t)
        else:
            return self.ext_grad_t


class InterfaceWallBoundary(DiffusionBoundary):
    """Interface boundary condition for the wall side."""

    # FIXME: Incomplete docs
    def __init__(self, u_plus, kappa_plus, grad_u_plus=None):
        """Initialize InterfaceWallBoundary."""
        self.u_plus = u_plus
        self.kappa_plus = kappa_plus
        self.grad_u_plus = grad_u_plus

    def get_grad_flux(self, dcoll, dd_bdry, u_minus):  # noqa: D102
        actx = u_minus.array_context
        u_plus = self.get_external_u(dcoll, dd_bdry)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        from mirgecom.diffusion import grad_facial_flux
        return grad_facial_flux(u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, u_minus, kappa_minus, grad_u_minus,
            lengthscales_minus, penalty_amount=None):  # noqa: D102
        actx = u_minus.array_context
        u_plus = self.get_external_u(dcoll, dd_bdry)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)
        kappa_plus = self.get_external_kappa(dcoll, dd_bdry)
        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_plus)
        grad_u_plus = self.get_external_grad_u(dcoll, dd_bdry)
        grad_u_tpair = TracePair(
            dd_bdry, interior=grad_u_minus, exterior=grad_u_plus)
        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        from mirgecom.diffusion import diffusion_facial_flux
        return diffusion_facial_flux(
            u_tpair, kappa_tpair, grad_u_tpair, lengthscales_tpair, normal,
            penalty_amount=penalty_amount)

    def get_external_u(self, dcoll, dd_bdry):
        """Get the exterior u on the boundary."""
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            return op.project(dcoll, dd_bdry_base, dd_bdry, self.u_plus)
        else:
            return self.u_plus

    def get_external_kappa(self, dcoll, dd_bdry):
        """Get the exterior grad(u) on the boundary."""
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            return op.project(dcoll, dd_bdry_base, dd_bdry, self.kappa_plus)
        else:
            return self.kappa_plus

    def get_external_grad_u(self, dcoll, dd_bdry):
        """Get the exterior grad(u) on the boundary."""
        if self.grad_u_plus is None:
            raise ValueError(
                "Boundary does not have external gradient data.")
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            return op.project(dcoll, dd_bdry_base, dd_bdry, self.grad_u_plus)
        else:
            return self.grad_u_plus


def _temperature_inter_volume_trace_pairs(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_state, wall_temperature):
    pairwise_temperature = {
        (fluid_dd, wall_dd):
            (fluid_state.temperature, wall_temperature)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_temperature, comm_tag=_TemperatureInterVolTag)


def _kappa_inter_volume_trace_pairs(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_state, wall_temperature):
    pairwise_kappa = {
        (fluid_dd, wall_dd):
            (fluid_state.thermal_conductivity, wall_model.thermal_conductivity)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_kappa, comm_tag=_KappaInterVolTag)


def _grad_temperature_inter_volume_trace_pairs(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_grad_temperature, wall_grad_temperature):
    pairwise_grad_temperature = {
        (fluid_dd, wall_dd):
            (fluid_grad_temperature, wall_grad_temperature)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_grad_temperature, comm_tag=_GradTemperatureInterVolTag)


def get_interface_boundaries(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_state, wall_temperature,
        fluid_grad_temperature=None, wall_grad_temperature=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        *,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _temperature_inter_vol_tpairs=None,
        _kappa_inter_vol_tpairs=None,
        _grad_temperature_inter_vol_tpairs=None):
    # FIXME: Incomplete docs
    """Get the fluid-wall interface boundaries."""
    include_gradient = (
        fluid_grad_temperature is not None and wall_grad_temperature is not None)

    if _temperature_inter_vol_tpairs is None:
        temperature_inter_vol_tpairs = _temperature_inter_volume_trace_pairs(
            dcoll,
            gas_model, wall_model,
            fluid_dd, wall_dd,
            fluid_state, wall_temperature)
    else:
        temperature_inter_vol_tpairs = _temperature_inter_vol_tpairs

    if _kappa_inter_vol_tpairs is None:
        kappa_inter_vol_tpairs = _kappa_inter_volume_trace_pairs(
            dcoll,
            gas_model, wall_model,
            fluid_dd, wall_dd,
            fluid_state, wall_temperature)
    else:
        kappa_inter_vol_tpairs = _kappa_inter_vol_tpairs

    if include_gradient:
        if _grad_temperature_inter_vol_tpairs is None:
            grad_temperature_inter_vol_tpairs = \
                _grad_temperature_inter_volume_trace_pairs(
                    dcoll,
                    gas_model, wall_model,
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
            temperature_tpair.dd.domain_tag: InterfaceFluidBoundary(
                temperature_tpair.ext,
                kappa_tpair.ext,
                grad_temperature_tpair.ext,
                wall_penalty_amount,
                lengthscales=op.project(dcoll,
                    fluid_dd, temperature_tpair.dd, fluid_lengthscales))
            for temperature_tpair, kappa_tpair, grad_temperature_tpair in zip(
                temperature_inter_vol_tpairs[wall_dd, fluid_dd],
                kappa_inter_vol_tpairs[wall_dd, fluid_dd],
                grad_temperature_inter_vol_tpairs[wall_dd, fluid_dd])}

        wall_interface_boundaries = {
            temperature_tpair.dd.domain_tag: InterfaceWallBoundary(
                temperature_tpair.ext,
                kappa_tpair.ext,
                grad_temperature_tpair.ext)
            for temperature_tpair, kappa_tpair, grad_temperature_tpair in zip(
                temperature_inter_vol_tpairs[fluid_dd, wall_dd],
                kappa_inter_vol_tpairs[fluid_dd, wall_dd],
                grad_temperature_inter_vol_tpairs[fluid_dd, wall_dd])}
    else:
        fluid_interface_boundaries = {
            temperature_tpair.dd.domain_tag: InterfaceFluidBoundary(
                temperature_tpair.ext,
                kappa_tpair.ext)
            for temperature_tpair, kappa_tpair in zip(
                temperature_inter_vol_tpairs[wall_dd, fluid_dd],
                kappa_inter_vol_tpairs[wall_dd, fluid_dd])}

        wall_interface_boundaries = {
            temperature_tpair.dd.domain_tag: InterfaceWallBoundary(
                temperature_tpair.ext,
                kappa_tpair.ext)
            for temperature_tpair, kappa_tpair in zip(
                temperature_inter_vol_tpairs[fluid_dd, wall_dd],
                kappa_inter_vol_tpairs[fluid_dd, wall_dd])}

    return fluid_interface_boundaries, wall_interface_boundaries


def coupled_grad_t_operator(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature, *,
        time=0.,
        fluid_numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _temperature_inter_vol_tpairs=None,
        _kappa_inter_vol_tpairs=None,
        _fluid_operator_states_quad=None):
    # FIXME: Incomplete docs
    """Compute grad(T) of the coupled fluid-wall system."""
    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        get_interface_boundaries(
            dcoll,
            gas_model, wall_model,
            fluid_dd, wall_dd,
            fluid_state, wall_temperature,
            _temperature_inter_vol_tpairs=_temperature_inter_vol_tpairs,
            _kappa_inter_vol_tpairs=_kappa_inter_vol_tpairs)

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
            operator_states_quad=_fluid_operator_states_quad),
        wall_grad_t_operator(
            dcoll, wall_all_boundaries_no_grad, wall_temperature,
            quadrature_tag=quadrature_tag, dd=wall_dd))


def _heat_operator(
        dcoll, wall_model, boundaries, temperature, *,
        penalty_amount, quadrature_tag, dd,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _grad_temperature=None):
    return (
        1/(wall_model.density * wall_model.heat_capacity)
        * diffusion_operator(
            dcoll, wall_model.thermal_conductivity, boundaries, temperature,
            penalty_amount=penalty_amount, quadrature_tag=quadrature_tag,
            dd=dd, grad_u=_grad_temperature))


def coupled_ns_heat_operator(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature, *,
        time=0., wall_time_scale=1,
        fluid_gradient_numerical_flux_func=num_flux_central,
        inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
        use_av=False,
        av_kwargs=None,
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

    temperature_inter_vol_tpairs = _temperature_inter_volume_trace_pairs(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_state, wall_temperature)

    kappa_inter_vol_tpairs = _kappa_inter_volume_trace_pairs(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_state, wall_temperature)

    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        get_interface_boundaries(
            dcoll,
            gas_model, wall_model,
            fluid_dd, wall_dd,
            fluid_state, wall_temperature,
            _temperature_inter_vol_tpairs=temperature_inter_vol_tpairs,
            _kappa_inter_vol_tpairs=kappa_inter_vol_tpairs)

    fluid_all_boundaries_no_grad = {}
    fluid_all_boundaries_no_grad.update(fluid_boundaries)
    fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    fluid_operator_states_quad = make_operator_fluid_states(
        dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=fluid_dd)

    fluid_grad_temperature, wall_grad_temperature = coupled_grad_t_operator(
        dcoll,
        gas_model, wall_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature,
        time=time,
        fluid_numerical_flux_func=fluid_gradient_numerical_flux_func,
        quadrature_tag=quadrature_tag,
        _temperature_inter_vol_tpairs=temperature_inter_vol_tpairs,
        _kappa_inter_vol_tpairs=kappa_inter_vol_tpairs,
        _fluid_operator_states_quad=fluid_operator_states_quad)

    fluid_interface_boundaries, wall_interface_boundaries = \
        get_interface_boundaries(
            dcoll,
            gas_model, wall_model,
            fluid_dd, wall_dd,
            fluid_state, wall_temperature,
            fluid_grad_temperature, wall_grad_temperature,
            wall_penalty_amount=wall_penalty_amount,
            _temperature_inter_vol_tpairs=temperature_inter_vol_tpairs,
            _kappa_inter_vol_tpairs=kappa_inter_vol_tpairs)

    fluid_all_boundaries = {}
    fluid_all_boundaries.update(fluid_boundaries)
    fluid_all_boundaries.update(fluid_interface_boundaries)

    wall_all_boundaries = {}
    wall_all_boundaries.update(wall_boundaries)
    wall_all_boundaries.update(wall_interface_boundaries)

    fluid_rhs = ns_operator(
        dcoll, gas_model, fluid_state, fluid_all_boundaries,
        time=time, quadrature_tag=quadrature_tag, dd=fluid_dd,
        viscous_numerical_flux_func=viscous_facial_flux_harmonic,
        operator_states_quad=fluid_operator_states_quad,
        grad_t=fluid_grad_temperature)

    if use_av:
        if av_kwargs is None:
            av_kwargs = {}
        fluid_rhs = fluid_rhs + av_laplacian_operator(
            dcoll, fluid_all_boundaries, fluid_state, quadrature_tag=quadrature_tag,
            dd=fluid_dd, **av_kwargs)

    wall_rhs = wall_time_scale * _heat_operator(
        dcoll, wall_model, wall_all_boundaries, wall_temperature,
        penalty_amount=wall_penalty_amount, quadrature_tag=quadrature_tag,
        dd=wall_dd, _grad_temperature=wall_grad_temperature)

    return fluid_rhs, wall_rhs
