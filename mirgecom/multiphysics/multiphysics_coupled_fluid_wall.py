r""":mod:`mirgecom.multiphysics.multiphysics_coupled_fluid_wall` for fully-coupled
fluid and wall.

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
with a wall subdomain governed by the porous media equation (on-going work)
by enforcing continuity of quantities and their respective fluxes

.. math::
    q_\text{fluid} &= q_\text{wall} \\
    - D_\text{fluid} \nabla q_\text{fluid} \cdot \hat{n} &=
        - D_\text{wall} \nabla q_\text{wall} \cdot \hat{n}.

at the interface.

.. autofunction:: get_interface_boundaries
.. autofunction:: coupled_grad_operator
.. autofunction:: coupled_ns_operator

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

from grudge.trace_pair import inter_volume_trace_pairs
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    as_dofdesc,
)
import grudge.op as op
from grudge.dt_utils import characteristic_lengthscales

from mirgecom.fluid import make_conserved
from mirgecom.math import harmonic_mean
from mirgecom.transport import GasTransportVars
from mirgecom.boundary import MengaldoBoundaryCondition
from mirgecom.flux import num_flux_central
from mirgecom.inviscid import inviscid_facial_flux_rusanov
from mirgecom.viscous import viscous_facial_flux_harmonic
from mirgecom.gas_model import (
    make_fluid_state,
    make_operator_fluid_states,
    ViscousFluidState
)
from mirgecom.navierstokes import (
    grad_t_operator,
    grad_cv_operator,
    ns_operator
)


class _StateInterVolTag:
    pass


class _GradCVInterVolTag:
    pass


class _GradTemperatureInterVolTag:
    pass


class _FluidOpStatesTag:
    pass


class _FluidGradTag:
    pass


class _FluidOperatorTag:
    pass


class _WallOpStatesTag:
    pass


class _WallGradTag:
    pass


class _WallOperatorTag:
    pass


def _project_from_base(dcoll, dd_bdry, field):
    """Project *field* from *DISCR_TAG_BASE* to the same discr. as *dd_bdry*."""
    if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
        dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
        return op.project(dcoll, dd_bdry_base, dd_bdry, field)
    else:
        return field


class _MultiphysicsCoupledHarmonicMeanBoundaryComponent:
    """."""

    def __init__(self, state_plus, no_slip, grad_cv_plus=None, grad_t_plus=None,
            use_kappa_weighted_bc=False):
        """."""
        self._state_plus = state_plus
        self._grad_cv_plus = grad_cv_plus
        self._grad_t_plus = grad_t_plus
        self._no_slip = no_slip
        self._use_kappa_weighted_bc = use_kappa_weighted_bc

    def state_plus(self, dcoll, dd_bdry, state_minus, **kwargs):
        """State to enforce inviscid BC at the interface."""
        projected_state = _project_from_base(dcoll, dd_bdry, self._state_plus)

        if self._no_slip:
            # use the same idea of no-slip walls
            cv_plus = make_conserved(dim=dcoll.dim,
                                     mass=state_minus.cv.mass,
                                     energy=state_minus.cv.energy,
                                     momentum=-state_minus.cv.momentum,
                                     species_mass=state_minus.cv.species_mass)

            # FIXME compute temperature from the projected cv?
            return ViscousFluidState(cv=cv_plus, dv=state_minus.dv,
                                     tv=state_minus.tv)

        # FIXME compute temperature from the projected cv?
        return ViscousFluidState(cv=projected_state.cv, dv=projected_state.dv,
                                 tv=projected_state.tv)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus):
        """State to enforce viscous BC at the interface."""
        # do not use the state_plus function because that is for inviscid fluxes
        state_plus = _project_from_base(dcoll, dd_bdry, self._state_plus)

        mass_bc = 0.5*(state_minus.mass_density + state_plus.mass_density)

        u_bc = self.velocity_bc(dcoll, dd_bdry, state_minus)

        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus)

        y_bc = self.species_mass_fractions_bc(dcoll, dd_bdry, state_minus)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc, species_mass_fractions=y_bc)

        total_energy_bc = mass_bc*(internal_energy_bc + 0.5*np.dot(u_bc, u_bc))

        smoothness_mu = 0.5*(
            state_minus.dv.smoothness_mu + state_plus.dv.smoothness_mu)
        smoothness_kappa = 0.5*(
            state_minus.dv.smoothness_kappa + state_plus.dv.smoothness_kappa)
        smoothness_beta = 0.5*(
            state_minus.dv.smoothness_beta + state_plus.dv.smoothness_beta)

        cv_bc = make_conserved(dim=dcoll.dim, mass=mass_bc, momentum=mass_bc*u_bc,
            energy=total_energy_bc, species_mass=mass_bc*y_bc)

        state_bc = make_fluid_state(cv=cv_bc, gas_model=gas_model,
            temperature_seed=t_bc, smoothness_mu=smoothness_mu,
            smoothness_kappa=smoothness_kappa, smoothness_beta=smoothness_beta)

        if self._no_slip:
            new_mu = state_minus.tv.viscosity
        else:
            new_mu = harmonic_mean(state_minus.tv.viscosity,
                                   state_plus.tv.viscosity)

        new_kappa = harmonic_mean(state_minus.tv.thermal_conductivity,
                                  state_plus.tv.thermal_conductivity)

        new_diff = harmonic_mean(state_minus.tv.species_diffusivity,
                                 state_plus.tv.species_diffusivity)

        new_tv = GasTransportVars(
            bulk_viscosity=state_bc.tv.bulk_viscosity,
            viscosity=new_mu,
            thermal_conductivity=new_kappa,
            species_diffusivity=new_diff
        )

        return ViscousFluidState(cv=state_bc.cv, dv=state_bc.dv, tv=new_tv)

    def velocity_bc(self, dcoll, dd_bdry, state_minus):
        """Velocity at the interface.

        If no-slip, force the velocity to be zero. Else, uses an averaging.
        """
        kappa_minus = state_minus.tv.viscosity
        u_minus = state_minus.cv.velocity

        # if the coupling involves a no-slip wall:
        if self._no_slip:
            return u_minus*0.0

        # do not use the state_plus function because that is for inviscid fluxes
        state_plus = _project_from_base(dcoll, dd_bdry, self._state_plus)
        u_plus = state_plus.velocity
        if self._use_kappa_weighted_bc:
            kappa_plus = _project_from_base(dcoll, dd_bdry,
                                            state_plus.tv.viscosity)
            kappa_sum = kappa_minus + kappa_plus
            return (u_minus * kappa_minus + u_plus * kappa_plus)/kappa_sum
        else:
            return (u_minus + u_plus)/2

    def species_mass_fractions_bc(self, dcoll, dd_bdry, state_minus):
        """Speciess mass fractions at the interface."""
        kappa_minus = state_minus.tv.species_diffusivity
        y_minus = state_minus.species_mass_fractions

        # do not use the state_plus function because that is for inviscid fluxes
        state_plus = _project_from_base(dcoll, dd_bdry, self._state_plus)
        y_plus = state_plus.species_mass_fractions
        if self._use_kappa_weighted_bc:
            kappa_plus = _project_from_base(dcoll, dd_bdry,
                                            state_plus.tv.species_diffusivity)
            kappa_sum = kappa_minus + kappa_plus
            return (y_minus * kappa_minus + y_plus * kappa_plus)/kappa_sum
        else:
            return (y_minus + y_plus)/2

    def temperature_bc(self, dcoll, dd_bdry, state_minus):
        """Temperature at the interface."""
        kappa_minus = state_minus.tv.thermal_conductivity
        t_minus = state_minus.temperature

        # do not use the state_plus function because that is for inviscid fluxes
        state_plus = _project_from_base(dcoll, dd_bdry, self._state_plus)
        t_plus = state_plus.temperature
        if self._use_kappa_weighted_bc:
            kappa_plus = _project_from_base(dcoll, dd_bdry,
                                            state_plus.tv.thermal_conductivity)
            kappa_sum = kappa_minus + kappa_plus
            return (t_minus * kappa_minus + t_plus * kappa_plus)/kappa_sum
        else:
            return (t_minus + t_plus)/2

    def grad_cv_bc(self, dcoll, dd_bdry, grad_cv_minus):
        """Gradient averaging for viscous flux."""
        if self._grad_cv_plus is None:
            raise ValueError(
                "Boundary does not have external CV gradient data.")
        grad_cv_plus = _project_from_base(dcoll, dd_bdry, self._grad_cv_plus)

        # if the coupling involves a no-slip wall:
        if self._no_slip:
            grad_cv_bc = (grad_cv_plus + grad_cv_minus)/2
            return make_conserved(dim=dcoll.dim, mass=grad_cv_bc.mass,
                                  momentum=grad_cv_minus.momentum,
                                  energy=grad_cv_bc.energy,
                                  species_mass=grad_cv_bc.species_mass)

        return (grad_cv_plus + grad_cv_minus)/2

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus):
        """Gradient averaging for viscous flux."""
        if self._grad_t_plus is None:
            raise ValueError(
                "Boundary does not have external temperature gradient data.")

        grad_t_plus = _project_from_base(dcoll, dd_bdry, self._grad_t_plus)
        return (grad_t_plus + grad_t_minus)/2


# Although Fluid and Wall look identical, keep separated due to radiation
# FIXME: Interior penalty should probably use an average of the lengthscales on
# both sides of the interface
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

    def __init__(self, state_plus, interface_noslip,
                 grad_cv_plus=None, grad_t_plus=None,
                 flux_penalty_amount=None, lengthscales_minus=None,
                 use_kappa_weighted_grad_flux=False):
        r"""Initialize InterfaceFluidBoundary.

        Arguments *grad_cv_plus*, *grad_t_plus*, *flux_penalty_amount*, and
        *lengthscales_minus* are only required if the boundary will be used to
        compute the viscous flux.

        Parameters
        ----------
        state_plus: :class:`~mirgecom.gas_model.FluidState`
            Fluid state from the wall side.

        interface_noslip: bool
            If `True`, interface boundaries on the fluid side will be treated
            as no-slip walls. If `False` they will be treated as slip walls.

        grad_cv_plus: :class:`meshmode.dof_array.DOFArray` or None
            CV gradient from the wall side.

        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None
            Temperature gradient from the wall side.

        flux_penalty_amount: float or None
            Coefficient $c$ for the interior penalty on the heat flux.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None
            Characteristic mesh spacing $h^-$.

        use_kappa_weighted_grad_flux: bool
            Indicates whether the gradient fluxes at the interface should
            be computed using a simple or weighted average of quantities
            from each side by its respective transport coefficients.
        """
        self._state_plus = state_plus
        self._flux_penalty_amount = flux_penalty_amount
        self._lengthscales_minus = lengthscales_minus

        self._coupled = _MultiphysicsCoupledHarmonicMeanBoundaryComponent(
            state_plus=state_plus,
            no_slip=interface_noslip,
            grad_cv_plus=grad_cv_plus,
            grad_t_plus=grad_t_plus,
            use_kappa_weighted_bc=use_kappa_weighted_grad_flux)

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce inviscid BC at the interface."""
        # Don't bother replacing anything since this is just for inviscid
        return self._coupled.state_plus(dcoll, dd_bdry, state_minus, **kwargs)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce viscous BC at the interface."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._coupled.state_bc(dcoll, dd_bdry, gas_model, state_minus)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Interface temperature to enforce viscous BC."""
        return self._coupled.temperature_bc(dcoll, dd_bdry, state_minus)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            normal, **kwargs):
        """Gradient of CV to enforce viscous BC."""
        return self._coupled.grad_cv_bc(dcoll, dd_bdry, grad_cv_minus)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Gradient of temperature to enforce viscous BC."""
        return self._coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)

    def viscous_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func=viscous_facial_flux_harmonic,
                                **kwargs):
        """Return the viscous flux at the interface boundaries.

        It is defined by
        :meth:`mirgecom.boundary.MengaldoBoundaryCondition.viscous_divergence_flux`
        with the additional flux penalty term.
        the additional heat flux interior penalty term.
        """
        dd_bdry = as_dofdesc(dd_bdry)

        state_plus = self.state_plus(dcoll, dd_bdry, gas_model, state_minus,
                                     **kwargs)

        state_bc = self.state_bc(dcoll=dcoll, dd_bdry=dd_bdry,
            gas_model=gas_model, state_minus=state_minus, **kwargs)

        flux_without_penalty = super().viscous_divergence_flux(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, numerical_flux_func=numerical_flux_func,
            grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus, **kwargs)

        lengthscales_minus = _project_from_base(
            dcoll, dd_bdry, self._lengthscales_minus)

        tau_mu = self._flux_penalty_amount * (
            state_bc.tv.viscosity / lengthscales_minus)
        tau_kappa = self._flux_penalty_amount * (
            state_bc.tv.thermal_conductivity / lengthscales_minus)
        tau_diff = self._flux_penalty_amount * (
            state_bc.tv.species_diffusivity / lengthscales_minus)

        penalization = make_conserved(dim=dcoll.dim, mass=0.0,
            momentum=tau_mu*(state_plus.cv.momentum - state_minus.cv.momentum),
            energy=tau_kappa*(state_plus.temperature - state_minus.temperature),
            species_mass=tau_diff*(state_plus.cv.species_mass
                                   - state_minus.cv.species_mass)
        )

        return flux_without_penalty + penalization


# Although Fluid and Wall look identical, keep separated due to radiation
# FIXME: Interior penalty should probably use an average of the lengthscales on
# both sides of the interface
class InterfaceWallBoundary(InterfaceFluidBoundary):
    """Boundary for the wall side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: viscous_divergence_flux
    """

    def __init__(self, state_plus, interface_noslip,
                 grad_cv_plus=None, grad_t_plus=None,
                 flux_penalty_amount=None, lengthscales_minus=None,
                 use_kappa_weighted_grad_flux=False):
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

        use_kappa_weighted_grad_flux: bool
            Indicates whether the gradient fluxes at the interface should
            be computed using a simple or weighted average of quantities
            from each side by its respective transport coefficients.
        """
        self._state_plus = state_plus
        self._flux_penalty_amount = flux_penalty_amount
        self._lengthscales_minus = lengthscales_minus

        self._coupled = _MultiphysicsCoupledHarmonicMeanBoundaryComponent(
            state_plus=state_plus,
            no_slip=interface_noslip,
            grad_cv_plus=grad_cv_plus,
            grad_t_plus=grad_t_plus,
            use_kappa_weighted_bc=use_kappa_weighted_grad_flux)

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce inviscid BC at the interface."""
        # Don't bother replacing anything since this is just for inviscid
        return self._coupled.state_plus(dcoll, dd_bdry, state_minus, **kwargs)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """State to enforce viscous BC at the interface."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._coupled.state_bc(dcoll, dd_bdry, gas_model, state_minus)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Interface temperature to enforce viscous BC."""
        return self._coupled.temperature_bc(dcoll, dd_bdry, state_minus)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Gradient of CV to enforce viscous BC."""
        return self._coupled.grad_cv_bc(dcoll, dd_bdry, grad_cv_minus)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Gradient of temperature to enforce viscous BC."""
        return self._coupled.grad_temperature_bc(dcoll, dd_bdry, grad_t_minus)

    def viscous_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func=viscous_facial_flux_harmonic,
                                **kwargs):
        """Return the viscous flux at the interface boundaries.

        It is defined by
        :meth:`mirgecom.boundary.MengaldoBoundaryCondition.viscous_divergence_flux`
        with the additional flux penalty term.
        """
        dd_bdry = as_dofdesc(dd_bdry)

        state_plus = self.state_plus(dcoll, dd_bdry, gas_model, state_minus,
                                     **kwargs)

        state_bc = self.state_bc(dcoll=dcoll, dd_bdry=dd_bdry,
            gas_model=gas_model, state_minus=state_minus, **kwargs)

        flux_without_penalty = super().viscous_divergence_flux(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, numerical_flux_func=numerical_flux_func,
            grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus, **kwargs)

        lengthscales_minus = _project_from_base(
            dcoll, dd_bdry, self._lengthscales_minus)

        tau_mu = self._flux_penalty_amount * (
            state_bc.tv.viscosity / lengthscales_minus)
        tau_kappa = self._flux_penalty_amount * (
            state_bc.tv.thermal_conductivity / lengthscales_minus)
        tau_diff = self._flux_penalty_amount * (
            state_bc.tv.species_diffusivity / lengthscales_minus)

        penalization = make_conserved(dim=dcoll.dim, mass=0.0,
            momentum=tau_mu*(state_plus.cv.momentum - state_minus.cv.momentum),
            energy=tau_kappa*(state_plus.temperature - state_minus.temperature),
            species_mass=tau_diff*(state_plus.cv.species_mass
                                   - state_minus.cv.species_mass)
        )

        return flux_without_penalty + penalization


# FIXME try to modify this so we dont have to create a PorousFluidState for
# the pure-fluid part during the coupling
def _state_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, fluid_state, wall_state):
    """Exchange state across the fluid-wall interface."""
    pairwise_state = {(fluid_dd, wall_dd): (fluid_state, wall_state)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_state, comm_tag=_StateInterVolTag)


def _grad_cv_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, fluid_grad_cv, wall_grad_cv):
    """Exchange CV gradients across the fluid-wall interface."""
    pairwise_grad_cv = {(fluid_dd, wall_dd): (fluid_grad_cv, wall_grad_cv)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_grad_cv, comm_tag=_GradCVInterVolTag)


def _grad_temperature_inter_volume_trace_pairs(
        dcoll,
        fluid_dd, wall_dd,
        fluid_grad_temperature, wall_grad_temperature):
    """Exchange temperature gradient across the fluid-wall interface."""
    pairwise_grad_temperature = {
        (fluid_dd, wall_dd):
            (fluid_grad_temperature, wall_grad_temperature)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_grad_temperature, comm_tag=_GradTemperatureInterVolTag)


def get_interface_boundaries(
        dcoll,
        fluid_dd, wall_dd,
        fluid_state, wall_state,
        _state_inter_vol_tpairs,
        interface_noslip,
        fluid_grad_cv=None, wall_grad_cv=None,
        fluid_grad_temperature=None, wall_grad_temperature=None,
        *,
        use_kappa_weighted_grad_flux_in_fluid=False,
        wall_penalty_amount=None,
        _grad_cv_inter_vol_tpairs=None,
        _grad_temperature_inter_vol_tpairs=None):
    """Get the fluid-wall interface boundaries.

    Return a tuple `(fluid_interface_boundaries, wall_interface_boundaries)` in
    which each of the two entries is a mapping from each interface boundary's
    :class:`grudge.dof_desc.BoundaryDomainTag` to a boundary condition object
    compatible with that subdomain's operators. The map contains one entry for
    the collection of faces whose opposite face reside on the current MPI rank
    and one-per-rank for each collection of faces whose opposite face resides on
    a different rank.

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

    wall_state: :class:`~mirgecom.gas_model.FluidState`
        Fluid state object with the conserved state and dependent
        quantities for the wall volume.

    _state_inter_vol_tpairs

    interface_noslip: bool
        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    fluid_grad_temperature: numpy.ndarray or None
        Temperature gradient for the fluid volume. Only needed if boundaries
        are used to compute viscous fluxes.

    wall_grad_temperature: numpy.ndarray or None
        Temperature gradient for the wall volume. Only needed if boundaries
        are used to compute diffusion fluxes.

    use_kappa_weighted_grad_flux_in_fluid: bool
        Indicates whether the gradient fluxes on the fluid side of the
        interface should be computed using a simple or weighted average of
        quantities from each side by its respective transport coefficient.

    wall_penalty_amount: float
        Coefficient $c$ for the interior penalty on the heat flux. See
        :class:`InterfaceFluidBoundary`
        for details.

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use
        during operator evaluations.

    Returns
    -------
        The tuple `(fluid_interface_boundaries, wall_interface_boundaries)`.
    """
    assert (
        (fluid_grad_cv is None) == (wall_grad_cv is None)), (
        "Expected both fluid_grad_cv and wall_grad_cv or neither")

    assert (
        (fluid_grad_temperature is None) == (wall_grad_temperature is None)), (
        "Expected both fluid_grad_t and wall_grad_t or neither")

    include_gradient = fluid_grad_temperature is not None
    if include_gradient:
        if wall_penalty_amount is None:
            from warnings import warn
            warn("Error in wall_penalty_amount = None")

    # Exchange state, and (optionally) gradients

    state_inter_vol_tpairs = _state_inter_vol_tpairs
    if state_inter_vol_tpairs is None:
        from warnings import warn
        warn("Error in state_inter_vol_tpairs = None")

    if include_gradient:

        if _grad_cv_inter_vol_tpairs is None:
            grad_cv_inter_vol_tpairs = \
                _grad_cv_inter_volume_trace_pairs(
                    dcoll,
                    fluid_dd, wall_dd,
                    fluid_grad_cv, wall_grad_cv)
        else:
            grad_cv_inter_vol_tpairs = _grad_cv_inter_vol_tpairs

        if _grad_temperature_inter_vol_tpairs is None:
            grad_temperature_inter_vol_tpairs = \
                _grad_temperature_inter_volume_trace_pairs(
                    dcoll,
                    fluid_dd, wall_dd,
                    fluid_grad_temperature, wall_grad_temperature)
        else:
            grad_temperature_inter_vol_tpairs = _grad_temperature_inter_vol_tpairs

    else:
        grad_cv_inter_vol_tpairs = None
        grad_temperature_inter_vol_tpairs = None

    # Set up the interface boundaries

    if include_gradient:

        actx = fluid_state.cv.mass.array_context

        # Need to pass lengthscales into the BC constructor

        fluid_lengthscales = (
            characteristic_lengthscales(
                actx, dcoll, fluid_dd) * (0*fluid_state.temperature+1))

        wall_lengthscales = (
            characteristic_lengthscales(
                actx, dcoll, wall_dd) * (0*wall_state.temperature+1))

        # Construct interface boundaries with temperature gradient

        fluid_interface_boundaries = {
            state_tpair.dd.domain_tag: InterfaceFluidBoundary(
                state_tpair.ext,
                interface_noslip,
                grad_cv_tpair.ext,
                grad_temperature_tpair.ext,
                wall_penalty_amount,
                lengthscales_minus=op.project(dcoll,
                    fluid_dd, state_tpair.dd, fluid_lengthscales),
                use_kappa_weighted_grad_flux=use_kappa_weighted_grad_flux_in_fluid)
            for state_tpair, grad_cv_tpair, grad_temperature_tpair in zip(
                state_inter_vol_tpairs[wall_dd, fluid_dd],
                grad_cv_inter_vol_tpairs[wall_dd, fluid_dd],
                grad_temperature_inter_vol_tpairs[wall_dd, fluid_dd])}

        wall_interface_boundaries = {
            state_tpair.dd.domain_tag: InterfaceWallBoundary(
                state_tpair.ext,
                interface_noslip,
                grad_cv_tpair.ext,
                grad_temperature_tpair.ext,
                wall_penalty_amount,
                lengthscales_minus=op.project(dcoll,
                    wall_dd, state_tpair.dd, wall_lengthscales),
                use_kappa_weighted_grad_flux=use_kappa_weighted_grad_flux_in_fluid)
            for state_tpair, grad_cv_tpair, grad_temperature_tpair in zip(
                state_inter_vol_tpairs[fluid_dd, wall_dd],
                grad_cv_inter_vol_tpairs[fluid_dd, wall_dd],
                grad_temperature_inter_vol_tpairs[fluid_dd, wall_dd])}
    else:

        # Construct interface boundaries without gradient

        fluid_interface_boundaries = {
            state_tpair.dd.domain_tag: InterfaceFluidBoundary(
                state_tpair.ext,
                interface_noslip,
                use_kappa_weighted_grad_flux=use_kappa_weighted_grad_flux_in_fluid)
            for state_tpair in state_inter_vol_tpairs[wall_dd, fluid_dd]}

        wall_interface_boundaries = {
            state_tpair.dd.domain_tag: InterfaceWallBoundary(
                state_tpair.ext,
                interface_noslip,
                use_kappa_weighted_grad_flux=use_kappa_weighted_grad_flux_in_fluid)
            for state_tpair in state_inter_vol_tpairs[fluid_dd, wall_dd]}

    return fluid_interface_boundaries, wall_interface_boundaries


def coupled_grad_operator(
        dcoll,
        gas_model_fluid, gas_model_wall,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_state,
        quadrature_tag,
        _state_inter_vol_tpairs,
        _fluid_operator_states_quad,
        _wall_operator_states_quad,
        _fluid_interface_boundaries_no_grad,
        _wall_interface_boundaries_no_grad,
        interface_noslip,
        *,
        time=0.,
        use_kappa_weighted_grad_flux_in_fluid=False,
        fluid_numerical_flux_func=num_flux_central):
    r"""
    Compute $\nabla CV$ and $\nabla T$ on both fluid and wall subdomains.

    Parameters
    ----------
    dcoll: class:`~grudge.discretization.DiscretizationCollection`
        A discretization collection encapsulating the DG elements

    gas_model_fluid: :class:`~mirgecom.gas_model.GasModel`
        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    gas_model_wall: :class:`~mirgecom.gas_model.GasModel`
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

    wall_state: :class:`~mirgecom.gas_model.FluidState`
        Fluid state object with the conserved state and dependent
        quantities for the wall volume.

    quadrature_tag:
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    _state_inter_vol_tpairs

    _fluid_operator_states_quad

    _wall_operator_states_quad

    _fluid_interface_boundaries_no_grad

    _wall_interface_boundaries_no_grad

    interface_noslip: bool
        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    time:
        Time

    use_kappa_weighted_grad_flux_in_fluid: bool
        Indicates whether the gradient fluxes on the fluid side of the
        interface should be computed using either a simple or weighted average
        by its respective transport coefficients.

    fluid_numerical_flux_func:
        Callable function to return the numerical flux to be used when computing
        the temperature gradient in the fluid subdomain. Defaults to
        :class:`~mirgecom.flux.num_flux_central`.

    Returns
    -------
        The tuple `(fluid_grad_temperature, wall_grad_temperature)`.
    """
    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    # Construct boundaries for the fluid-wall interface; no temperature gradient
    # yet because that's what we're trying to compute

    assert (
        (_fluid_interface_boundaries_no_grad is None)
        == (_wall_interface_boundaries_no_grad is None)), (
        "Expected both _fluid_interface_boundaries_no_grad and "
        "_wall_interface_boundaries_no_grad or neither")

    if _fluid_interface_boundaries_no_grad is None:
        # Note: We don't need to supply wall_penalty_amount here since we're only
        # using these to compute the temperature gradient
        fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
            get_interface_boundaries(
                dcoll,
                fluid_dd, wall_dd,
                fluid_state, wall_state,
                interface_noslip=interface_noslip,
                use_kappa_weighted_grad_flux_in_fluid=(
                    use_kappa_weighted_grad_flux_in_fluid),
                _state_inter_vol_tpairs=_state_inter_vol_tpairs)
    else:
        fluid_interface_boundaries_no_grad = _fluid_interface_boundaries_no_grad
        wall_interface_boundaries_no_grad = _wall_interface_boundaries_no_grad

    # Augment the domain boundaries with the interface boundaries

    fluid_all_boundaries_no_grad = {}
    fluid_all_boundaries_no_grad.update(fluid_boundaries)
    fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    wall_all_boundaries_no_grad = {}
    wall_all_boundaries_no_grad.update(wall_boundaries)
    wall_all_boundaries_no_grad.update(wall_interface_boundaries_no_grad)

    # Compute the subdomain gradient operators using the augmented boundaries

    return (
        # fluid grad CV
        grad_cv_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=fluid_dd,
            operator_states_quad=_fluid_operator_states_quad,
            comm_tag=_FluidGradTag),

        # fluid grad T
        grad_t_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=fluid_dd,
            operator_states_quad=_fluid_operator_states_quad,
            comm_tag=_FluidGradTag),

        # wall grad CV
        grad_cv_operator(
            dcoll, gas_model_wall, wall_all_boundaries_no_grad, wall_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=wall_dd,
            operator_states_quad=_wall_operator_states_quad,
            comm_tag=_WallGradTag),

        # wall grad T
        grad_t_operator(
            dcoll, gas_model_wall, wall_all_boundaries_no_grad, wall_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=wall_dd,
            operator_states_quad=_wall_operator_states_quad,
            comm_tag=_WallGradTag)
        )


def coupled_ns_operator(
        dcoll,
        gas_model_fluid, gas_model_wall,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_state,
        interface_noslip,
        *,
        time=0.,
        use_kappa_weighted_grad_flux_in_fluid=False,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        fluid_limiter_func=None, wall_limiter_func=None,
        fluid_gradient_numerical_flux_func=num_flux_central,
        inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
        viscous_numerical_flux_func=viscous_facial_flux_harmonic,
        inviscid_fluid_terms_on=True, inviscid_wall_terms_on=True,
        return_gradients=False):
    r"""Compute the RHS of the fluid and wall subdomains.

    Parameters
    ----------
    dcoll: class:`~grudge.discretization.DiscretizationCollection`
        A discretization collection encapsulating the DG elements

    gas_model_fluid: :class:`~mirgecom.gas_model.GasModel`
        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    gas_model_wall: :class:`~mirgecom.gas_model.GasModel`
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

    interface_noslip: bool
        If `True`, interface boundaries on the fluid side will be treated as
        no-slip walls. If `False` they will be treated as slip walls.

    time:
        Time

    use_kappa_weighted_grad_flux_in_fluid: bool
        Indicates whether the gradient fluxes on the fluid side of the
        interface should be computed using either a simple or weighted average
        by its respective transport coefficients.

    wall_penalty_amount: float
        Coefficient $c$ for the interior penalty on the heat flux. See
        :class:`InterfaceFluidBoundary`
        for details.

    quadrature_tag:
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    fluid_gradient_numerical_flux_func:
        Callable function to return the numerical flux to be used when computing
        the temperature gradient in the fluid subdomain. Defaults to
        :class:`~mirgecom.flux.num_flux_central`.

    inviscid_numerical_flux_func:
        Callable function providing the face-normal flux to be used
        for the divergence of the inviscid transport flux.  This defaults to
        :func:`~mirgecom.inviscid.inviscid_facial_flux_rusanov`.

    viscous_numerical_flux_func:
        Callable function providing the face-normal flux to be used
        for the divergence of the viscous transport flux.  This defaults to
        :func:`~mirgecom.viscous.viscous_facial_flux_harmonic`.

    limiter_func:
        Callable function to be passed to
        :func:`~mirgecom.gas_model.make_operator_fluid_states`
        that filters or limits the produced fluid states.  This is used to keep
        species mass fractions in physical and realizable states, for example.

    inviscid_fluid_terms_on: bool
        Optional boolean to en/disable inviscid terms in the fluid operator.
        Defaults to ON (True).

    inviscid_wall_terms_on: bool
        Optional boolean to en/disable inviscid terms in the wall operator.
        Defaults to ON (True).

    Returns
    -------
        The tuple `(fluid_rhs, wall_rhs)` and, if desired, the gradients
        fluid_grad_cv, fluid_grad_t, wall_grad_cv, wall_grad_t.
    """
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

    state_inter_volume_trace_pairs = _state_inter_volume_trace_pairs(
        dcoll, fluid_dd, wall_dd, fluid_state, wall_state)

    # Construct boundaries for the fluid-wall interface; no temperature gradient
    # yet because we need to compute it

    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        get_interface_boundaries(
            dcoll=dcoll,
            fluid_dd=fluid_dd, wall_dd=wall_dd,
            fluid_state=fluid_state, wall_state=wall_state,
            interface_noslip=interface_noslip,
            use_kappa_weighted_grad_flux_in_fluid=(
                use_kappa_weighted_grad_flux_in_fluid),
            _state_inter_vol_tpairs=state_inter_volume_trace_pairs)

    # Augment the domain boundaries with the interface boundaries (fluid only;
    # needed for make_operator_fluid_states)

    fluid_all_boundaries_no_grad = {}
    fluid_all_boundaries_no_grad.update(fluid_boundaries)
    fluid_all_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    wall_all_boundaries_no_grad = {}
    wall_all_boundaries_no_grad.update(wall_boundaries)
    wall_all_boundaries_no_grad.update(wall_interface_boundaries_no_grad)

    # Get the operator fluid states
    fluid_operator_states_quad = make_operator_fluid_states(
        dcoll, fluid_state, gas_model_fluid, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=fluid_dd, comm_tag=_FluidOpStatesTag,
        limiter_func=fluid_limiter_func)

    wall_operator_states_quad = make_operator_fluid_states(
        dcoll, wall_state, gas_model_wall, wall_all_boundaries_no_grad,
        quadrature_tag, dd=wall_dd, comm_tag=_WallOpStatesTag,
        limiter_func=wall_limiter_func)

    # Compute the gradients of CV and T for both subdomains

    fluid_grad_cv, fluid_grad_t, wall_grad_cv, wall_grad_t = \
        coupled_grad_operator(
            dcoll,
            gas_model_fluid, gas_model_wall,
            fluid_dd, wall_dd,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_state,
            quadrature_tag=quadrature_tag,
            _state_inter_vol_tpairs=state_inter_volume_trace_pairs,
            _fluid_operator_states_quad=fluid_operator_states_quad,
            _wall_operator_states_quad=wall_operator_states_quad,
            _fluid_interface_boundaries_no_grad=fluid_interface_boundaries_no_grad,
            _wall_interface_boundaries_no_grad=wall_interface_boundaries_no_grad,
            interface_noslip=interface_noslip,
            time=time,
            use_kappa_weighted_grad_flux_in_fluid=(
                use_kappa_weighted_grad_flux_in_fluid),
            fluid_numerical_flux_func=fluid_gradient_numerical_flux_func)

    # Construct boundaries for the fluid-wall interface, now with the temperature
    # gradient

    fluid_interface_boundaries, wall_interface_boundaries = \
        get_interface_boundaries(
            dcoll=dcoll,
            fluid_dd=fluid_dd, wall_dd=wall_dd,
            fluid_state=fluid_state, wall_state=wall_state,
            fluid_grad_cv=fluid_grad_cv,
            wall_grad_cv=wall_grad_cv,
            fluid_grad_temperature=fluid_grad_t,
            wall_grad_temperature=wall_grad_t,
            interface_noslip=interface_noslip,
            use_kappa_weighted_grad_flux_in_fluid=(
                use_kappa_weighted_grad_flux_in_fluid),
            wall_penalty_amount=wall_penalty_amount,
            _state_inter_vol_tpairs=state_inter_volume_trace_pairs)

    # Augment the domain boundaries with the interface boundaries

    fluid_all_boundaries = {}
    fluid_all_boundaries.update(fluid_boundaries)
    fluid_all_boundaries.update(fluid_interface_boundaries)

    wall_all_boundaries = {}
    wall_all_boundaries.update(wall_boundaries)
    wall_all_boundaries.update(wall_interface_boundaries)

    # Compute the respective subdomain NS operators using the augmented boundaries

    fluid_result = ns_operator(
        dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries, time=time,
        quadrature_tag=quadrature_tag, dd=fluid_dd,
        viscous_numerical_flux_func=viscous_numerical_flux_func,
        # TODO do we really need the return gradients or
        # can we use the already-evaluated gradient?
        return_gradients=return_gradients,
        operator_states_quad=fluid_operator_states_quad,
        grad_t=fluid_grad_t, comm_tag=_FluidOperatorTag,
        inviscid_terms_on=inviscid_fluid_terms_on)

    wall_result = ns_operator(
        dcoll, gas_model_wall, wall_state, wall_all_boundaries, time=time,
        quadrature_tag=quadrature_tag, dd=wall_dd,
        viscous_numerical_flux_func=viscous_numerical_flux_func,
        # TODO do we really need the return gradients or
        # can we use the already-evaluated gradient?
        return_gradients=return_gradients,
        operator_states_quad=wall_operator_states_quad,
        grad_t=wall_grad_t, comm_tag=_WallOperatorTag,
        inviscid_terms_on=inviscid_wall_terms_on)

    if return_gradients:
        fluid_rhs, fluid_grad_cv, fluid_grad_t = fluid_result
        wall_rhs, wall_grad_cv, wall_grad_t = wall_result
        return (
            fluid_rhs, wall_rhs, fluid_grad_cv, fluid_grad_t,
            wall_grad_cv, wall_grad_t)

    return fluid_result, wall_result
