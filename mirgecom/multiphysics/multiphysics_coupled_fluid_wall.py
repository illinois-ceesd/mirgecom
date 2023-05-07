r""":mod:`mirgecom.multiphysics.multiphysics_coupled_fluid_wall` for fully-coupled
fluid and wall.

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
with a wall subdomain governed by the diffusion equation by enforcing
continuity of quantities and their respective fluxes

.. math::
    q_\text{fluid} &= q_\text{wall} \\
    - D_\text{fluid} \nabla q_\text{fluid} \cdot \hat{n} &=
        - D_\text{wall} \nabla q_\text{wall} \cdot \hat{n}.

at the interface.

.. autofunction:: get_interface_boundaries
.. autofunction:: coupled_grad_t_operator
.. autofunction:: coupled_grad_cv_operator
.. autofunction:: coupled_ns_heat_operator

.. autoclass:: InterfaceFluidBoundary
.. autoclass:: InterfaceFluidNoslipBoundary
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

# from dataclasses import replace
import numpy as np
# from abc import abstractmethod

from grudge.trace_pair import (
    # TracePair,
    inter_volume_trace_pairs
)
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    as_dofdesc,
)
import grudge.op as op

from mirgecom.fluid import make_conserved
from mirgecom.math import harmonic_mean
from mirgecom.transport import GasTransportVars
from mirgecom.boundary import (
    MengaldoBoundaryCondition,
    # _SlipBoundaryComponent,
    # _NoSlipBoundaryComponent,
    # _ImpermeableBoundaryComponent
)
from mirgecom.flux import num_flux_central
from mirgecom.inviscid import inviscid_facial_flux_rusanov
from mirgecom.viscous import viscous_facial_flux_harmonic
from mirgecom.gas_model import (
    # replace_fluid_state,
    make_fluid_state,
    make_operator_fluid_states,
)
from mirgecom.navierstokes import (
    grad_t_operator as fluid_grad_t_operator,
    grad_cv_operator as fluid_grad_cv_operator,
    ns_operator,
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

    def __init__(
            self, state_plus, grad_cv_plus=None, grad_t_plus=None,
            use_kappa_weighted_bc=False):
        self._state_plus = state_plus
        self._grad_cv_plus = grad_cv_plus
        self._grad_t_plus = grad_t_plus
        self._use_kappa_weighted_bc = use_kappa_weighted_bc

    def grad_cv_bc(self, dcoll, dd_bdry, grad_cv_minus):
        if self._grad_cv_plus is None:
            raise ValueError(
                "Boundary does not have external CV gradient data.")
        grad_cv_plus = _project_from_base(dcoll, dd_bdry, self._grad_cv_plus)
        return (grad_cv_plus + grad_cv_minus)/2

    def velocity_bc(self, dcoll, dd_bdry, kappa_minus, u_minus):
        u_plus = _project_from_base(dcoll, dd_bdry, self._state_plus.velocity)
        if self._use_kappa_weighted_bc:
            actx = u_minus.array_context
            kappa_plus = _project_from_base(dcoll, dd_bdry,
                self._state_plus.tv.viscosity)
            kappa_sum = actx.np.where(
                actx.np.greater(kappa_minus + kappa_plus, 0*kappa_minus),
                kappa_minus + kappa_plus,
                0*kappa_minus + 1)
            return (u_minus * kappa_minus + u_plus * kappa_plus)/kappa_sum
        else:
            return (u_minus + u_plus)/2

    def species_mass_fractions_bc(self, dcoll, dd_bdry, kappa_minus, y_minus):
        y_plus = _project_from_base(dcoll, dd_bdry,
                                    self._state_plus.species_mass_fractions)
        if self._use_kappa_weighted_bc:
            actx = y_minus.array_context
            kappa_plus = _project_from_base(dcoll, dd_bdry,
                self._state_plus.tv.species_diffusivity)
            kappa_sum = actx.np.where(
                actx.np.greater(kappa_minus + kappa_plus, 0*kappa_minus),
                kappa_minus + kappa_plus,
                0*kappa_minus + 1)
            return (y_minus * kappa_minus + y_plus * kappa_plus)/kappa_sum
        else:
            return (y_minus + y_plus)/2

    def temperature_bc(self, dcoll, dd_bdry, kappa_minus, t_minus):
        t_plus = _project_from_base(dcoll, dd_bdry, self._state_plus.temperature)
        if self._use_kappa_weighted_bc:
            actx = t_minus.array_context
            kappa_plus = _project_from_base(dcoll, dd_bdry,
                self._state_plus.tv.thermal_conductivity)
            kappa_sum = actx.np.where(
                actx.np.greater(kappa_minus + kappa_plus, 0*kappa_minus),
                kappa_minus + kappa_plus,
                0*kappa_minus + 1)
            return (t_minus * kappa_minus + t_plus * kappa_plus)/kappa_sum
        else:
            return (t_minus + t_plus)/2

#    def temperature_plus(self, dcoll, dd_bdry):
#        return _project_from_base(dcoll, dd_bdry, self._state_plus.temperature)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, grad_t_minus):
        if self._grad_t_plus is None:
            raise ValueError(
                "Boundary does not have external temperature gradient data.")
        grad_t_plus = _project_from_base(dcoll, dd_bdry, self._grad_t_plus)
        return (grad_t_plus + grad_t_minus)/2


class InterfaceFluidBoundary(MengaldoBoundaryCondition):
    """
    Boundary for the fluid side on the interface between fluid and no-slip wall.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_plus
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(
            self, state_plus, grad_cv_plus=None, grad_t_plus=None,
            flux_penalty_amount=None, lengthscales_minus=None,
            use_kappa_weighted_grad_flux=False):
        r"""
        Initialize InterfaceFluidNoslipBoundary.

        Arguments *grad_cv_plus*, *grad_t_plus*, *flux_penalty_amount*, and
        *lengthscales_minus* are only required if the boundary will be used to
        compute the viscous flux.

        Parameters
        ----------
        state_plus: float or :class:`meshmode.dof_array.DOFArray`


        grad_cv_plus: :class:`meshmode.dof_array.DOFArray` or None


        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None
            Temperature gradient from the wall side.

        heat_flux_penalty_amount: float or None
            Coefficient $c$ for the interior penalty on the heat flux.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None
            Characteristic mesh spacing $h^-$.

        use_kappa_weighted_grad_flux: bool

        """
        self._flux_penalty_amount=flux_penalty_amount
        self._lengthscales_minus=lengthscales_minus

        self._coupled = _MultiphysicsCoupledHarmonicMeanBoundaryComponent(
            state_plus=state_plus,
            grad_cv_plus=grad_cv_plus,
            grad_t_plus=grad_t_plus,
            use_kappa_weighted_bc=use_kappa_weighted_grad_flux)

    def state_plus(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        # Don't bother replacing anything since this is just for inviscid
        return self._state_plus

    def state_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        dd_bdry = as_dofdesc(dd_bdry)

        mass_bc = 0.5*(state_minus.mass_density + self._state_plus.mass_density)

        u_bc = self._coupled.velocity_bc(dcoll, dd_bdry,
            state_minus.tv.viscosity, state_minus.cv.velocity)

        t_bc = self._coupled.temperature_bc(dcoll, dd_bdry,
            state_minus.tv.thermal_conductivity, state_minus.temperature)

        y_bc = self._coupled.species_mass_fractions_bc(dcoll, dd_bdry,
            state_minus.tv.species_diffusivity, state_minus.species_mass_fractions)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc, species_mass_fractions=y_bc)

        # general case
        total_energy_bc = mass_bc*(internal_energy_bc + 0.5*np.dot(u_bc, u_bc))

        smoothness_mu = 0.5*(
            state_minus.dv.smoothness_mu + self._state_plus.dv.smoothness_mu)
        smoothness_kappa = 0.5*(
            state_minus.dv.smoothness_kappa + self._state_plus.dv.smoothness_kappa)
        smoothness_beta = 0.5*(
            state_minus.dv.smoothness_beta + self._state_plus.dv.smoothness_beta)

        cv_bc = make_conserved(dim=dcoll.dim, mass=mass_bc, momentum=mass_bc*u_bc,
            energy=total_energy_bc, species_mass=mass_bc*y_bc)

        state_bc = make_fluid_state(cv=cv_bc, gas_model=gas_model,
            temperature_seed=t_bc, smoothness_mu=smoothness_mu,
            smoothness_kappa=smoothness_kappa, smoothness_beta=smoothness_beta)

        new_kappa = harmonic_mean(state_minus.tv.thermal_conductivity,
                                  self._state_plus.tv.thermal_conductivity)
        new_diff = harmonic_mean(state_minus.tv.species_diffusivity,
                                 self._state_plus.tv.species_diffusivity)
        new_mu = harmonic_mean(state_minus.tv.viscosity,
                               self._state_plus.tv.viscosity)

        return GasTransportVars(
            bulk_viscosity=state_bc.tv.bulk_viscosity,
            viscosity=new_mu,
            thermal_conductivity=new_kappa,
            species_diffusivity=new_diff
        )

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus, normal,
            **kwargs):  # noqa: D102
        return self._coupled.grad_cv_bc(dcoll, dd_bdry)

#    def temperature_plus(
#            self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
#        return self._coupled.temperature_plus(dcoll, dd_bdry)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)
        return self._coupled.temperature_bc(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):  # noqa: D102
        return self._coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)

    # FIXME
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

        lengthscales_minus = _project_from_base(
            dcoll, dd_bdry, self._lengthscales_minus)

        tau = (
            self._penalty_amount * state_bc.thermal_conductivity
            / lengthscales_minus)

#        t_minus = state_minus.temperature
#        t_plus = self.temperature_plus(
#            dcoll, dd_bdry=dd_bdry, state_minus=state_minus, **kwargs)

        # XXX double check this
        return make_conserved(dim=dcoll.dim,
            mass=flux_without_penalty.mass
            - tau*(self._state_plus.cv.mass - state_minus.cv.mass),
            momentum=flux_without_penalty.momentum
            - tau*(self._state_plus.cv.momentum - state_minus.cv.momentum),
            energy=flux_without_penalty.energy
            - tau*(self._state_plus.temperature - state_minus.temperature),
            species_mass=flux_without_penalty.momentum
            - tau*(self._state_plus.cv.species_mass - state_minus.cv.species_mass)
        )


# FIXME: Interior penalty should probably use an average of the lengthscales on
# both sides of the interface
class InterfaceWallBoundary(InterfaceFluidBoundary):
    """
    Boundary for the wall side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_plus
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    """
    def __init__(
            self, state_plus, grad_cv_plus=None, grad_t_plus=None,
            flux_penalty_amount=None, lengthscales_minus=None,
            use_kappa_weighted_grad_flux=False):
        r"""
        Initialize InterfaceWallBoundary.

        Arguments *grad_cv_plus*, *grad_t_plus*, *flux_penalty_amount*, and
        *lengthscales_minus* are only required if the boundary will be used to
        compute the viscous flux.

        Parameters
        ----------
        state_plus: float or :class:`meshmode.dof_array.DOFArray`


        grad_cv_plus: :class:`meshmode.dof_array.DOFArray` or None


        grad_t_plus: :class:`meshmode.dof_array.DOFArray` or None
            Temperature gradient from the wall side.

        heat_flux_penalty_amount: float or None
            Coefficient $c$ for the interior penalty on the heat flux.

        lengthscales_minus: :class:`meshmode.dof_array.DOFArray` or None
            Characteristic mesh spacing $h^-$.

        use_kappa_weighted_grad_flux: bool

        """
        self._flux_penalty_amount=flux_penalty_amount
        self._lengthscales_minus=lengthscales_minus

        self._coupled = _MultiphysicsCoupledHarmonicMeanBoundaryComponent(
            state_plus=state_plus,
            grad_cv_plus=grad_cv_plus,
            grad_t_plus=grad_t_plus,
            use_kappa_weighted_bc=use_kappa_weighted_grad_flux)

    def state_plus(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        # Don't bother replacing anything since this is just for inviscid
        return self._state_plus

    def state_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):  # noqa: D102
        dd_bdry = as_dofdesc(dd_bdry)

        mass_bc = 0.5*(state_minus.mass_density + self._state_plus.mass_density)

        u_bc = self._coupled.velocity_bc(dcoll, dd_bdry,
            state_minus.tv.viscosity, state_minus.cv.velocity)

        t_bc = self._coupled.temperature_bc(dcoll, dd_bdry,
            state_minus.tv.thermal_conductivity, state_minus.temperature)

        y_bc = self._coupled.species_mass_fractions_bc(dcoll, dd_bdry,
            state_minus.tv.species_diffusivity, state_minus.species_mass_fractions)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc, species_mass_fractions=y_bc)

        # general case
        total_energy_bc = mass_bc*(internal_energy_bc + 0.5*np.dot(u_bc, u_bc))

        smoothness_mu = 0.5*(
            state_minus.dv.smoothness_mu + self._state_plus.dv.smoothness_mu)
        smoothness_kappa = 0.5*(
            state_minus.dv.smoothness_kappa + self._state_plus.dv.smoothness_kappa)
        smoothness_beta = 0.5*(
            state_minus.dv.smoothness_beta + self._state_plus.dv.smoothness_beta)

        cv_bc = make_conserved(dim=dcoll.dim, mass=mass_bc, momentum=mass_bc*u_bc,
            energy=total_energy_bc, species_mass=mass_bc*y_bc)

        state_bc = make_fluid_state(cv=cv_bc, gas_model=gas_model,
            temperature_seed=t_bc, smoothness_mu=smoothness_mu,
            smoothness_kappa=smoothness_kappa, smoothness_beta=smoothness_beta)

        new_kappa = harmonic_mean(state_minus.tv.thermal_conductivity,
                                  self._state_plus.tv.thermal_conductivity)
        new_diff = harmonic_mean(state_minus.tv.species_diffusivity,
                                 self._state_plus.tv.species_diffusivity)
        new_mu = harmonic_mean(state_minus.tv.viscosity,
                               self._state_plus.tv.viscosity)

        return GasTransportVars(
            bulk_viscosity=state_bc.tv.bulk_viscosity,
            viscosity=new_mu,
            thermal_conductivity=new_kappa,
            species_diffusivity=new_diff
        )

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus, normal,
            **kwargs):  # noqa: D102
        return self._coupled.grad_cv_bc(dcoll, dd_bdry)

#    def temperature_plus(
#            self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
#        return self._coupled.temperature_plus(dcoll, dd_bdry)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):  # noqa: D102
        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)
        return self._coupled.temperature_bc(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):  # noqa: D102
        return self._coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)

    # FIXME
    def viscous_divergence_flux(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, numerical_flux_func=viscous_facial_flux_harmonic,
            **kwargs):
        """
        Return the viscous flux as defined by
        :meth:`mirgecom.boundary.MengaldoBoundaryCondition.viscous_divergence_flux`
        with the additional flux penalty term.
        """
        dd_bdry = as_dofdesc(dd_bdry)

        state_bc = self.state_bc(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

        flux_without_penalty = super().viscous_divergence_flux(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, numerical_flux_func=numerical_flux_func,
            grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus, **kwargs)

        lengthscales_minus = _project_from_base(
            dcoll, dd_bdry, self._lengthscales_minus)

        tau = (
            self._penalty_amount * state_bc.thermal_conductivity
            / lengthscales_minus)

#        t_minus = state_minus.temperature
#        t_plus = self.temperature_plus(
#            dcoll, dd_bdry=dd_bdry, state_minus=state_minus, **kwargs)

        # XXX double check this
        return make_conserved(dim=dcoll.dim,
            mass=flux_without_penalty.mass
            - tau*(self._state_plus.cv.mass - state_minus.cv.mass),
            momentum=flux_without_penalty.momentum
            - tau*(self._state_plus.cv.momentum - state_minus.cv.momentum),
            energy=flux_without_penalty.energy
            - tau*(self._state_plus.temperature - state_minus.temperature),
            species_mass=flux_without_penalty.momentum
            - tau*(self._state_plus.cv.species_mass - state_minus.cv.species_mass)
        )


def _state_inter_volume_trace_pairs(
        dcoll, gas_model, fluid_dd, wall_dd, fluid_state, wall_state):
    """Exchange state across the fluid-wall interface."""
    pairwise_state = {(fluid_dd, wall_dd): (fluid_state, wall_state)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_state, comm_tag=_StateInterVolTag)


def _grad_cv_inter_volume_trace_pairs(
        dcoll, gas_model, fluid_dd, wall_dd, fluid_grad_cv, wall_grad_cv):
    """Exchange gradients across the fluid-wall interface."""
    pairwise_grad_cv = {(fluid_dd, wall_dd): (fluid_grad_cv, wall_grad_cv)}
    return inter_volume_trace_pairs(
        dcoll, pairwise_grad_cv, comm_tag=_GradCVInterVolTag)


def _grad_temperature_inter_volume_trace_pairs(
        dcoll,
        gas_model,
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
        gas_model,
        fluid_dd, wall_dd,
        fluid_state, wall_state,
        fluid_grad_cv=None, wall_grad_cv=None,
        fluid_grad_temperature=None, wall_grad_temperature=None,
        *,
        interface_noslip=True,
        use_kappa_weighted_grad_flux_in_fluid=False,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _state_inter_vol_tpairs=None,
        _grad_cv_inter_vol_tpairs=None,
        _grad_temperature_inter_vol_tpairs=None):
    """
    Get the fluid-wall interface boundaries.

    Return a tuple `(fluid_interface_boundaries, wall_interface_boundaries)` in
    which each of the two entries is a mapping from each interface boundary's
    :class:`grudge.dof_desc.BoundaryDomainTag` to a boundary condition object
    compatible with that subdomain's operators. The map contains one entry for
    the collection of faces whose opposite face reside on the current MPI rank
    and one-per-rank for each collection of faces whose opposite face resides on
    a different rank.
    """
    assert (
        (fluid_grad_cv is None) == (wall_grad_cv is None)), (
        "Expected both fluid_grad_cv and wall_grad_cv or neither")

    assert (
        (fluid_grad_temperature is None) == (wall_grad_temperature is None)), (
        "Expected both fluid_grad_t and wall_grad_t or neither")

    include_gradient = fluid_grad_temperature is not None

    # Exchange state, and (optionally) gradients

    if _state_inter_vol_tpairs is None:
        state_inter_vol_tpairs = _state_inter_volume_trace_pairs(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_state)
    else:
        state_inter_vol_tpairs = _state_inter_vol_tpairs

    if include_gradient:

        if _grad_cv_inter_vol_tpairs is None:
            grad_cv_inter_vol_tpairs = \
                _grad_cv_inter_volume_trace_pairs(
                    dcoll,
                    gas_model,
                    fluid_dd, wall_dd,
                    fluid_grad_temperature, wall_grad_temperature)
        else:
            grad_cv_inter_vol_tpairs = _grad_cv_inter_vol_tpairs

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
        grad_cv_inter_vol_tpairs = None
        grad_temperature_inter_vol_tpairs = None

    # Set up the interface boundaries

    if include_gradient:

        # Diffusion operator passes lengthscales_minus into the boundary flux
        # functions, but NS doesn't; thus we need to pass lengthscales into
        # the fluid boundary condition constructor
        from grudge.dt_utils import characteristic_lengthscales
        fluid_lengthscales = (
            characteristic_lengthscales(
                fluid_state.array_context, dcoll, fluid_dd)
            * (0*fluid_state.temperature+1))

        # Construct interface boundaries with temperature gradient

        fluid_interface_boundaries = {
            state_tpair.dd.domain_tag: InterfaceFluidBoundary(
                state_tpair.ext,
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
    else:

        # Construct interface boundaries without gradient

        fluid_interface_boundaries = {
            state_tpair.dd.domain_tag: InterfaceFluidBoundary(
                state_tpair.ext,
                use_kappa_weighted_grad_flux=use_kappa_weighted_grad_flux_in_fluid)
            for state_tpair in state_inter_vol_tpairs[wall_dd, fluid_dd]}

        wall_interface_boundaries = {
            state_tpair.dd.domain_tag: InterfaceWallBoundary(
                state_tpair.ext,
                use_kappa_weighted_grad_flux=use_kappa_weighted_grad_flux_in_fluid)
            for state_tpair in state_inter_vol_tpairs[wall_dd, fluid_dd]}

    return fluid_interface_boundaries, wall_interface_boundaries


def coupled_grad_cv_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_state,
        *,
        time=0.,
        interface_noslip=True,
        use_kappa_weighted_grad_flux_in_fluid=False,
        quadrature_tag=DISCR_TAG_BASE,
        fluid_numerical_flux_func=num_flux_central,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _state_inter_vol_tpairs=None,
        _fluid_operator_states_quad=None,
        _wall_operator_states_quad=None,
        _fluid_interface_boundaries_no_grad=None,
        _wall_interface_boundaries_no_grad=None):
    r"""
    Compute $\nabla CV$ on the fluid and wall subdomains.

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
                gas_model,
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
        fluid_grad_cv_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=fluid_dd,
            operator_states_quad=_fluid_operator_states_quad,
            comm_tag=_FluidGradTag),
        fluid_grad_cv_operator(
            dcoll, gas_model, wall_all_boundaries_no_grad, wall_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=wall_dd,
            operator_states_quad=_wall_operator_states_quad,
            comm_tag=_WallGradTag))


def coupled_grad_t_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_state,
        *,
        time=0.,
        interface_noslip=True,
        use_kappa_weighted_grad_flux_in_fluid=False,
        quadrature_tag=DISCR_TAG_BASE,
        fluid_numerical_flux_func=num_flux_central,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _state_inter_vol_tpairs=None,
        _fluid_operator_states_quad=None,
        _wall_operator_states_quad=None,
        _fluid_interface_boundaries_no_grad=None,
        _wall_interface_boundaries_no_grad=None):
    r"""
    Compute $\nabla T$ on the fluid and wall subdomains.

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
                gas_model,
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
        fluid_grad_t_operator(
            dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=fluid_dd,
            operator_states_quad=_fluid_operator_states_quad,
            comm_tag=_FluidGradTag),
        fluid_grad_t_operator(
            dcoll, gas_model, wall_all_boundaries_no_grad, wall_state,
            time=time, quadrature_tag=quadrature_tag,
            numerical_flux_func=fluid_numerical_flux_func, dd=wall_dd,
            operator_states_quad=_wall_operator_states_quad,
            comm_tag=_WallGradTag))


def coupled_ns_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_state,
        *,
        time=0.,
        interface_noslip=True,
        use_kappa_weighted_grad_flux_in_fluid=False,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        limiter_func=None,
        fluid_gradient_numerical_flux_func=num_flux_central,
        inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
        viscous_numerical_flux_func=viscous_facial_flux_harmonic,
        return_gradients=False):
    r"""
    Compute the RHS of the fluid and wall subdomains.

    Returns
    -------
        The tuple `(fluid_rhs, wall_rhs)`.
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
        dcoll, gas_model, fluid_dd, wall_dd, fluid_state, wall_state)

    # Construct boundaries for the fluid-wall interface; no temperature gradient
    # yet because we need to compute it

    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        get_interface_boundaries(
            dcoll=dcoll,
            gas_model=gas_model,
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
        dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=fluid_dd, comm_tag=_FluidOpStatesTag,
        limiter_func=limiter_func)

    wall_operator_states_quad = make_operator_fluid_states(
        dcoll, wall_state, gas_model, wall_all_boundaries_no_grad,
        quadrature_tag, dd=wall_dd, comm_tag=_WallOpStatesTag,
        limiter_func=limiter_func)

    # Compute the CV gradient for both subdomains
    fluid_grad_cv, wall_grad_cv = coupled_grad_cv_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_state,
        time=time,
        interface_noslip=interface_noslip,
        use_kappa_weighted_grad_flux_in_fluid=use_kappa_weighted_grad_flux_in_fluid,
        quadrature_tag=quadrature_tag,
        fluid_numerical_flux_func=fluid_operator_states_quad,
        _state_inter_vol_tpairs=state_inter_volume_trace_pairs,
        _fluid_operator_states_quad=fluid_operator_states_quad,
        _wall_operator_states_quad=wall_operator_states_quad,
        _fluid_interface_boundaries_no_grad=fluid_interface_boundaries_no_grad,
        _wall_interface_boundaries_no_grad=wall_interface_boundaries_no_grad)

    # Compute the temperature gradient for both subdomains

    fluid_grad_temperature, wall_grad_temperature = coupled_grad_t_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_state,
        time=time,
        interface_noslip=interface_noslip,
        use_kappa_weighted_grad_flux_in_fluid=use_kappa_weighted_grad_flux_in_fluid,
        quadrature_tag=quadrature_tag,
        fluid_numerical_flux_func=fluid_gradient_numerical_flux_func,
        _state_inter_vol_tpairs=state_inter_volume_trace_pairs,
        _fluid_operator_states_quad=fluid_operator_states_quad,
        _wall_operator_states_quad=wall_operator_states_quad,
        _fluid_interface_boundaries_no_grad=fluid_interface_boundaries_no_grad,
        _wall_interface_boundaries_no_grad=wall_interface_boundaries_no_grad)

    # Construct boundaries for the fluid-wall interface, now with the temperature
    # gradient

    fluid_interface_boundaries, wall_interface_boundaries = \
        get_interface_boundaries(
            dcoll=dcoll,
            gas_model=gas_model,
            fluid_dd=fluid_dd, wall_dd=wall_dd,
            fluid_state=fluid_state, wall_state=wall_state,
            fluid_grad_temperature=fluid_grad_temperature,
            wall_grad_temperature=wall_grad_temperature,
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
        dcoll, gas_model, fluid_state, fluid_all_boundaries, time=time,
        quadrature_tag=quadrature_tag, dd=fluid_dd,
        viscous_numerical_flux_func=viscous_numerical_flux_func,
        return_gradients=return_gradients,
        operator_states_quad=fluid_operator_states_quad,
        grad_t=fluid_grad_temperature, comm_tag=_FluidOperatorTag)

    # diffusion only, but all equations are considered now
    # velocity should be zero
    wall_result = ns_operator(
        dcoll, gas_model, wall_state, wall_all_boundaries, time=time,
        quadrature_tag=quadrature_tag, dd=wall_dd,
        viscous_numerical_flux_func=viscous_numerical_flux_func,
        return_gradients=return_gradients,
        operator_states_quad=wall_operator_states_quad,
        grad_t=wall_grad_temperature, comm_tag=_WallOperatorTag,
        inviscid_terms_on=False)

    if return_gradients:
        fluid_rhs, fluid_grad_cv, fluid_grad_t = fluid_result
        wall_rhs, wall_grad_cv, wall_grad_t = wall_result
        return (
            fluid_rhs, wall_rhs, fluid_grad_cv, fluid_grad_t,
            wall_grad_cv, wall_grad_temperature)

    return fluid_result, wall_result
