""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Treatment Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FluidBoundary

Boundary Conditions Base Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedFluidBoundary

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: DummyBoundary
.. autoclass:: AdiabaticSlipBoundary
.. autoclass:: AdiabaticNoslipMovingBoundary
.. autoclass:: IsothermalNoSlipBoundary
.. autoclass:: FarfieldBoundary
.. autoclass:: InflowBoundary
.. autoclass:: OutflowBoundary
.. autoclass:: IsothermalWallBoundary
.. autoclass:: AdiabaticNoslipWallBoundary
.. autoclass:: SymmetryBoundary

Auxilliary Utilities
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: grad_cv_wall_bc
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
from dataclasses import replace
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.dof_desc import VolumeDomainTag, as_dofdesc
from mirgecom.fluid import make_conserved
from grudge.trace_pair import TracePair
import grudge.op as op
from mirgecom.viscous import viscous_facial_flux_central
from mirgecom.flux import num_flux_central
from mirgecom.gas_model import make_fluid_state

from mirgecom.inviscid import inviscid_facial_flux_rusanov

from abc import ABCMeta, abstractmethod


class FluidBoundary(metaclass=ABCMeta):
    r"""Abstract interface to fluid boundary treatment.

    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux
    """

    @abstractmethod
    def inviscid_divergence_flux(self, discr, dd_bdry, gas_model, state_minus,
                                 numerical_flux_func, **kwargs):
        """Get the inviscid boundary flux for the divergence operator.

        This routine returns the facial flux used in the divergence
        of the inviscid fluid transport flux.

        Parameters
        ----------
        discr: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified by
            *dd_bdry*.

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        numerical_flux_func:

            Function should return the numerical flux corresponding to
            the divergence of the inviscid transport flux. This function
            is typically backed by an approximate Riemann solver, such as
            :func:`~mirgecom.inviscid.inviscid_facial_flux_rusanov`.

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """

    @abstractmethod
    def viscous_divergence_flux(self, discr, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func, **kwargs):
        """Get the viscous boundary flux for the divergence operator.

        This routine returns the facial flux used in the divergence
        of the viscous fluid transport flux.

        Parameters
        ----------
        discr: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified
            by *dd_bdry*.

        grad_cv_minus: :class:`~mirgecom.fluid.ConservedVars`

            The gradient of the conserved quantities on the (-) side
            of the boundary specified by *dd_bdry*.

        grad_t_minus: numpy.ndarray

            The gradient of the fluid temperature on the (-) side
            of the boundary specified by *dd_bdry*.

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        numerical_flux_func:

            Function should return the numerical flux corresponding to
            the divergence of the viscous transport flux. This function
            is typically backed by a helper, such as
            :func:`~mirgecom.viscous.viscous_facial_flux_central`.

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """

    @abstractmethod
    def cv_gradient_flux(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the boundary flux for the gradient of the fluid conserved variables.

        This routine returns the facial flux used by the gradient operator to
        compute the gradient of the fluid solution on a domain boundary.

        Parameters
        ----------
        discr: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified by
            *dd_bdry*.

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """

    @abstractmethod
    def temperature_gradient_flux(self, discr, dd_bdry, gas_model, state_minus,
                                  **kwargs):
        """Get the boundary flux for the gradient of the fluid temperature.

        This method returns the boundary flux to be used by the gradient
        operator when computing the gradient of the fluid temperature at a
        domain boundary.

        Parameters
        ----------
        discr: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified by
            *dd_bdry*.

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        Returns
        -------
        numpy.ndarray
        """


# This class is a FluidBoundary that provides default implementations of
# the abstract methods in FluidBoundary. This class will be eliminated
# by resolution of https://github.com/illinois-ceesd/mirgecom/issues/576.
# TODO: Don't do this. Make every boundary condition implement its own
# version of the FluidBoundary methods.
class PrescribedFluidBoundary(FluidBoundary):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux
    .. automethod:: av_flux
    """

    def __init__(self,
                 # returns the flux to be used in div op (prescribed flux)
                 inviscid_flux_func=None,
                 # returns CV+, to be used in num flux func (prescribed soln)
                 boundary_state_func=None,
                 # Flux to be used in grad(Temperature) op
                 temperature_gradient_flux_func=None,
                 # Function returns boundary temperature_plus
                 boundary_temperature_func=None,
                 # Function returns the flux to be used in grad(cv)
                 cv_gradient_flux_func=None,
                 # Function computes the numerical flux for a gradient
                 gradient_numerical_flux_func=None,
                 # Function computes the flux to be used in the div op
                 viscous_flux_func=None,
                 # Returns the boundary value for grad(cv)
                 boundary_gradient_cv_func=None,
                 # Returns the boundary value for grad(temperature)
                 boundary_gradient_temperature_func=None,
                 # For artificial viscosity - grad fluid soln on boundary
                 boundary_grad_av_func=None,
                 ):
        """Initialize the PrescribedFluidBoundary and methods."""
        self._bnd_state_func = boundary_state_func
        self._temperature_grad_flux_func = temperature_gradient_flux_func
        self._inviscid_flux_func = inviscid_flux_func
        self._bnd_temperature_func = boundary_temperature_func
        self._grad_num_flux_func = gradient_numerical_flux_func
        self._cv_gradient_flux_func = cv_gradient_flux_func
        self._viscous_flux_func = viscous_flux_func
        self._bnd_grad_cv_func = boundary_gradient_cv_func
        self._bnd_grad_temperature_func = boundary_gradient_temperature_func
        self._av_num_flux_func = num_flux_central
        self._bnd_grad_av_func = boundary_grad_av_func

        if not self._bnd_grad_av_func:
            self._bnd_grad_av_func = self._identical_grad_av

        if not self._inviscid_flux_func and not self._bnd_state_func:
            from warnings import warn
            warn("Using dummy boundary: copies interior solution.", stacklevel=2)

        if not self._inviscid_flux_func:
            self._inviscid_flux_func = self._inviscid_flux_for_prescribed_state

        if not self._bnd_state_func:
            self._bnd_state_func = self._identical_state

        if not self._bnd_temperature_func:
            self._bnd_temperature_func = self._temperature_for_prescribed_state
        if not self._grad_num_flux_func:
            self._grad_num_flux_func = num_flux_central

        if not self._cv_gradient_flux_func:
            self._cv_gradient_flux_func = self._gradient_flux_for_prescribed_cv
        if not self._temperature_grad_flux_func:
            self._temperature_grad_flux_func = \
                self._gradient_flux_for_prescribed_temperature

        if not self._viscous_flux_func:
            self._viscous_flux_func = self._viscous_flux_for_prescribed_state
        if not self._bnd_grad_cv_func:
            self._bnd_grad_cv_func = self._identical_grad_cv
        if not self._bnd_grad_temperature_func:
            self._bnd_grad_temperature_func = self._identical_grad_temperature

    def _boundary_quantity(self, discr, dd_bdry, quantity, local=False, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        dd_allfaces = dd_bdry.with_domain_tag(
            replace(dd_bdry.domain_tag, tag=FACE_RESTR_ALL))
        return quantity if local else op.project(discr,
            dd_bdry, dd_allfaces, quantity)

    def _boundary_state_pair(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        return TracePair(dd_bdry,
                         interior=state_minus,
                         exterior=self._bnd_state_func(discr=discr, dd_bdry=dd_bdry,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus,
                                                       **kwargs))
    # The following methods provide default implementations of the fluid
    # boundary functions and helpers in an effort to eliminate much
    # repeated code. They will be eliminated by the resolution of
    # https://github.com/illinois-ceesd/mirgecom/issues/576.

    # {{{ Default boundary helpers

    # Returns temperature(+) for boundaries that prescribe CV(+)
    def _temperature_for_prescribed_state(self, discr, dd_bdry,
                                          gas_model, state_minus, **kwargs):
        boundary_state = self._bnd_state_func(discr=discr, dd_bdry=dd_bdry,
                                              gas_model=gas_model,
                                              state_minus=state_minus,
                                              **kwargs)
        return boundary_state.temperature

    def _temperature_for_interior_state(self, discr, dd_bdry, gas_model, state_minus,
                                        **kwargs):
        return state_minus.temperature

    def _identical_state(self, state_minus, **kwargs):
        return state_minus

    def _identical_grad_cv(self, grad_cv_minus, **kwargs):
        return grad_cv_minus

    def _identical_grad_temperature(self, discr, dd_bdry, grad_t_minus, **kwargs):
        return grad_t_minus

    # Returns the flux to be used by the gradient operator when computing the
    # gradient of the fluid solution on boundaries that prescribe CV(+).
    def _gradient_flux_for_prescribed_cv(self, discr, dd_bdry, gas_model,
                                         state_minus, **kwargs):
        # Use prescribed external state and gradient numerical flux function
        boundary_state = self._bnd_state_func(discr=discr, dd_bdry=dd_bdry,
                                              gas_model=gas_model,
                                              state_minus=state_minus,
                                              **kwargs)
        cv_pair = TracePair(dd_bdry,
                            interior=state_minus.cv,
                            exterior=boundary_state.cv)

        actx = state_minus.array_context
        nhat = actx.thaw(discr.normal(dd_bdry))
        from arraycontext import outer
        return outer(self._grad_num_flux_func(cv_pair.int, cv_pair.ext), nhat)

    # Returns the flux to be used by the gradient operator when computing the
    # gradient of fluid temperature using prescribed fluid temperature(+).
    def _gradient_flux_for_prescribed_temperature(self, discr, dd_bdry, gas_model,
                                                  state_minus, **kwargs):
        # Feed a boundary temperature to numerical flux for grad op
        actx = state_minus.array_context
        nhat = actx.thaw(discr.normal(dd_bdry))
        bnd_tpair = TracePair(dd_bdry,
                              interior=state_minus.temperature,
                              exterior=self._bnd_temperature_func(
                                  discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
                                  state_minus=state_minus, **kwargs))
        from arraycontext import outer
        return outer(self._grad_num_flux_func(bnd_tpair.int, bnd_tpair.ext), nhat)

    # Returns the flux to be used by the divergence operator when computing the
    # divergence of inviscid fluid transport flux using the boundary's
    # prescribed CV(+).
    def _inviscid_flux_for_prescribed_state(
            self, discr, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        # Use a prescribed boundary state and the numerical flux function
        boundary_state_pair = self._boundary_state_pair(discr=discr, dd_bdry=dd_bdry,
                                                        gas_model=gas_model,
                                                        state_minus=state_minus,
                                                        **kwargs)
        normal = state_minus.array_context.thaw(discr.normal(dd_bdry))
        return numerical_flux_func(boundary_state_pair, gas_model, normal)

    # Returns the flux to be used by the divergence operator when computing the
    # divergence of viscous fluid transport flux using the boundary's
    # prescribed CV(+).
    def _viscous_flux_for_prescribed_state(
            self, discr, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, numerical_flux_func=viscous_facial_flux_central, **kwargs):

        state_pair = self._boundary_state_pair(
            discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

        grad_cv_pair = \
            TracePair(dd_bdry, interior=grad_cv_minus,
                      exterior=self._bnd_grad_cv_func(
                          discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
                          state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                          grad_t_minus=grad_t_minus))

        grad_t_pair = \
            TracePair(
                dd_bdry, interior=grad_t_minus,
                exterior=self._bnd_grad_temperature_func(
                    discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
                    state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                    grad_t_minus=grad_t_minus))

        return numerical_flux_func(
            discr=discr, gas_model=gas_model, state_pair=state_pair,
            grad_cv_pair=grad_cv_pair, grad_t_pair=grad_t_pair)

    # }}} Default boundary helpers

    def inviscid_divergence_flux(self, discr, dd_bdry, gas_model, state_minus,
                                 numerical_flux_func=inviscid_facial_flux_rusanov,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._inviscid_flux_func(discr, dd_bdry, gas_model, state_minus,
                                        numerical_flux_func=numerical_flux_func,
                                        **kwargs)

    def cv_gradient_flux(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the flux for *dd_bdry* for use in grad(CV)."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._cv_gradient_flux_func(
            discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

    def temperature_gradient_flux(self, discr, dd_bdry, gas_model, state_minus,
                                  **kwargs):
        """Get the flux for *dd_bdry* for use in grad(T)."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._temperature_grad_flux_func(discr, dd_bdry, gas_model,
                                                state_minus, **kwargs)

    def viscous_divergence_flux(self, discr, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func=viscous_facial_flux_central,
                                **kwargs):
        """Get the viscous flux for *dd_bdry* for use in the divergence operator."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._viscous_flux_func(discr=discr, dd_bdry=dd_bdry,
                                       gas_model=gas_model,
                                       state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       grad_t_minus=grad_t_minus,
                                       numerical_flux_func=numerical_flux_func,
                                       **kwargs)

    # {{{ Boundary interface for artificial viscosity

    def _identical_grad_av(self, grad_av_minus, **kwargs):
        return grad_av_minus

    def av_flux(self, discr, dd_bdry, diffusion, **kwargs):
        """Get the diffusive fluxes for the AV operator API."""
        dd_bdry = as_dofdesc(dd_bdry)
        dd_vol = dd_bdry.with_domain_tag(
            VolumeDomainTag(dd_bdry.domain_tag.volume_tag))
        grad_av_minus = op.project(discr, dd_vol, dd_bdry, diffusion)
        actx = grad_av_minus.mass[0].array_context
        nhat = actx.thaw(discr.normal(dd_bdry))
        grad_av_plus = self._bnd_grad_av_func(
            discr=discr, dd_bdry=dd_bdry, grad_av_minus=grad_av_minus, **kwargs)
        bnd_grad_pair = TracePair(dd_bdry, interior=grad_av_minus,
                                  exterior=grad_av_plus)
        num_flux = self._av_num_flux_func(bnd_grad_pair.int, bnd_grad_pair.ext)@nhat
        return self._boundary_quantity(discr, dd_bdry, num_flux, **kwargs)

    # }}}


class DummyBoundary(PrescribedFluidBoundary):
    """Boundary type that assigns boundary-adjacent solution to the boundary."""

    def __init__(self):
        """Initialize the DummyBoundary boundary type."""
        PrescribedFluidBoundary.__init__(self)


class AdiabaticSlipBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing inviscid slip boundary.

    This function is deprecated and should be replaced by
    :class:`~mirgecom.boundary.SymmetryBoundary`

    .. automethod:: adiabatic_slip_state
    .. automethod:: adiabatic_slip_grad_av
    """

    def __init__(self):
        """Initialize AdiabaticSlipBoundary."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_slip_state,
            boundary_temperature_func=self._temperature_for_interior_state,
            boundary_grad_av_func=self.adiabatic_slip_grad_av
        )

    from warnings import warn
    warn("AdiabaticSlipBoundary is deprecated. Use SymmetryBoundary instead.",
        PendingDeprecationWarning)

    def adiabatic_slip_state(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        The exterior solution is set such that there will be vanishing
        flux through the boundary, preserving mass, momentum (magnitude) and
        energy.
        rho_plus = rho_minus
        v_plus = v_minus - 2 * (v_minus . n_hat) * n_hat
        mom_plus = rho_plus * v_plus
        E_plus = E_minus
        """
        # Grab some boundary-relevant data
        dim = state_minus.dim
        actx = state_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(discr.normal(dd_bdry))

        # Subtract out the 2*wall-normal component
        # of velocity from the velocity at the wall to
        # induce an equal but opposite wall-normal (reflected) wave
        # preserving the tangential component
        cv_minus = state_minus.cv
        ext_mom = (cv_minus.momentum
                   - 2.0*np.dot(cv_minus.momentum, nhat)*nhat)

        # Form the external boundary solution with the new momentum
        ext_cv = make_conserved(dim=dim, mass=cv_minus.mass, energy=cv_minus.energy,
                                momentum=ext_mom, species_mass=cv_minus.species_mass)
        return make_fluid_state(cv=ext_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def adiabatic_slip_grad_av(self, discr, dd_bdry, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary for artificial viscosity."""
        # Grab some boundary-relevant data
        dim, = grad_av_minus.mass.shape
        actx = grad_av_minus.mass[0].array_context
        nhat = actx.thaw(discr.norm(dd_bdry))

        # Subtract 2*wall-normal component of q
        # to enforce q=0 on the wall
        s_mom_normcomp = np.outer(nhat,
                                  np.dot(grad_av_minus.momentum, nhat))
        s_mom_flux = grad_av_minus.momentum - 2*s_mom_normcomp

        # flip components to set a neumann condition
        return make_conserved(dim, mass=-grad_av_minus.mass,
                              energy=-grad_av_minus.energy,
                              momentum=-s_mom_flux,
                              species_mass=-grad_av_minus.species_mass)


class AdiabaticNoslipMovingBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing a no-slip moving boundary.

    This function is deprecated and should be replaced by
    :class:`~mirgecom.boundary.AdiabaticNoslipWallBoundary`

    .. automethod:: adiabatic_noslip_state
    .. automethod:: adiabatic_noslip_grad_av
    """

    def __init__(self, wall_velocity=None, dim=2):
        """Initialize boundary device."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_noslip_state,
            boundary_temperature_func=self._temperature_for_interior_state,
            boundary_grad_av_func=self.adiabatic_noslip_grad_av,
        )
        # Check wall_velocity (assumes dim is correct)
        if wall_velocity is None:
            wall_velocity = np.zeros(shape=(dim,))
        if len(wall_velocity) != dim:
            raise ValueError(f"Specified wall velocity must be {dim}-vector.")
        self._wall_velocity = wall_velocity

    from warnings import warn
    warn("AdiabaticNoslipMovingBoundary is deprecated."
         " Use AdiabaticNoslipWallBoundary instead.",
        PendingDeprecationWarning)

    def adiabatic_noslip_state(
            self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        Sets the external state s.t. $v^+ = -v^-$, giving vanishing contact velocity
        in the approximate Riemann solver used to compute the inviscid flux.
        """
        wall_pen = 2.0 * self._wall_velocity * state_minus.mass_density
        ext_mom = wall_pen - state_minus.momentum_density  # no-slip

        # Form the external boundary solution with the new momentum
        cv = make_conserved(dim=state_minus.dim, mass=state_minus.mass_density,
                            energy=state_minus.energy_density,
                            momentum=ext_mom,
                            species_mass=state_minus.species_mass_density)
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def adiabatic_noslip_grad_av(self, grad_av_minus, **kwargs):
        """Get the exterior solution on the boundary for artificial viscosity."""
        return(-grad_av_minus)


class IsothermalNoSlipBoundary(PrescribedFluidBoundary):
    r"""Isothermal no-slip viscous wall boundary.

    This function is deprecated and should be replaced by
    :class:`~mirgecom.boundary.IsothermalWallBoundary`

    .. automethod:: isothermal_noslip_state
    .. automethod:: temperature_bc
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.isothermal_noslip_state,
            boundary_temperature_func=self.temperature_bc
        )

    from warnings import warn
    warn("IsothermalWallBoundary is deprecated."
         " Use IsothermalWallBoundary instead.",
        PendingDeprecationWarning)

    def isothermal_noslip_state(
            self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        r"""Get the interior and exterior solution (*state_minus*) on the boundary.

        Sets the external state s.t. $v^+ = -v^-$, giving vanishing contact velocity
        in the approximate Riemann solver used to compute the inviscid flux.
        """
        temperature_wall = self._wall_temp + 0*state_minus.mass_density
        velocity_plus = -state_minus.velocity
        mass_frac_plus = state_minus.species_mass_fractions

        internal_energy_plus = gas_model.eos.get_internal_energy(
            temperature=temperature_wall, species_mass_fractions=mass_frac_plus)

        total_energy_plus = state_minus.mass_density*(internal_energy_plus
                                           + .5*np.dot(velocity_plus, velocity_plus))

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density, energy=total_energy_plus,
            momentum=-state_minus.momentum_density,
            species_mass=state_minus.species_mass_density
        )
        tseed = state_minus.temperature if state_minus.is_mixture else None
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=tseed)

    def temperature_bc(self, state_minus, **kwargs):
        r"""Get temperature value to weakly prescribe wall bc.

        Returns $2T_\text{wall} - T^-$ so that a central gradient flux
        will get the correct $T_\text{wall}$ BC.
        """
        return 2*self._wall_temp - state_minus.temperature


class FarfieldBoundary(PrescribedFluidBoundary):
    r"""Farfield boundary treatment.

    This class implements a farfield boundary as described by
    [Mengaldo_2014]_ eqn. 30 and eqn. 42.  The boundary condition is implemented
    as:

    .. math::
        q^{+} = q_\infty

    and the gradients

    .. math::
        \nabla q_{bc} = \nabla q^{-}

    .. automethod:: __init__
    .. automethod:: farfield_state
    .. automethod:: temperature_bc
    """

    def __init__(self, numdim, numspecies,
                 free_stream_pressure=None,
                 free_stream_velocity=None,
                 free_stream_density=None,
                 free_stream_temperature=None,
                 free_stream_mass_fractions=None):
        """Initialize the boundary condition object."""
        if free_stream_velocity is None:
            raise ValueError("Free-stream velocity must be specified")
        if len(free_stream_velocity) != numdim:
            raise ValueError("Free-stream velocity must be of ambient dimension.")

        if numspecies > 0:
            if free_stream_pressure is None:
                raise ValueError("Free-stream pressure must be given.")
            if free_stream_temperature is None:
                raise ValueError("Free-stream temperature must be given.")
            if free_stream_mass_fractions is None:
                raise ValueError("Free-stream species mixture fractions must be"
                                 " given.")
            if len(free_stream_mass_fractions) != numspecies:
                raise ValueError("Free-stream species mixture fractions of improper"
                                 " size.")

        self._temperature = free_stream_temperature
        self._pressure = free_stream_pressure
        self._species_mass_fractions = free_stream_mass_fractions
        self._velocity = free_stream_velocity
        self._density = free_stream_density

        num_specified = sum(x is not None for x in [free_stream_temperature,
                                                    free_stream_pressure,
                                                    free_stream_density])
        if num_specified != 2:
            raise ValueError("Two of [free_stream_pressure, free_stream_temperature,"
                             " free_stream_density], must be provided.")

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.farfield_state
        )

    def farfield_state(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        dd_bdry = as_dofdesc(dd_bdry)
        free_stream_mass_fractions = (0.*state_minus.species_mass_fractions
                                      + self._species_mass_fractions)

        if self._temperature is not None:
            free_stream_temperature = 0.*state_minus.temperature + self._temperature
        if self._pressure is not None:
            free_stream_pressure = 0.*state_minus.pressure + self._pressure
        if self._density is not None:
            free_stream_density = 0.*state_minus.cv.mass + self._density

        if self._temperature is None:
            self._temperature = \
                self._pressure/(gas_model.eos.gas_const()*self._density)
            free_stream_temperature = 0.*state_minus.temperature + self._temperature
        if self._pressure is None:
            free_stream_pressure = \
                gas_model.eos.gas_const()*self._density*self._temperature
        if self._density is None:
            free_stream_density = gas_model.eos.get_density(
                pressure=free_stream_pressure, temperature=free_stream_temperature,
                species_mass_fractions=free_stream_mass_fractions)

        free_stream_velocity = 0.*state_minus.velocity + self._velocity
        free_stream_internal_energy = gas_model.eos.get_internal_energy(
            temperature=free_stream_temperature,
            species_mass_fractions=free_stream_mass_fractions)

        free_stream_total_energy = \
            free_stream_density*(free_stream_internal_energy
                                 + .5*np.dot(free_stream_velocity,
                                             free_stream_velocity))
        free_stream_spec_mass = free_stream_density * free_stream_mass_fractions

        cv_infinity = make_conserved(
            state_minus.dim, mass=free_stream_density,
            energy=free_stream_total_energy,
            momentum=free_stream_density*free_stream_velocity,
            species_mass=free_stream_spec_mass
        )

        return make_fluid_state(cv=cv_infinity, gas_model=gas_model,
                                temperature_seed=free_stream_temperature)

    def temperature_bc(self, state_minus, **kwargs):
        """Get temperature value to weakly prescribe flow temperature at boundary."""
        return 0*state_minus.temperature + self._temperature


class OutflowBoundary(PrescribedFluidBoundary):
    r"""Outflow boundary treatment.

    This class implements an outflow boundary as described by
    [Mengaldo_2014]_.  The boundary condition is implemented
    as:

    .. math::

        \rho^+ &= \rho^-

        \rho\mathbf{Y}^+ &= \rho\mathbf{Y}^-

        \rho\mathbf{V}^+ &= \rho\mathbf{V}^-

    For an ideal gas at super-sonic flow conditions, i.e. when:

    .. math::

       \rho\mathbf{V} \cdot \hat{\mathbf{n}} \ge c,

    then the pressure is extrapolated from interior points:

    .. math::

        P^+ = P^-

    Otherwise, if the flow is sub-sonic, then the prescribed boundary pressure,
    $P^+$, is used. In both cases, the energy is computed as:

    .. math::

        \rho{E}^+ = \frac{\left(2~P^+ - P^-\right)}{\left(\gamma-1\right)}
        + \frac{1}{2}\rho^+\left(\mathbf{V}^+\cdot\mathbf{V}^+\right).

    For mixtures, the pressure is imposed or extrapolated in a similar fashion
    to the ideal gas case.
    However, the total energy depends on the temperature to account for the
    species enthalpy and variable specific heat at constant volume. For super-sonic
    flows, it is extrapolated from interior points:

    .. math::

       T^+ = T^-

    while for sub-sonic flows, it is evaluated using ideal gas law

    .. math::

        T^+ = \frac{P^+}{R_{mix} \rho^+}

    .. automethod:: __init__
    .. automethod:: outflow_state
    """

    def __init__(self, boundary_pressure=101325):
        """Initialize the boundary condition object."""
        self._pressure = boundary_pressure
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.outflow_state
        )

    def outflow_state(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.

        For super-sonic outflow, the interior flow properties (minus) are
        extrapolated to the exterior point (plus).
        For sub-sonic outflow, the pressure is imposed on the external point.

        For mixtures, the internal energy is obtained via temperature, which comes
        from ideal gas law with the mixture-weighted gas constant.
        For ideal gas, the internal energy is obtained directly from pressure.
        """
        dd_bdry = as_dofdesc(dd_bdry)
        actx = state_minus.array_context
        nhat = actx.thaw(discr.normal(dd_bdry))
        # boundary-normal velocity
        boundary_vel = np.dot(state_minus.velocity, nhat)*nhat
        boundary_speed = actx.np.sqrt(np.dot(boundary_vel, boundary_vel))
        speed_of_sound = state_minus.speed_of_sound
        kinetic_energy = gas_model.eos.kinetic_energy(state_minus.cv)
        gamma = gas_model.eos.gamma(state_minus.cv, state_minus.temperature)

        pressure_plus = 2.0*self._pressure - state_minus.pressure
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(state_minus.cv)
            temp_plus = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                state_minus.temperature,
                pressure_plus/(state_minus.cv.mass*gas_const))
            )

            internal_energy = state_minus.cv.mass*(
                gas_model.eos.get_internal_energy(temp_plus,
                                            state_minus.species_mass_fractions))
        else:
            boundary_pressure = actx.np.where(actx.np.greater(boundary_speed,
                                                              speed_of_sound),
                                              state_minus.pressure, pressure_plus)
            internal_energy = (boundary_pressure / (gamma - 1.0))

        total_energy = internal_energy + kinetic_energy
        cv_outflow = make_conserved(dim=state_minus.dim, mass=state_minus.cv.mass,
                                    momentum=state_minus.cv.momentum,
                                    energy=total_energy,
                                    species_mass=state_minus.cv.species_mass)

        return make_fluid_state(cv=cv_outflow, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)


class InflowBoundary(PrescribedFluidBoundary):
    r"""Inflow boundary treatment.

    This class implements an Riemann invariant for inflow boundary as described by
    [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: inflow_state
    """

    def __init__(self, dim, free_stream_pressure=None, free_stream_temperature=None,
                 free_stream_density=None, free_stream_velocity=None,
                 free_stream_mass_fractions=None, gas_model=None):
        """Initialize the boundary condition object."""
        if free_stream_velocity is None:
            raise ValueError("InflowBoundary requires *free_stream_velocity*.")

        from mirgecom.initializers import initialize_fluid_state
        self._free_stream_state = initialize_fluid_state(
            dim, gas_model, density=free_stream_density,
            velocity=free_stream_velocity,
            mass_fractions=free_stream_mass_fractions, pressure=free_stream_pressure,
            temperature=free_stream_temperature)

        self._gamma = gas_model.eos.gamma(
            self._free_stream_state.cv,
            temperature=self._free_stream_state.temperature
        )

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.inflow_state
        )

    def inflow_state(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.
        """
        dd_bdry = as_dofdesc(dd_bdry)
        actx = state_minus.array_context
        nhat = actx.thaw(discr.normal(dd_bdry))

        v_plus = np.dot(self._free_stream_state.velocity, nhat)
        rho_plus = self._free_stream_state.mass_density
        c_plus = self._free_stream_state.speed_of_sound
        gamma_plus = self._gamma

        v_minus = np.dot(state_minus.velocity, nhat)
        gamma_minus = gas_model.eos.gamma(state_minus.cv,
                                          temperature=state_minus.temperature)
        c_minus = state_minus.speed_of_sound

        ones = 0*v_minus + 1
        r_plus_subsonic = v_minus + 2*c_minus/(gamma_minus - 1)
        r_plus_supersonic = (v_plus + 2*c_plus/(gamma_plus - 1))*ones
        r_minus = v_plus - 2*c_plus/(gamma_plus - 1)*ones
        r_plus = actx.np.where(actx.np.greater(v_minus, c_minus), r_plus_supersonic,
                               r_plus_subsonic)

        velocity_boundary = (r_minus + r_plus)/2
        velocity_boundary = (
            self._free_stream_state.velocity + (velocity_boundary - v_plus)*nhat
        )

        c_boundary = (gamma_plus - 1)*(r_plus - r_minus)/4
        c_boundary2 = c_boundary**2
        entropy_boundary = c_plus*c_plus/(gamma_plus*rho_plus**(gamma_plus-1))
        rho_boundary = c_boundary*c_boundary/(gamma_plus * entropy_boundary)
        pressure_boundary = rho_boundary * c_boundary2 / gamma_plus
        energy_boundary = (
            pressure_boundary / (gamma_plus - 1)
            + rho_boundary*np.dot(velocity_boundary, velocity_boundary)
        )
        species_mass_boundary = None
        if self._free_stream_state.is_mixture:
            species_mass_boundary = (
                rho_boundary * self._free_stream_state.species_mass_fractions
            )

        boundary_cv = make_conserved(dim=state_minus.dim, mass=rho_boundary,
                                     energy=energy_boundary,
                                     momentum=rho_boundary * velocity_boundary,
                                     species_mass=species_mass_boundary)

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)


def grad_cv_wall_bc(self, state_minus, grad_cv_minus, normal, **kwargs):
    """Return grad(CV) modified for no-penetration of solid wall."""
    from mirgecom.fluid import (
        velocity_gradient,
        species_mass_fraction_gradient
    )

    # Velocity part
    grad_v_minus = velocity_gradient(state_minus, grad_cv_minus)
    grad_v_plus = grad_v_minus - np.outer(grad_v_minus@normal, normal)
    grad_mom_plus = 0*grad_v_plus
    for i in range(state_minus.dim):
        grad_mom_plus[i] = (state_minus.mass_density*grad_v_plus[i]
                            + state_minus.velocity[i]*grad_cv_minus.mass)

    # species mass fraction part
    grad_species_mass_plus = 0.*grad_cv_minus.species_mass
    if state_minus.nspecies:
        grad_y_minus = species_mass_fraction_gradient(state_minus.cv, grad_cv_minus)
        grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)

        for i in range(state_minus.nspecies):
            grad_species_mass_plus[i] = \
                (state_minus.mass_density*grad_y_plus[i]
                 + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

    return make_conserved(state_minus.dim, mass=grad_cv_minus.mass,
                          energy=grad_cv_minus.energy, momentum=grad_mom_plus,
                          species_mass=grad_species_mass_plus)


class IsothermalWallBoundary(PrescribedFluidBoundary):
    r"""Isothermal viscous wall boundary.

    This class implements an isothermal no-slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: inviscid_wall_flux
    .. automethod:: viscous_wall_flux
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: isothermal_wall_state
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.isothermal_wall_state,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_temperature_func=self.temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

    def isothermal_wall_state(
            self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero-velocity and the respective internal energy."""
        temperature_wall = self._wall_temp + 0*state_minus.mass_density
        mom_plus = state_minus.mass_density*0.*state_minus.velocity
        mass_frac_plus = state_minus.species_mass_fractions

        internal_energy_plus = gas_model.eos.get_internal_energy(
            temperature=temperature_wall, species_mass_fractions=mass_frac_plus)

        # Velocity is pinned to 0 here, no kinetic energy
        total_energy_plus = state_minus.mass_density*internal_energy_plus

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density, energy=total_energy_plus,
            momentum=mom_plus, species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def inviscid_wall_flux(self, discr, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """Return Riemann flux using state with mom opposite of interior state."""
        dd_bdry = as_dofdesc(dd_bdry)
        wall_cv = make_conserved(dim=state_minus.dim,
                                 mass=state_minus.mass_density,
                                 momentum=-state_minus.momentum_density,
                                 energy=state_minus.energy_density,
                                 species_mass=state_minus.species_mass_density)
        wall_state = make_fluid_state(cv=wall_cv, gas_model=gas_model,
                                      temperature_seed=state_minus.temperature)
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=wall_state)

        normal = state_minus.array_context.thaw(discr.normal(dd_bdry))
        return numerical_flux_func(state_pair, gas_model, normal)

    def temperature_bc(self, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        # return 2*self._wall_temp - state_minus.temperature
        return 0.*state_minus.temperature + self._wall_temp

    def grad_cv_bc(self, state_minus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        grad_species_mass_plus = 1.*grad_cv_minus.species_mass
        if state_minus.nspecies > 0:
            from mirgecom.fluid import species_mass_fraction_gradient
            grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                          grad_cv_minus)
            grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)
            grad_species_mass_plus = 0.*grad_y_plus

            for i in range(state_minus.nspecies):
                grad_species_mass_plus[i] = \
                    (state_minus.mass_density*grad_y_plus[i]
                     + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        return make_conserved(grad_cv_minus.dim,
                              mass=grad_cv_minus.mass,
                              energy=grad_cv_minus.energy,
                              momentum=grad_cv_minus.momentum,
                              species_mass=grad_species_mass_plus)

    def viscous_wall_flux(self, discr, dd_bdry, gas_model, state_minus,
                          grad_cv_minus, grad_t_minus,
                          numerical_flux_func=viscous_facial_flux_central,
                          **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(discr.normal(dd_bdry))

        state_plus = self.isothermal_wall_state(discr=discr, dd_bdry=dd_bdry,
                                                gas_model=gas_model,
                                                state_minus=state_minus, **kwargs)
        grad_cv_plus = self.grad_cv_bc(state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)

        grad_t_plus = self._bnd_grad_temperature_func(
            discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, grad_cv_minus=grad_cv_minus,
            grad_t_minus=grad_t_minus)

        # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
        # *not* the numerical viscous flux as advised by [Bassi_1997]_.
        f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                             grad_t=grad_t_plus)
        return f_ext@normal


class AdiabaticNoslipWallBoundary(PrescribedFluidBoundary):
    r"""Adiabatic viscous wall boundary.

    This class implements an adiabatic no-slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: inviscid_wall_flux
    .. automethod:: viscous_wall_flux
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: adiabatic_wall_state_for_advection
    .. automethod:: adiabatic_wall_state_for_diffusion
    .. automethod:: grad_temperature_bc
    .. automethod:: adiabatic_noslip_grad_av
    """

    def __init__(self):
        """Initialize the boundary condition object."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_wall_state_for_advection,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_temperature_func=self.temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

    def adiabatic_wall_state_for_advection(self, discr, dd_bdry, gas_model,
                                           state_minus, **kwargs):
        """Return state with zero-velocity."""
        dd_bdry = as_dofdesc(dd_bdry)
        mom_plus = -state_minus.momentum_density
        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density,
            energy=state_minus.energy_density, momentum=mom_plus,
            species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def adiabatic_wall_state_for_diffusion(self, discr, dd_bdry, gas_model,
                                           state_minus, **kwargs):
        """Return state with zero-velocity."""
        dd_bdry = as_dofdesc(dd_bdry)
        mom_plus = 0*state_minus.momentum_density
        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density,
            energy=state_minus.energy_density, momentum=mom_plus,
            species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def inviscid_wall_flux(self, discr, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """Return Riemann flux using state with mom opposite of interior state."""
        dd_bdry = as_dofdesc(dd_bdry)
        wall_state = self.adiabatic_wall_state_for_advection(
            discr, dd_bdry, gas_model, state_minus)
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=wall_state)

        normal = state_minus.array_context.thaw(discr.normal(dd_bdry))
        return numerical_flux_func(state_pair, gas_model, normal)

    def temperature_bc(self, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def grad_cv_bc(self, state_minus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        grad_species_mass_plus = 1.*grad_cv_minus.species_mass
        if state_minus.nspecies > 0:
            from mirgecom.fluid import species_mass_fraction_gradient
            grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                          grad_cv_minus)
            grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)
            grad_species_mass_plus = 0.*grad_y_plus

            for i in range(state_minus.nspecies):
                grad_species_mass_plus[i] = \
                    (state_minus.mass_density*grad_y_plus[i]
                     + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        return make_conserved(grad_cv_minus.dim,
                              mass=grad_cv_minus.mass,
                              energy=grad_cv_minus.energy,
                              momentum=grad_cv_minus.momentum,
                              species_mass=grad_species_mass_plus)

    def grad_temperature_bc(self, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus - np.dot(grad_t_minus, normal)*normal

    def viscous_wall_flux(self, discr, dd_bdry, gas_model, state_minus,
                          grad_cv_minus, grad_t_minus,
                          numerical_flux_func=viscous_facial_flux_central,
                          **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(discr.normal(dd_bdry))

        state_plus = self.adiabatic_wall_state_for_diffusion(
            discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus)

        grad_cv_plus = self.grad_cv_bc(state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)
        grad_t_plus = self.grad_temperature_bc(grad_t_minus, normal)

        # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
        # *not* the numerical viscous flux as advised by [Bassi_1997]_.
        f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                             grad_t=grad_t_plus)

        return f_ext@normal

    def adiabatic_noslip_grad_av(self, grad_av_minus, **kwargs):
        """Get the exterior solution on the boundary for artificial viscosity."""
        return(-grad_av_minus)


class SymmetryBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing symmetry/slip wall boundary.

    a.k.a. Reflective inviscid wall boundary

    This class implements an adiabatic reflective slip boundary given
    by
    $\mathbf{q^{+}} = [\rho^{-}, (\rho{E})^{-}, (\rho\vec{V})^{-}
    - 2((\rho\vec{V})^{-}\cdot\hat{\mathbf{n}}) \hat{\mathbf{n}}]$
    wherein the normal component of velocity at the wall is 0, and
    tangential components are preserved. These perfectly reflecting
    conditions are used by the forward-facing step case in
    [Hesthaven_2008]_, Section 6.6, and correspond to the characteristic
    boundary conditions described in detail in [Poinsot_1992]_.

    For the gradients, the no-shear condition implies that cross-terms are absent
    and that temperature gradients are null due to the adiabatic condition.

    .. math::

        \nabla u ^+ = \nabla{u}^- \circ I

        \nabla T \cdot n = 0

    .. automethod:: inviscid_wall_flux
    .. automethod:: viscous_wall_flux
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: adiabatic_wall_state_for_advection
    .. automethod:: adiabatic_wall_state_for_diffusion
    .. automethod:: grad_temperature_bc
    .. automethod:: adiabatic_slip_grad_av
    """

    def __init__(self, dim=2):
        """Initialize the boundary condition object."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_wall_state_for_advection,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_temperature_func=self.temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

        self._dim = dim

    def adiabatic_wall_state_for_advection(self, discr, dd_bdry, gas_model,
                                           state_minus, **kwargs):
        """Return state with opposite normal momentum."""
        actx = state_minus.array_context
        nhat = actx.thaw(discr.normal(dd_bdry))

        # flip the normal component of the velocity
        mom_plus = (state_minus.momentum_density
             - 2*(np.dot(state_minus.momentum_density, nhat)*nhat))

        # no changes are necessary to the energy equation because the velocity
        # magnitude is the same, only the (normal) direction changes.

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density,
            energy=state_minus.energy_density, momentum=mom_plus,
            species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def adiabatic_wall_state_for_diffusion(self, discr, dd_bdry, gas_model,
                                           state_minus, **kwargs):
        """Return state with zero normal-velocity and energy(Twall)."""
        actx = state_minus.array_context
        nhat = actx.thaw(discr.normal(dd_bdry))

        # remove normal component from velocity/momentum
        mom_plus = (state_minus.momentum_density
                      - 1*(np.dot(state_minus.momentum_density, nhat)*nhat))

        # modify energy accordingly
        kinetic_energy_plus = 0.5*np.dot(mom_plus, mom_plus)/state_minus.mass_density
        internal_energy_plus = (
            state_minus.mass_density * gas_model.eos.get_internal_energy(
                temperature=state_minus.temperature,
                species_mass_fractions=state_minus.species_mass_fractions))

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density,
            energy=kinetic_energy_plus + internal_energy_plus,
            momentum=mom_plus,
            species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def inviscid_wall_flux(self, discr, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """Return Riemann flux using state with mom opposite of interior state."""
        dd_bdry = as_dofdesc(dd_bdry)
        wall_state = self.adiabatic_wall_state_for_advection(
            discr, dd_bdry, gas_model, state_minus)
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=wall_state)

        actx = state_minus.array_context
        normal = actx.thaw(discr.normal(dd_bdry))
        return numerical_flux_func(state_pair, gas_model, normal)

    def temperature_bc(self, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def grad_cv_bc(self, state_minus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        grad_species_mass_plus = 1.*grad_cv_minus.species_mass
        if state_minus.nspecies > 0:
            from mirgecom.fluid import species_mass_fraction_gradient
            grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                          grad_cv_minus)
            grad_y_plus = grad_y_minus - np.outer(grad_y_minus@normal, normal)
            grad_species_mass_plus = 0.*grad_y_plus

            for i in range(state_minus.nspecies):
                grad_species_mass_plus[i] = \
                    (state_minus.mass_density*grad_y_plus[i]
                     + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        # extrapolate density and its gradient
        mass_plus = state_minus.mass_density
        grad_mass_plus = grad_cv_minus.mass

        from mirgecom.fluid import velocity_gradient
        v_minus = state_minus.velocity
        grad_v_minus = velocity_gradient(state_minus.cv, grad_cv_minus)

        # modify velocity gradient at the boundary:
        # remove normal component of velocity
        v_plus = state_minus.velocity \
                      - 1*np.dot(state_minus.velocity, normal)*normal
        # retain only the diagonal terms to force zero shear stress
        grad_v_plus = grad_v_minus*np.eye(self._dim)

        # product rule for momentum
        grad_momentum_density_plus = mass_plus*grad_v_plus + v_plus*grad_mass_plus

        # the energy has to be modified accordingly:
        # first, get gradient of internal energy, i.e., no kinetic energy
        grad_int_energy_minus = grad_cv_minus.energy \
            - 0.5*(np.dot(v_minus, v_minus)*grad_cv_minus.mass
                + 2.0*state_minus.mass_density * np.dot(v_minus, grad_v_minus))
        grad_int_energy_plus = grad_int_energy_minus
        # then modify gradient of kinetic energy to match the changes in velocity
        grad_kin_energy_plus = \
            0.5*(np.dot(v_plus, v_plus)*grad_mass_plus
                + 2.0*mass_plus * np.dot(v_plus, grad_v_plus))
        grad_energy_plus = grad_int_energy_plus + grad_kin_energy_plus

        return make_conserved(grad_cv_minus.dim,
                              mass=grad_mass_plus,
                              energy=grad_energy_plus,
                              momentum=grad_momentum_density_plus,
                              species_mass=grad_species_mass_plus)

    def grad_temperature_bc(self, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus - np.dot(grad_t_minus, normal)*normal

    def viscous_wall_flux(self, discr, dd_bdry, gas_model, state_minus,
                          grad_cv_minus, grad_t_minus,
                          numerical_flux_func=viscous_facial_flux_central,
                                           **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(discr.normal(dd_bdry))

        state_plus = self.adiabatic_wall_state_for_diffusion(
            discr=discr, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus)

        grad_cv_plus = self.grad_cv_bc(state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)
        grad_t_plus = self.grad_temperature_bc(grad_t_minus, normal)

        # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
        # *not* the numerical viscous flux as advised by [Bassi_1997]_.
        f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                             grad_t=grad_t_plus)

        return f_ext@normal

    def adiabatic_slip_grad_av(self, discr, dd_bdry, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary for artificial viscosity."""
        # Grab some boundary-relevant data
        dim, = grad_av_minus.mass.shape
        actx = grad_av_minus.mass[0].array_context
        nhat = actx.thaw(discr.normal(dd_bdry))

        # Subtract 2*wall-normal component of q
        # to enforce q=0 on the wall
        s_mom_normcomp = np.outer(nhat,
                                  np.dot(grad_av_minus.momentum, nhat))
        s_mom_flux = grad_av_minus.momentum - 2*s_mom_normcomp

        # flip components to set a neumann condition
        return make_conserved(dim, mass=-grad_av_minus.mass,
                              energy=-grad_av_minus.energy,
                              momentum=-s_mom_flux,
                              species_mass=-grad_av_minus.species_mass)
