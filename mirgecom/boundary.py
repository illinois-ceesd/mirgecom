""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Treatment Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass FluidBoundary
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
from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import make_conserved
from grudge.trace_pair import TracePair
from mirgecom.inviscid import inviscid_flux_rusanov
from mirgecom.viscous import viscous_flux_central
from mirgecom.flux import (
    gradient_flux_central,
    divergence_flux_central
)
from mirgecom.gas_model import (
    make_fluid_state,
    project_fluid_state,
)
from abc import ABCMeta, abstractmethod


class FluidBoundary(metaclass=ABCMeta):
    r"""Abstract interface to fluid boundary treatment.

    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux
    """

    @abstractmethod
    def inviscid_divergence_flux(self, discr, btag, eos, cv_minus, dv_minus,
                                 numerical_flux_func, **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""

    @abstractmethod
    def viscous_divergence_flux(self, discr, btag, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus, **kwargs):
        """Get the viscous boundary flux for the divergence operator."""

    @abstractmethod
    def cv_gradient_flux(self, discr, btag, gas_model, state_minus, **kwargs):
        """Get the fluid soln boundary flux for the gradient operator."""

    @abstractmethod
    def temperature_gradient_flux(self, discr, btag, gas_model, state_minus,
                                  **kwargs):
        r"""Get temperature flux across the boundary faces."""


class PrescribedFluidBoundary(FluidBoundary):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: inviscid_divergence_flux
    .. automethod:: temperature_gradient_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: av_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: soln_gradient_flux
    """

    def __init__(self,
                 # returns the flux to be used in div op (prescribed flux)
                 inviscid_flux_func=None,
                 # returns CV+, to be used in num flux func (prescribed soln)
                 boundary_state_func=None,
                 # Inviscid facial flux func given CV(+/-)
                 # inviscid_numerical_flux_func=None,
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
        self._inviscid_flux_func = inviscid_flux_func
        self._temperature_grad_flux_func = temperature_gradient_flux_func
        self._bnd_temperature_func = boundary_temperature_func
        self._grad_num_flux_func = gradient_numerical_flux_func
        self._cv_gradient_flux_func = cv_gradient_flux_func
        self._viscous_flux_func = viscous_flux_func
        self._bnd_grad_cv_func = boundary_gradient_cv_func
        self._bnd_grad_temperature_func = boundary_gradient_temperature_func
        self._av_div_num_flux_func = divergence_flux_central
        self._bnd_grad_av_func = boundary_grad_av_func

        if not self._bnd_grad_av_func:
            self._bnd_grad_av_func = self._identical_grad_av

        if not self._inviscid_flux_func and not self._bnd_state_func:
            from warnings import warn
            warn("Using dummy boundary: copies interior solution.", stacklevel=2)

        if not self._inviscid_flux_func:
            self._inviscid_flux_func = self._inviscid_flux_for_prescribed_state
        # if not self._inviscid_num_flux_func:
        #     self._inviscid_num_flux_func = inviscid_facial_flux
        if not self._bnd_state_func:
            self._bnd_state_func = self._identical_state

        if not self._bnd_temperature_func:
            self._bnd_temperature_func = self._temperature_for_prescribed_state
        if not self._grad_num_flux_func:
            self._grad_num_flux_func = gradient_flux_central

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

    def _boundary_quantity(self, discr, btag, quantity, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        from grudge.dof_desc import as_dofdesc
        btag = as_dofdesc(btag)
        if "local" in kwargs:
            if kwargs["local"]:
                return quantity
        return discr.project(btag, btag.with_dtag("all_faces"), quantity)

    def _boundary_state_pair(self, discr, btag, gas_model, state_minus, **kwargs):
        return TracePair(btag,
                         interior=state_minus,
                         exterior=self._bnd_state_func(discr=discr, btag=btag,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus,
                                                       **kwargs))

    def _temperature_for_prescribed_state(self, discr, btag,
                                          gas_model, state_minus, **kwargs):
        boundary_state = self._bnd_state_func(discr=discr, btag=btag,
                                              gas_model=gas_model,
                                              state_minus=state_minus,
                                              **kwargs)
        return boundary_state.temperature

    def _temperature_for_interior_state(self, discr, btag, gas_model, state_minus,
                                        **kwargs):
        return state_minus.temperature

    def _identical_state(self, state_minus, **kwargs):
        return state_minus

    def _identical_grad_cv(self, grad_cv_minus, **kwargs):
        return grad_cv_minus

    def _identical_grad_temperature(self, grad_t_minus, **kwargs):
        return grad_t_minus

    def _gradient_flux_for_prescribed_cv(self, discr, btag, gas_model, state_minus,
                                         **kwargs):
        # Use prescribed external state and gradient numerical flux function
        boundary_state = self._bnd_state_func(discr=discr, btag=btag,
                                              gas_model=gas_model,
                                              state_minus=state_minus,
                                              **kwargs)
        cv_pair = TracePair(btag,
                            interior=state_minus.cv,
                            exterior=boundary_state.cv)

        actx = state_minus.array_context
        nhat = thaw(discr.normal(btag), actx)
        return self._boundary_quantity(
            discr, btag=btag,
            quantity=self._grad_num_flux_func(cv_pair, nhat), **kwargs)

    def _gradient_flux_for_prescribed_temperature(self, discr, btag, gas_model,
                                                  state_minus, **kwargs):
        # Feed a boundary temperature to numerical flux for grad op
        actx = state_minus.array_context
        nhat = thaw(discr.normal(btag), actx)
        bnd_tpair = TracePair(btag,
                              interior=state_minus.temperature,
                              exterior=self._bnd_temperature_func(
                                  discr=discr, btag=btag, gas_model=gas_model,
                                  state_minus=state_minus, **kwargs))
        return self._boundary_quantity(discr, btag,
                                       self._grad_num_flux_func(bnd_tpair, nhat),
                                       **kwargs)

    def _inviscid_flux_for_prescribed_state(
            self, discr, btag, gas_model, state_minus,
            numerical_flux_func=inviscid_flux_rusanov, **kwargs):
        # Use a prescribed boundary state and the numerical flux function
        boundary_state_pair = self._boundary_state_pair(discr=discr, btag=btag,
                                                        gas_model=gas_model,
                                                        state_minus=state_minus,
                                                        **kwargs)

        from mirgecom.inviscid import inviscid_facial_flux
        return self._boundary_quantity(
            discr, btag,
            inviscid_facial_flux(discr, gas_model=gas_model,
                                 state_pair=boundary_state_pair,
                                 numerical_flux_func=numerical_flux_func,
                                 local=True),
            **kwargs)

    def _viscous_flux_for_prescribed_state(self, discr, btag, gas_model, state_minus,
                                           grad_cv_minus, grad_t_minus,
                                           numerical_flux_func=viscous_flux_central,
                                           **kwargs):
        state_pair = self._boundary_state_pair(discr=discr, btag=btag,
                                               gas_model=gas_model,
                                               state_minus=state_minus, **kwargs)
        grad_cv_pair = \
            TracePair(btag, interior=grad_cv_minus,
                      exterior=self._bnd_grad_cv_func(
                          discr=discr, btag=btag, gas_model=gas_model,
                          state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                          grad_t_minus=grad_t_minus))

        grad_t_pair = \
            TracePair(
                btag, interior=grad_t_minus,
                exterior=self._bnd_grad_temperature_func(
                    discr=discr, btag=btag, gas_model=gas_model,
                    state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                    grad_t_minus=grad_t_minus))

        return self._boundary_quantity(
            discr, btag,
            quantity=numerical_flux_func(discr=discr, gas_model=gas_model,
                                         state_pair=state_pair,
                                         grad_cv_pair=grad_cv_pair,
                                         grad_t_pair=grad_t_pair))

    def inviscid_divergence_flux(
            self, discr, btag, gas_model, state_minus,
            numerical_flux_func=inviscid_flux_rusanov, **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        return self._inviscid_flux_func(discr, btag, gas_model, state_minus,
                                        numerical_flux_func=numerical_flux_func,
                                        **kwargs)

    def cv_gradient_flux(self, discr, btag, gas_model, state_minus, **kwargs):
        """Get the cv flux for *btag* for use in the gradient operator."""
        return self._cv_gradient_flux_func(
            discr=discr, btag=btag, gas_model=gas_model, state_minus=state_minus,
            **kwargs)

    def temperature_gradient_flux(self, discr, btag, gas_model, state_minus,
                                  **kwargs):
        """Get the "temperature flux" for *btag* for use in the gradient operator."""
        return self._temperature_grad_flux_func(discr, btag, gas_model, state_minus,
                                                **kwargs)

    def viscous_divergence_flux(self, discr, btag, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func=viscous_flux_central, **kwargs):
        """Get the viscous flux for *btag* for use in the divergence operator."""
        return self._viscous_flux_func(discr=discr, btag=btag, gas_model=gas_model,
                                       state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       grad_t_minus=grad_t_minus,
                                       numerical_flux_func=numerical_flux_func,
                                       **kwargs)

    # {{{ Boundary interface for artificial viscosity

    def _identical_grad_av(self, grad_av_minus, **kwargs):
        return grad_av_minus

    def soln_gradient_flux(self, discr, btag, fluid_state, gas_model, **kwargs):
        """Get the flux for solution gradient with AV API."""
        # project the conserved and thermal state to the boundary
        fluid_state_minus = project_fluid_state(discr=discr,
                                                src="vol",
                                                tgt=btag,
                                                gas_model=gas_model,
                                                state=fluid_state)
        # get the boundary flux for the grad(CV)
        return self.cv_gradient_flux(discr=discr, btag=btag,
                                     gas_model=gas_model,
                                     state_minus=fluid_state_minus,
                                     **kwargs)

    def av_flux(self, discr, btag, diffusion, **kwargs):
        """Get the diffusive fluxes for the AV operator API."""
        grad_av_minus = discr.project("vol", btag, diffusion)
        actx = grad_av_minus.mass[0].array_context
        nhat = thaw(discr.normal(btag), actx)
        grad_av_plus = self._bnd_grad_av_func(
            discr=discr, btag=btag, grad_av_minus=grad_av_minus, **kwargs)
        bnd_grad_pair = TracePair(btag, interior=grad_av_minus,
                                  exterior=grad_av_plus)
        num_flux = self._av_div_num_flux_func(bnd_grad_pair, nhat)
        return self._boundary_quantity(discr, btag, num_flux, **kwargs)

    # }}}


class DummyBoundary(PrescribedFluidBoundary):
    """Boundary type that assigns boundary-adjacent soln as the boundary solution."""

    def __init__(self):
        """Initialize the DummyBoundary boundary type."""
        PrescribedFluidBoundary.__init__(self)


class AdiabaticSlipBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing inviscid slip boundary.

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

    def adiabatic_slip_state(self, discr, btag, gas_model, state_minus, **kwargs):
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
        dim = discr.dim
        actx = state_minus.array_context

        # Grab a unit normal to the boundary
        nhat = thaw(discr.normal(btag), actx)

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

    def adiabatic_slip_grad_av(self, discr, btag, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary."""
        # Grab some boundary-relevant data
        dim, = grad_av_minus.mass.shape
        actx = grad_av_minus.mass[0].array_context
        nhat = thaw(discr.norm(btag), actx)

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
    r"""Boundary condition implementing a noslip moving boundary.

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

    def adiabatic_noslip_state(self, discr, btag, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
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
        """Get the exterior solution on the boundary."""
        return(-grad_av_minus)


class IsothermalNoSlipBoundary(PrescribedFluidBoundary):
    r"""Isothermal no-slip viscous wall boundary.

    This class implements an isothermal no-slip wall by:
    (TBD)
    [Hesthaven_2008]_, Section 6.6, and correspond to the characteristic
    boundary conditions described in detail in [Poinsot_1992]_.
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.isothermal_noslip_state,
            boundary_temperature_func=self.temperature_bc
        )

    def isothermal_noslip_state(self, discr, btag, gas_model, state_minus, **kwargs):
        """Get the interior and exterior solution (*state_minus*) on the boundary."""
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
        """Get temperature value to weakly prescribe wall bc."""
        return 2*self._wall_temp - state_minus.temperature


class FarfieldBoundary(PrescribedFluidBoundary):
    r"""Farfield boundary treatment.

    This class implements a farfield boundary as described by
    [Mengaldo_2014]_.  The boundary condition is implemented
    as:

    .. math::
        q_bc = q_\infty
    """

    def __init__(self, numdim, numspecies, free_stream_temperature=300,
                 free_stream_pressure=101325, free_stream_velocity=None,
                 free_stream_mass_fractions=None):
        """Initialize the boundary condition object."""
        if free_stream_velocity is None:
            free_stream_velocity = np.zeros(numdim)
        if len(free_stream_velocity) != numdim:
            raise ValueError("Free-stream velocity must be of ambient dimension.")
        if numspecies > 0:
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

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.farfield_state,
            boundary_temperature_func=self.temperature_bc
        )

    def farfield_state(self, discr, btag, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        free_stream_mass_fractions = (0*state_minus.species_mass_fractions
                                      + self._species_mass_fractions)
        free_stream_temperature = 0*state_minus.temperature + self._temperature
        free_stream_pressure = 0*state_minus.pressure + self._pressure
        free_stream_density = gas_model.eos.get_density(
            pressure=free_stream_pressure, temperature=free_stream_temperature,
            mass_fractions=free_stream_mass_fractions)
        free_stream_velocity = 0*state_minus.velocity + self._velocity
        free_stream_internal_energy = gas_model.eos.get_internal_energy(
            temperature=free_stream_temperature,
            mass_fractions=free_stream_mass_fractions)

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

    .. math:

        \rho^+ &= \rho^-
        \rho\mathbf{Y}^+ &= \rho\mathbf{Y}^-
        \rho\mathbf{V}^+ &= \rho^\mathbf{V}^-

    Total energy for the flow is computed as follows:


    When the flow is super-sonic, i.e. when:

    .. math:

       \rho\mathbf{V} \cdot \hat\mathbf{n} \ge c,

    then the internal solution is used outright:

    .. math:

        \rho{E}^+ &= \rho{E}^-

    otherwise the flow is sub-sonic, and the prescribed boundary pressure,
    $P^+$, is used to compute the energy:

    .. math:

        \rho{E}^+ &= \frac{\left(2~P^+ - P^-\right)}{\left(\gamma-1\right)}
        + \frac{1}{2\rho^+}\left(\rho\mathbf{V}^+\cdot\rho\mathbf{V}^+\right).
    """

    def __init__(self, boundary_pressure=101325):
        """Initialize the boundary condition object."""
        self._pressure = boundary_pressure
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.outflow_state
        )

    def outflow_state(self, discr, btag, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.
        """
        actx = state_minus.array_context
        nhat = thaw(discr.normal(btag), actx)
        # boundary-normal velocity
        boundary_vel = np.dot(state_minus.velocity, nhat)*nhat
        boundary_speed = actx.np.sqrt(np.dot(boundary_vel, boundary_vel))
        speed_of_sound = state_minus.speed_of_sound
        kinetic_energy = gas_model.eos.kinetic_energy(state_minus.cv)
        gamma = gas_model.eos.gamma(state_minus.cv, state_minus.temperature)
        external_pressure = 2*self._pressure - state_minus.pressure
        boundary_pressure = actx.np.where(actx.np.greater(boundary_speed,
                                                          speed_of_sound),
                                          state_minus.pressure, external_pressure)
        internal_energy = boundary_pressure / (gamma - 1)
        total_energy = internal_energy + kinetic_energy
        cv_outflow = make_conserved(dim=state_minus.dim, mass=state_minus.cv.mass,
                                    momentum=state_minus.cv.momentum,
                                    energy=total_energy,
                                    species_mass=state_minus.cv.species_mass)

        return make_fluid_state(cv=cv_outflow, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)


class InflowBoundary(PrescribedFluidBoundary):
    r"""Inflow boundary treatment.

    This class implements an inflow boundary as described by
    [Mengaldo_2014]_.
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

    def inflow_state(self, discr, btag, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.
        """
        actx = state_minus.array_context
        nhat = thaw(discr.normal(btag), actx)

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

        def temperature_bc(self, state_minus, **kwargs):
            """Temperature value that prescribes the desired temperature."""
            return -state_minus.temperature + 2.0*self._free_stream_temperature
