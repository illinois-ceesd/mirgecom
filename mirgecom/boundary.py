""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Treatment Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass FluidBoundary

Inviscid Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedFluidBoundary
.. autoclass:: DummyBoundary
.. autoclass:: AdiabaticSlipBoundary
.. autoclass:: AdiabaticNoslipMovingBoundary

Viscous Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: IsothermalNoSlipBoundary
.. autoclass:: PrescribedViscousBoundary
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
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import make_conserved
from grudge.trace_pair import TracePair
from mirgecom.inviscid import inviscid_facial_divergence_flux

from abc import ABCMeta, abstractmethod


class FluidBoundary(metaclass=ABCMeta):
    r"""Abstract interface to fluid boundary treatment.

    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: t_gradient_flux
    """

    @abstractmethod
    def inviscid_divergence_flux(self, discr, btag, eos, cv_minus, dv_minus,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""

    @abstractmethod
    def viscous_divergence_flux(self, discr, btag, cv, grad_cv, grad_t,
                              eos, **kwargs):
        """Get the viscous boundary flux for the divergence operator."""

    @abstractmethod
    def cv_gradient_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the fluid soln boundary flux for the gradient operator."""

    @abstractmethod
    def temperature_gradient_flux(self, discr, btag, cv, eos, **kwargs):
        r"""Get temperature flux across the boundary faces."""


class PrescribedFluidBoundary(FluidBoundary):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: inviscid_divergence_flux
    .. automethod:: soln_gradient_flux
    """

    def __init__(self,
                 # returns the flux to be used in div op (prescribed flux)
                 inviscid_boundary_flux_func=None,
                 # returns CV+, to be used in num flux func (prescribed soln)
                 boundary_state_func=None,
                 # returns the DV+, to be used with CV+
                 boundary_dv_func=None,
                 # Inviscid facial flux func given CV(+/-)
                 inviscid_facial_flux_func=None,
                 # Flux to be used in grad(Temperature) op
                 temperature_gradient_flux_func=None,
                 # Function returns boundary temperature_plus
                 boundary_temperature_func=None):
        """Initialize the PrescribedFluidBoundary and methods."""
        self._bnd_state_func = boundary_state_func
        self._bnd_dv_func = boundary_dv_func
        self._inviscid_bnd_flux_func = inviscid_boundary_flux_func
        self._inviscid_div_flux_func = inviscid_facial_flux_func
        self._temperature_grad_flux_func = temperature_gradient_flux_func
        self._bnd_temperature_func = boundary_temperature_func

        if not self._inviscid_bnd_flux_func and not self._bnd_state_func:
            from warnings import warn
            warn("Using dummy boundary: copies interior solution.", stacklevel=2)

        if not self._inviscid_div_flux_func:
            self._inviscid_div_flux_func = inviscid_facial_divergence_flux
        if not self._bnd_state_func:
            self._bnd_state_func = self._dummy_state_func
        # if not self._bnd_dv_func:
        #     self._bnd_dv_func = self._dummy_dv_func
        if not self._bnd_temperature_func:
            self._bnd_temperature_func = self._dummy_temperature_func

    def _dummy_temperature_func(self, temperature_minus, **kwargs):
        return -temperature_minus

    def _dummy_state_func(self, state_minus, **kwargs):
        return state_minus

    # def _dummy_dv_func(self, eos, cv_pair, **kwargs):
    #     dv_minus = eos.dependent_vars(cv_pair.int)
    #     return eos.dependent_vars(cv_pair.ext, dv_minus.temperature)

    def _boundary_quantity(self, discr, btag, quantity, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        if "local" in kwargs:
            if kwargs["local"]:
                return quantity
        return discr.project(btag, "all_faces", quantity)

    def inviscid_divergence_flux(self, discr, btag, eos, state_minus, **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        # This one is when the user specified a function that directly
        # prescribes the flux components at the boundary
        if self._inviscid_bnd_flux_func:
            return self._inviscid_bnd_flux_func(discr, btag, eos, state_minus,
                                                **kwargs)

        state_plus = self._bnd_state_func(discr=discr, btag=btag, eos=eos,
                                          state_minus=state_minus, **kwargs)
        boundary_state_pair = TracePair(btag, interior=state_minus,
                                        exterior=state_plus)

        return self._inviscid_div_flux_func(discr, state_tpair=boundary_state_pair)

    def cv_gradient_flux(self, discr, btag, eos, cv_minus, **kwargs):
        """Get the flux through boundary *btag* for each scalar in *q*."""
        # If specified, use the function that *prescribes* cv grad flux
        if self._cv_gradient_flux_func:
            return self._cv_gradient_flux_func(discr, btag, cv_minus=cv_minus,
                                               **kwargs)
        actx = cv_minus.array_context
        # Otherwise calculate the flux using the CV+ and numerical flux function
        cv_pair = TracePair(
            btag, interior=cv_minus,
            exterior=self._bnd_state_func(discr, btag, eos, cv_minus, **kwargs)
        )
        dv_plus = self._bnd_dv_func(discr, btag, eos, cv_pair, **kwargs) # noqa
        nhat = thaw(discr.normal(btag), actx)
        return self._boundary_quantity(
            discr, btag=btag,
            quantity=self._scalar_gradient_flux_func(cv_pair, normal=nhat),
            **kwargs
        )

    # def soln_gradient_flux(self, discr, btag, soln, **kwargs):
    #     """Get the flux for solution gradient with AV API."""
    #     cv = make_conserved(discr.dim, q=soln)
    #     return self.cv_gradient_flux(discr, btag, cv, **kwargs).join()

    def temperature_gradient_flux(self, discr, btag, eos, cv_minus, **kwargs):
        """Get the "temperature flux" through boundary *btag*."""
        # Use prescriptive flux function if it exists
        if self._temperature_gradient_flux_func:
            return self._boundary_quantity(
                discr, btag,
                self._temperature_gradient_flux_func(discr, btag, eos, cv_minus,
                                                     **kwargs),
                **kwargs
            )

        # Otherwise feed a boundary temperature to numerical flux for grad op
        t_minus = eos.temperature(cv_minus)
        actx = cv_minus.array_context
        if self._bnd_temperature_func:  # use boundary temperature func if supplied
            t_plus = self._bnd_temperature_func(discr, btag, eos, cv_minus=cv_minus,
                                                **kwargs)
        else:  # or - last resort; make zero temperature flux for gradient operator
            t_plus = -t_minus  # 0's out the num flux

        # Now pass the temperature pair to the numerical flux func for grad op
        nhat = thaw(actx, discr.normal(btag))
        bnd_tpair = TracePair(btag, interior=t_minus, exterior=t_plus)
        return self._boundary_quantity(discr, btag,
                                       self._scalar_num_flux_func(bnd_tpair, nhat),
                                       **kwargs)

    def viscous_divergence_flux(self, discr, btag, eos, cv_minus, grad_cv_minus,
                                grad_t_minus, **kwargs):
        """Get the viscous part of the physical flux across the boundary *btag*."""
        cv_tpair = TracePair(btag,
                             interior=cv_minus,
                             exterior=self._bnd_state_func(discr, btag, eos, cv_minus,
                                                        **kwargs))
        dv_plus = self._bnd_dv_func(discr, btag, eos, cv_tpair, **kwargs) # noqa

        grad_cv_minus = discr.project("vol", btag, grad_cv_minus)
        grad_cv_tpair = TracePair(btag, interior=grad_cv_minus,
                                  exterior=grad_cv_minus)

        grad_t_minus = discr.project("vol", btag, grad_t_minus)
        grad_t_tpair = TracePair(btag, interior=grad_t_minus, exterior=grad_t_minus)

        from mirgecom.viscous import viscous_facial_flux
        return viscous_facial_flux(discr, eos, cv_tpair, grad_cv_tpair, grad_t_tpair)


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

    .. automethod:: adiabatic_slip_cv
    """

    def __init__(self):
        """Initialize AdiabaticSlipBoundary."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_slip_state
        )

    def adiabatic_slip_state(self, discr, btag, eos, state_minus, **kwargs):
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
        nhat = thaw(actx, discr.normal(btag))

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
        t_seed = None if ext_cv.nspecies == 0 else state_minus.dv.temperature

        from mirgecom.gas_model import make_fluid_state
        return make_fluid_state(cv=ext_cv, eos=eos, temperature_seed=t_seed)


class AdiabaticNoslipMovingBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing a noslip moving boundary.

    .. automethod:: adiabatic_noslip_pair
    .. automethod:: exterior_soln
    .. automethod:: exterior_grad_q
    """

    def __init__(self, wall_velocity=None, dim=2):
        """Initialize boundary device."""
        PrescribedFluidBoundary.__init__(
            self, boundary_cv_func=self.cv_plus,
            fluid_solution_gradient_func=self.grad_cv_plus
        )
        # Check wall_velocity (assumes dim is correct)
        if wall_velocity is None:
            wall_velocity = np.zeros(shape=(dim,))
        if len(wall_velocity) != dim:
            raise ValueError(f"Specified wall velocity must be {dim}-vector.")
        self._wall_velocity = wall_velocity

    def adiabatic_noslip_state(self, discr, btag, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        dim = discr.dim

        # Compute momentum solution
        wall_pen = 2.0 * self._wall_velocity * cv_minus.mass
        ext_mom = wall_pen - cv_minus.momentum  # no-slip

        # Form the external boundary solution with the new momentum
        return make_conserved(dim=dim, mass=cv_minus.mass, energy=cv_minus.energy,
                              momentum=ext_mom, species_mass=cv_minus.species_mass)

    def grad_cv_plus(self, nodes, nhat, grad_cv_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        return(-grad_cv_minus)


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
            self, boundary_cv_func=self.isothermal_noslip_cv,
            fluid_temperature_func=self.temperature_bc
        )

    def isothermal_noslip_cv(self, discr, btag, eos, cv_minus, **kwargs):
        """Get the interior and exterior solution (*cv*) on the boundary."""

        temperature_wall = self._wall_temp + 0*cv_minus.mass
        velocity_plus = -cv_minus.velocity
        mass_frac_plus = -cv_minus.species_mass_fractions

        internal_energy_plus = eos.get_internal_energy(
            temperature=temperature_wall, species_mass_fractions=mass_frac_plus,
            mass=cv_minus.mass
        )
        total_energy_plus = cv_minus.mass*(internal_energy_plus
                                           + .5*np.dot(velocity_plus, velocity_plus))

        cv_plus = make_conserved(
            discr.dim, mass=cv_minus.mass, energy=total_energy_plus,
            momentum=-cv_minus.momentum, species_mass=cv_minus.species_mass
        )

        return TracePair(btag, interior=cv_minus, exterior=cv_plus)

    def temperature_bc(self, nodes, cv, temperature, eos, **kwargs):
        """Get temperature value to weakly prescribe wall bc."""
        return 2*self._wall_temp - temperature
