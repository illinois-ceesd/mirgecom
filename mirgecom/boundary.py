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
    """

    @abstractmethod
    def inviscid_divergence_flux(self, discr, btag, eos, cv_minus, dv_minus,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""


class PrescribedFluidBoundary(FluidBoundary):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: inviscid_divergence_flux
    """

    def __init__(self,
                 # returns the flux to be used in div op (prescribed flux)
                 inviscid_boundary_flux_func=None,
                 # returns CV+, to be used in num flux func (prescribed soln)
                 boundary_state_func=None,
                 # Inviscid facial flux func given CV(+/-)
                 inviscid_facial_flux_func=None,
                 # Flux to be used in grad(Temperature) op
                 temperature_gradient_flux_func=None,
                 # Function returns boundary temperature_plus
                 boundary_temperature_func=None):
        """Initialize the PrescribedFluidBoundary and methods."""
        self._bnd_state_func = boundary_state_func
        self._inviscid_bnd_flux_func = inviscid_boundary_flux_func
        self._inviscid_div_flux_func = inviscid_facial_flux_func
        self._bnd_temperature_func = boundary_temperature_func

        if not self._inviscid_bnd_flux_func and not self._bnd_state_func:
            from warnings import warn
            warn("Using dummy boundary: copies interior solution.", stacklevel=2)

        if not self._inviscid_div_flux_func:
            self._inviscid_div_flux_func = inviscid_facial_divergence_flux
        if not self._bnd_state_func:
            self._bnd_state_func = self._dummy_state_func
        if not self._bnd_temperature_func:
            self._bnd_temperature_func = self._dummy_temperature_func

    def _dummy_temperature_func(self, temperature_minus, **kwargs):
        return -temperature_minus

    def _dummy_state_func(self, state_minus, **kwargs):
        return state_minus

    def _boundary_quantity(self, discr, btag, quantity, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        if "local" in kwargs:
            if kwargs["local"]:
                return quantity
        return discr.project(btag, "all_faces", quantity)

    def inviscid_divergence_flux(self, discr, btag, gas_model, state_minus,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        # This one is when the user specified a function that directly
        # prescribes the flux components at the boundary
        if self._inviscid_bnd_flux_func:
            return self._inviscid_bnd_flux_func(discr, btag, gas_model, state_minus,
                                                **kwargs)

        state_plus = self._bnd_state_func(discr=discr, btag=btag,
                                          gas_model=gas_model,
                                          state_minus=state_minus, **kwargs)
        boundary_state_pair = TracePair(btag, interior=state_minus,
                                        exterior=state_plus)

        return self._inviscid_div_flux_func(discr, state_tpair=boundary_state_pair)


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
    """

    def __init__(self):
        """Initialize AdiabaticSlipBoundary."""
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_slip_state
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
        t_seed = None if ext_cv.nspecies == 0 else state_minus.temperature

        from mirgecom.gas_model import make_fluid_state
        return make_fluid_state(cv=ext_cv, gas_model=gas_model,
                                temperature_seed=t_seed)


class AdiabaticNoslipMovingBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing a noslip moving boundary.

    .. automethod:: adiabatic_noslip_state
    .. automethod:: grad_cv_plus
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
        wall_pen = 2.0 * self._wall_velocity * state_minus.mass
        ext_mom = wall_pen - state_minus.momentum  # no-slip

        # Form the external boundary solution with the new momentum
        return make_conserved(dim=dim, mass=state_minus.mass,
                              energy=state_minus.energy, momentum=ext_mom,
                              species_mass=state_minus.species_mass)

    def grad_cv_plus(self, nodes, nhat, grad_cv_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        return(-grad_cv_minus)
