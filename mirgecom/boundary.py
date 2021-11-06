""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Treatment Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass FluidBoundary

Inviscid Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedFluidBoundary
.. autoclass:: DummyBoundary
.. autoclass:: AdiabaticSlipBoundary
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
                 # returns CV+, to be used in num flux func (prescribe soln)
                 boundary_cv_func=None,
                 # returns the DV+, to be used with CV+
                 boundary_dv_func=None,
                 # Numerical flux func given CV(+/-)
                 inviscid_facial_flux_func=None):
        """Initialize the PrescribedFluidBoundary and methods."""
        self._bnd_cv_func = boundary_cv_func
        self._bnd_dv_func = boundary_dv_func
        self._inviscid_bnd_flux_func = inviscid_boundary_flux_func
        self._inviscid_div_flux_func = inviscid_facial_flux_func

        if not self._inviscid_bnd_flux_func and not self._bnd_cv_func:
            from warnings import warn
            warn("Using dummy boundary: copies interior solution.", stacklevel=2)

        if not self._inviscid_div_flux_func:
            self._inviscid_div_flux_func = inviscid_facial_divergence_flux
        if not self._bnd_cv_func:
            self._bnd_cv_func = self._cv_func
        if not self._bnd_dv_func:
            self._bnd_dv_func = self._dv_func

    def _cv_func(self, discr, btag, eos, cv_minus, dv_minus):
        return cv_minus

    def _dv_func(self, discr, btag, eos, cv_pair, dv_minus):
        return eos.dependent_vars(cv_pair.ext, dv_minus.temperature)

    def inviscid_divergence_flux(self, discr, btag, eos, cv_minus, dv_minus,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        # This one is when the user specified a function that directly
        # prescribes the flux components at the boundary
        if self._inviscid_bnd_flux_func:
            return self._inviscid_bnd_flux_func(discr, btag, eos, cv_minus,
                                                dv_minus, **kwargs)

        # Otherwise, fall through to here, where the user specified instead
        # a function that provides CV+ and DV+.
        cv_pair = TracePair(
            btag, interior=cv_minus,
            exterior=self._bnd_cv_func(discr, btag, eos, cv_minus, dv_minus)
        )
        dv_pair = TracePair(
            btag, interior=dv_minus,
            exterior=self._bnd_dv_func(discr, btag, eos, cv_pair, dv_minus)
        )

        return self._inviscid_div_flux_func(discr, cv_tpair=cv_pair,
                                            dv_tpair=dv_pair)


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
            self, boundary_cv_func=self.adiabatic_slip_cv
        )

    def adiabatic_slip_cv(self, discr, btag, eos, cv_minus, dv_minus, **kwargs):
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
        actx = cv_minus.mass.array_context

        # Grab a unit normal to the boundary
        nhat = thaw(actx, discr.normal(btag))

        # Subtract out the 2*wall-normal component
        # of velocity from the velocity at the wall to
        # induce an equal but opposite wall-normal (reflected) wave
        # preserving the tangential component
        mom_normcomp = np.dot(cv_minus.momentum, nhat)  # wall-normal component
        wnorm_mom = nhat * mom_normcomp  # wall-normal mom vec
        ext_mom = cv_minus.momentum - 2.0 * wnorm_mom  # prescribed ext momentum

        # Form the external boundary solution with the new momentum
        return make_conserved(dim=dim, mass=cv_minus.mass, energy=cv_minus.energy,
                              momentum=ext_mom, species_mass=cv_minus.species_mass)
