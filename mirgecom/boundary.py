""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Treatment Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass FluidBoundary
.. autoclass FluidBC

Inviscid Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedInviscidBoundary
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
from mirgecom.inviscid import inviscid_facial_flux

from abc import ABCMeta, abstractmethod


class FluidBoundary(metaclass=ABCMeta):
    r"""Abstract interface to fluid boundary treatment.

    .. automethod:: inviscid_boundary_flux
    """

    @abstractmethod
    def inviscid_boundary_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the inviscid flux across the boundary faces."""


class FluidBC(FluidBoundary):
    r"""Abstract interface to boundary conditions.

    .. automethod:: inviscid_boundary_flux
    .. automethod:: boundary_pair
    """

    @abstractmethod
    def inviscid_boundary_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the inviscid part of the physical flux across the boundary *btag*."""

    @abstractmethod
    def boundary_pair(self, discr, btag, cv, eos, **kwargs):
        """Get the interior and exterior solution (*u*) on the boundary."""


class PrescribedInviscidBoundary(FluidBC):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: boundary_pair
    .. automethod:: inviscid_boundary_flux
    """

    def __init__(self, inviscid_boundary_flux_func=None, boundary_pair_func=None,
                 inviscid_facial_flux_func=None, fluid_solution_func=None,
                 fluid_solution_flux_func=None):
        """Initialize the PrescribedInviscidBoundary and methods."""
        self._bnd_pair_func = boundary_pair_func
        self._inviscid_bnd_flux_func = inviscid_boundary_flux_func
        self._inviscid_facial_flux_func = inviscid_facial_flux_func
        if not self._inviscid_facial_flux_func:
            self._inviscid_facial_flux_func = inviscid_facial_flux
        self._fluid_soln_func = fluid_solution_func
        self._fluid_soln_flux_func = fluid_solution_flux_func

    def boundary_pair(self, discr, btag, cv, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        if self._bnd_pair_func:
            return self._bnd_pair_func(discr, cv=cv, btag=btag, **kwargs)
        if not self._fluid_soln_func:
            raise NotImplementedError()
        actx = cv.array_context
        boundary_discr = discr.discr_from_dd(btag)
        nodes = thaw(actx, boundary_discr.nodes())
        nhat = thaw(actx, discr.normal(btag))
        int_soln = discr.project("vol", btag, cv)
        ext_soln = self._fluid_soln_func(nodes, cv=int_soln, normal=nhat, **kwargs)
        return TracePair(btag, interior=int_soln, exterior=ext_soln)

    def inviscid_boundary_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the inviscid flux across the boundary faces."""
        if self._inviscid_bnd_flux_func:
            actx = cv.array_context
            boundary_discr = discr.discr_from_dd(btag)
            nodes = thaw(actx, boundary_discr.nodes())
            nhat = thaw(actx, discr.normal(btag))
            int_soln = discr.project("vol", btag, cv)
            return self._inviscid_bnd_flux_func(nodes, normal=nhat,
                                                cv=int_soln, eos=eos, **kwargs)
        bnd_tpair = self.boundary_pair(discr, btag=btag, cv=cv, eos=eos, **kwargs)
        return self._inviscid_facial_flux_func(discr, eos=eos, cv_tpair=bnd_tpair)


class PrescribedBoundary(PrescribedInviscidBoundary):
    """Boundary condition prescribes boundary soln with user-specified function.

    .. automethod:: __init__
    """

    def __init__(self, userfunc):
        """Set the boundary function.

        Parameters
        ----------
        userfunc
            User function that prescribes the solution values on the exterior
            of the boundary. The given user function (*userfunc*) must take at
            least one parameter that specifies the coordinates at which to prescribe
            the solution.
        """
        from warnings import warn
        warn("Do not use PrescribedBoundary; use PrescribedInvscidBoundary. This"
             "boundary type will vanish by August 2021.", DeprecationWarning,
             stacklevel=2)
        PrescribedInviscidBoundary.__init__(self, fluid_solution_func=userfunc)


class DummyBoundary(PrescribedInviscidBoundary):
    """Boundary condition that assigns boundary-adjacent soln as the boundary solution.

    .. automethod:: dummy_pair
    """

    def __init__(self):
        """Initialize the DummyBoundary boundary type."""
        PrescribedInviscidBoundary.__init__(self, boundary_pair_func=self.dummy_pair)

    def dummy_pair(self, discr, cv, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        dir_soln = self.exterior_q(discr, cv, btag, **kwargs)
        return TracePair(btag, interior=dir_soln, exterior=dir_soln)

    def exterior_q(self, discr, cv, btag, **kwargs):
        """Get the exterior solution on the boundary."""
        return discr.project("vol", btag, cv)


class AdiabaticSlipBoundary(PrescribedInviscidBoundary):
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

    .. automethod:: adiabatic_slip_pair
    """

    def __init__(self):
        """Initialize AdiabaticSlipBoundary."""
        PrescribedInviscidBoundary.__init__(
            self, boundary_pair_func=self.adiabatic_slip_pair
        )

    def adiabatic_slip_pair(self, discr, cv, btag, **kwargs):
        """Get the interior and exterior solution on the boundary.

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
        actx = cv.mass.array_context

        # Grab a unit normal to the boundary
        nhat = thaw(actx, discr.normal(btag))

        # Get the interior/exterior solns
        int_cv = discr.project("vol", btag, cv)

        # Subtract out the 2*wall-normal component
        # of velocity from the velocity at the wall to
        # induce an equal but opposite wall-normal (reflected) wave
        # preserving the tangential component
        mom_normcomp = np.dot(int_cv.momentum, nhat)  # wall-normal component
        wnorm_mom = nhat * mom_normcomp  # wall-normal mom vec
        ext_mom = int_cv.momentum - 2.0 * wnorm_mom  # prescribed ext momentum

        # Form the external boundary solution with the new momentum
        ext_cv = make_conserved(dim=dim, mass=int_cv.mass, energy=int_cv.energy,
                                momentum=ext_mom, species_mass=int_cv.species_mass)
        return TracePair(btag, interior=int_cv, exterior=ext_cv)
