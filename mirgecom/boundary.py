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
from mirgecom.inviscid import inviscid_facial_flux

from abc import ABCMeta, abstractmethod


class FluidBoundary(metaclass=ABCMeta):
    r"""Abstract interface to fluid boundary treatment.

    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: t_gradient_flux
    """

    @abstractmethod
    def inviscid_divergence_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""

    @abstractmethod
    def viscous_divergence_flux(self, discr, btag, cv, grad_cv, grad_t,
                              eos, **kwargs):
        """Get the viscous boundary flux for the divergence operator."""

    @abstractmethod
    def cv_gradient_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the fluid soln boundary flux for the gradient operator."""

    @abstractmethod
    def t_gradient_flux(self, discr, btag, cv, eos, **kwargs):
        r"""Get temperature flux across the boundary faces."""


class FluidBC(FluidBoundary):
    r"""Abstract interface to viscous boundary conditions.

    .. automethod:: cv_gradient_flux
    .. automethod:: t_gradient_flux
    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: boundary_pair
    """

    def cv_gradient_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the flux through boundary *btag* for each scalar in *q*."""
        raise NotImplementedError()

    def t_gradient_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the "temperature flux" through boundary *btag*."""
        raise NotImplementedError()

    def inviscid_divergence_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        raise NotImplementedError()

    def viscous_divergence_flux(self, discr, btag, cv, grad_cv, grad_t, eos,
                                **kwargs):
        """Get the viscous part of the physical flux across the boundary *btag*."""
        raise NotImplementedError()

    def boundary_pair(self, discr, btag, cv, eos, **kwargs):
        """Get the interior and exterior solution (*u*) on the boundary."""
        raise NotImplementedError()


class PrescribedInviscidBoundary(FluidBC):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: boundary_pair
    .. automethod:: inviscid_divergence_flux
    .. automethod:: soln_gradient_flux
    .. automethod:: av_flux
    """

    def __init__(self, inviscid_divergence_flux_func=None, boundary_pair_func=None,
                 inviscid_facial_flux_func=None, fluid_solution_func=None,
                 fluid_solution_flux_func=None, scalar_numerical_flux_func=None,
                 fluid_solution_gradient_func=None,
                 fluid_solution_gradient_flux_func=None,
                 fluid_temperature_func=None):
        """Initialize the PrescribedInviscidBoundary and methods."""
        self._bnd_pair_func = boundary_pair_func
        self._inviscid_bnd_flux_func = inviscid_divergence_flux_func
        self._inviscid_facial_flux_func = inviscid_facial_flux_func
        if not self._inviscid_facial_flux_func:
            self._inviscid_facial_flux_func = inviscid_facial_flux
        self._fluid_soln_func = fluid_solution_func
        self._fluid_soln_flux_func = fluid_solution_flux_func
        self._scalar_num_flux_func = scalar_numerical_flux_func
        from mirgecom.flux import gradient_flux_central
        if not self._scalar_num_flux_func:
            self._scalar_num_flux_func = gradient_flux_central
        self._fluid_soln_grad_func = fluid_solution_gradient_func
        self._fluid_soln_grad_flux_func = fluid_solution_gradient_flux_func
        from mirgecom.flux import divergence_flux_central
        if not self._fluid_soln_grad_flux_func:
            self._fluid_soln_grad_flux_func = divergence_flux_central
        self._fluid_temperature_func = fluid_temperature_func

    def _boundary_quantity(self, discr, btag, quantity, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        if "local" in kwargs:
            if kwargs["local"]:
                return quantity
        return discr.project(btag, "all_faces", quantity)

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

    def inviscid_divergence_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
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

    def cv_gradient_flux(self, discr, btag, cv, **kwargs):
        """Get the flux through boundary *btag* for each scalar in *q*."""
        actx = cv.array_context
        boundary_discr = discr.discr_from_dd(btag)
        nodes = thaw(actx, boundary_discr.nodes())
        nhat = thaw(actx, discr.normal(btag))
        if self._fluid_soln_flux_func:
            cv_minus = discr.project("vol", btag, cv)
            flux_weak = self._fluid_soln_flux_func(nodes, cv=cv_minus, nhat=nhat,
                                                   **kwargs)
        else:
            bnd_pair = self.boundary_pair(discr, btag=btag, cv=cv, **kwargs)
            flux_weak = self._scalar_num_flux_func(bnd_pair, normal=nhat)

        return self._boundary_quantity(discr, btag=btag, quantity=flux_weak,
                                       **kwargs)

    def soln_gradient_flux(self, discr, btag, soln, **kwargs):
        """Get the flux for solution gradient with AV API."""
        cv = make_conserved(discr.dim, q=soln)
        return self.cv_gradient_flux(discr, btag, cv, **kwargs).join()

    def s_boundary_flux(self, discr, btag, grad_cv, **kwargs):
        r"""Get $\nabla\mathbf{Q}$ flux across the boundary faces."""
        actx = grad_cv.mass[0].array_context
        boundary_discr = discr.discr_from_dd(btag)
        nodes = thaw(actx, boundary_discr.nodes())
        nhat = thaw(actx, discr.normal(btag))
        grad_cv_minus = discr.project("vol", btag, grad_cv)
        if self._fluid_soln_grad_func:
            grad_cv_plus = \
                self._fluid_soln_grad_func(nodes, nhat=nhat,
                                           grad_cv=grad_cv_minus, **kwargs)
        else:
            grad_cv_plus = grad_cv_minus
        bnd_grad_pair = TracePair(btag, interior=grad_cv_minus,
                                  exterior=grad_cv_plus)

        return self._boundary_quantity(
            discr, btag, self._fluid_soln_grad_flux_func(bnd_grad_pair, nhat),
            **kwargs
        )

    def av_flux(self, discr, btag, diffusion, **kwargs):
        """Get the diffusive fluxes for the AV operator API."""
        diff_cv = make_conserved(discr.dim, q=diffusion)
        return self.s_boundary_flux(discr, btag, grad_cv=diff_cv, **kwargs).join()

    def t_gradient_flux(self, discr, btag, cv, eos, **kwargs):
        """Get the "temperature flux" through boundary *btag*."""
        cv_minus = discr.project("vol", btag, cv)
        t_minus = eos.temperature(cv_minus)
        actx = cv.array_context
        if self._fluid_temperature_func:
            boundary_discr = discr.discr_from_dd(btag)
            nodes = thaw(actx, boundary_discr.nodes())
            t_plus = self._fluid_temperature_func(nodes, cv=cv_minus,
                                                  temperature=t_minus, eos=eos,
                                                  **kwargs)
        else:
            t_plus = -t_minus
        nhat = thaw(actx, discr.normal(btag))
        bnd_tpair = TracePair(btag, interior=t_minus, exterior=t_plus)

        return self._boundary_quantity(discr, btag,
                                       self._scalar_num_flux_func(bnd_tpair, nhat),
                                       **kwargs)

    def viscous_divergence_flux(self, discr, btag, eos, cv, grad_cv, grad_t,
                                **kwargs):
        """Get the viscous part of the physical flux across the boundary *btag*."""
        cv_tpair = self.boundary_pair(discr, btag=btag, cv=cv, eos=eos, **kwargs)

        grad_cv_minus = discr.project("vol", btag, grad_cv)
        grad_cv_tpair = TracePair(btag, interior=grad_cv_minus,
                                  exterior=grad_cv_minus)

        grad_t_minus = discr.project("vol", btag, grad_t)
        grad_t_tpair = TracePair(btag, interior=grad_t_minus, exterior=grad_t_minus)

        from mirgecom.viscous import viscous_facial_flux
        return viscous_facial_flux(discr, eos, cv_tpair, grad_cv_tpair, grad_t_tpair)


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
            self, boundary_pair_func=self.adiabatic_slip_pair,
            fluid_solution_gradient_func=self.exterior_grad_q
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

    def exterior_grad_q(self, nodes, nhat, grad_cv, **kwargs):
        """Get the exterior grad(Q) on the boundary."""
        # Grab some boundary-relevant data
        dim, = grad_cv.mass.shape

        # Subtract 2*wall-normal component of q
        # to enforce q=0 on the wall
        s_mom_normcomp = np.outer(nhat, np.dot(grad_cv.momentum, nhat))
        s_mom_flux = grad_cv.momentum - 2*s_mom_normcomp

        # flip components to set a neumann condition
        return make_conserved(dim, mass=-grad_cv.mass, energy=-grad_cv.energy,
                              momentum=-s_mom_flux,
                              species_mass=-grad_cv.species_mass)


class AdiabaticNoslipMovingBoundary(PrescribedInviscidBoundary):
    r"""Boundary condition implementing a noslip moving boundary.

    .. automethod:: adiabatic_noslip_pair
    .. automethod:: exterior_soln
    .. automethod:: exterior_grad_q
    """

    def __init__(self, wall_velocity=None, dim=2):
        """Initialize boundary device."""
        PrescribedInviscidBoundary.__init__(
            self, boundary_pair_func=self.adiabatic_noslip_pair,
            fluid_solution_gradient_func=self.exterior_grad_q
        )
        # Check wall_velocity (assumes dim is correct)
        if wall_velocity is None:
            wall_velocity = np.zeros(shape=(dim,))
        if len(wall_velocity) != dim:
            raise ValueError(f"Specified wall velocity must be {dim}-vector.")
        self._wall_velocity = wall_velocity

    def adiabatic_noslip_pair(self, discr, cv, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        bndry_soln = self.exterior_soln(discr, cv, btag, **kwargs)
        int_soln = discr.project("vol", btag, cv)

        return TracePair(btag, interior=int_soln, exterior=bndry_soln)

    def exterior_soln(self, discr, cv, btag, **kwargs):
        """Get the exterior solution on the boundary."""
        dim = discr.dim

        # Get the interior/exterior solns
        int_cv = discr.project("vol", btag, cv)

        # Compute momentum solution
        wall_pen = 2.0 * self._wall_velocity * int_cv.mass
        ext_mom = wall_pen - int_cv.momentum  # no-slip

        # Form the external boundary solution with the new momentum
        return make_conserved(dim=dim, mass=int_cv.mass, energy=int_cv.energy,
                              momentum=ext_mom, species_mass=int_cv.species_mass)

    def exterior_grad_q(self, nodes, nhat, grad_cv, **kwargs):
        """Get the exterior solution on the boundary."""
        return(-grad_cv)


class IsothermalNoSlipBoundary(PrescribedInviscidBoundary):
    r"""Isothermal no-slip viscous wall boundary.

    This class implements an isothermal no-slip wall by:
    (TBD)
    [Hesthaven_2008]_, Section 6.6, and correspond to the characteristic
    boundary conditions described in detail in [Poinsot_1992]_.
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature
        PrescribedInviscidBoundary.__init__(
            self, boundary_pair_func=self.isothermal_noslip_pair,
            fluid_temperature_func=self.temperature_bc
        )

    def isothermal_noslip_pair(self, discr, btag, eos, cv, **kwargs):
        """Get the interior and exterior solution (*cv*) on the boundary."""
        cv_minus = discr.project("vol", btag, cv)

        t_plus = self._wall_temp + 0*cv_minus.mass
        velocity_plus = -cv_minus.momentum / cv_minus.mass
        mass_frac_plus = cv_minus.species_mass / cv_minus.mass

        internal_energy_plus = eos.get_internal_energy(
            temperature=t_plus, species_mass_fractions=mass_frac_plus
        )
        total_energy_plus = cv_minus.mass*(internal_energy_plus
                             + .5**np.dot(velocity_plus, velocity_plus))

        cv_plus = make_conserved(
            discr.dim, mass=cv_minus.mass, energy=total_energy_plus,
            momentum=-cv_minus.momentum, species_mass=cv_minus.species_mass
        )

        return TracePair(btag, interior=cv_minus, exterior=cv_plus)

    def temperature_bc(self, nodes, cv, temperature, eos, **kwargs):
        """Get temperature value to weakly prescribe wall bc."""
        return self._wall_temp + 0*temperature


class PrescribedViscousBoundary(FluidBC):
    r"""Fully prescribed boundary for viscous flows.

    This class implements an inflow/outflow by:
    (TBD)
    [Hesthaven_2008]_, Section 6.6, and correspond to the characteristic
    boundary conditions described in detail in [Poinsot_1992]_.
    """

    def __init__(self, q_func=None, grad_q_func=None, t_func=None,
                 grad_t_func=None, inviscid_flux_func=None,
                 viscous_flux_func=None, t_flux_func=None,
                 q_flux_func=None):
        """Initialize the boundary condition object."""
        self._q_func = q_func
        self._q_flux_func = q_flux_func
        self._grad_q_func = grad_q_func
        self._t_func = t_func
        self._t_flux_func = t_flux_func
        self._grad_t_func = grad_t_func
        self._inviscid_flux_func = inviscid_flux_func
        self._viscous_flux_func = viscous_flux_func

    def _boundary_quantity(self, discr, btag, quantity, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        if "local" in kwargs:
            if kwargs["local"]:
                return quantity
        return discr.project(btag, "all_faces", quantity)

    def cv_gradient_flux(self, discr, btag, eos, cv, **kwargs):
        """Get the flux through boundary *btag* for each scalar in *q*."""
        actx = cv.array_context
        boundary_discr = discr.discr_from_dd(btag)
        cv_minus = discr.project("vol", btag, cv)
        nodes = thaw(actx, boundary_discr.nodes())
        nhat = thaw(actx, discr.normal(btag))

        flux_weak = 0
        if self._q_flux_func:
            flux_weak = self._q_flux_func(nodes, eos, cv_minus, nhat, **kwargs)
        elif self._q_func:
            cv_plus = self._q_func(nodes, eos=eos, cv=cv_minus, **kwargs)
        else:
            cv_plus = cv_minus

        cv_tpair = TracePair(btag, interior=cv_minus, exterior=cv_plus)

        from mirgecom.flux import gradient_flux_central
        flux_func = gradient_flux_central
        if "numerical_flux_func" in kwargs:
            flux_func = kwargs["numerical_flux_func"]

        flux_weak = flux_func(cv_tpair, nhat)

        return self._boundary_quantity(discr, btag, flux_weak, **kwargs)

    def t_gradient_flux(self, discr, btag, eos, cv, **kwargs):
        """Get the "temperature flux" through boundary *btag*."""
        actx = cv.array_context
        boundary_discr = discr.discr_from_dd(btag)
        cv_minus = discr.project("vol", btag, cv)
        nodes = thaw(actx, boundary_discr.nodes())
        nhat = thaw(actx, discr.normal(btag))

        if self._t_flux_func:
            flux_weak = self._t_flux_func(nodes, eos, cv=cv_minus, nhat=nhat,
                                          **kwargs)
        else:
            t_minus = eos.temperature(cv_minus)
            if self._t_func:
                t_plus = self._t_func(nodes, eos, cv=cv_minus, **kwargs)
            elif self._q_func:
                cv_plus = self._q_func(nodes, eos=eos, cv=cv_minus, **kwargs)
                t_plus = eos.temperature(cv_plus)
            else:
                t_plus = t_minus

            bnd_tpair = TracePair(btag, interior=t_minus, exterior=t_plus)

            from mirgecom.flux import gradient_flux_central
            flux_func = gradient_flux_central
            if "numerical_flux_func" in kwargs:
                flux_func = kwargs["numerical_flux_func"]

            flux_weak = flux_func(bnd_tpair, nhat)

        return self._boundary_quantity(discr, btag, flux_weak, **kwargs)

    def inviscid_divergence_flux(self, discr, btag, eos, cv, **kwargs):
        """Get the inviscid part of the physical flux across the boundary *btag*."""
        actx = cv.array_context
        boundary_discr = discr.discr_from_dd(btag)
        cv_minus = discr.project("vol", btag, cv)
        nodes = thaw(actx, boundary_discr.nodes())
        nhat = thaw(actx, discr.normal(btag))

        flux_weak = 0
        if self._inviscid_flux_func:
            flux_weak = self._inviscid_flux_func(nodes, eos, cv=cv_minus,
                                                 nhat=nhat, **kwargs)
        else:
            if self._q_func:
                cv_plus = self._q_func(nodes, eos=eos, cv=cv_minus, **kwargs)
            else:
                cv_plus = cv_minus

            bnd_tpair = TracePair(btag, interior=cv_minus, exterior=cv_plus)
            from mirgecom.inviscid import inviscid_facial_flux
            return inviscid_facial_flux(discr, eos, bnd_tpair)

        return self._boundary_quantity(discr, btag, flux_weak, **kwargs)

    def viscous_divergence_flux(self, discr, btag, eos, cv, grad_cv, grad_t,
                                **kwargs):
        """Get the viscous part of the physical flux across the boundary *btag*."""
        actx = cv.array_context
        boundary_discr = discr.discr_from_dd(btag)
        cv_minus = discr.project("vol", btag, cv)
        s_minus = discr.project("vol", btag, grad_cv)
        grad_t_minus = discr.project("vol", btag, grad_t)
        nodes = thaw(actx, boundary_discr.nodes())
        nhat = thaw(actx, discr.normal(btag))

        flux_weak = 0
        if self._viscous_flux_func:
            flux_weak = self._viscous_flux_func(nodes, eos, cv=cv_minus,
                                                grad_cv=s_minus,
                                                grad_temperature=grad_t_minus,
                                                nhat=nhat, **kwargs)
            return self._boundary_quantity(discr, btag, flux_weak, **kwargs)
        else:
            if self._q_func:
                cv_plus = self._q_func(nodes, eos=eos, cv=cv_minus, **kwargs)
            else:
                cv_plus = cv_minus

            if self._grad_q_func:
                s_plus = self._grad_q_func(nodes, eos, cv=cv_minus,
                                           grad_cv=s_minus, **kwargs)
            else:
                s_plus = s_minus

            if self._grad_t_func:
                grad_t_plus = self._grad_t_func(nodes, eos, cv=cv_minus,
                                                grad_temperature=grad_t_minus,
                                                **kwargs)
            else:
                grad_t_plus = grad_t_minus

            cv_tpair = TracePair(btag, interior=cv_minus, exterior=cv_plus)
            s_tpair = TracePair(btag, interior=s_minus, exterior=s_plus)
            grad_t_tpair = TracePair(btag, interior=grad_t_minus,
                                     exterior=grad_t_plus)

            from mirgecom.viscous import viscous_facial_flux
            return viscous_facial_flux(discr, eos, cv_tpair=cv_tpair,
                                       grad_cv_tpair=s_tpair,
                                       grad_t_tpair=grad_t_tpair)
