""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Inviscid Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedBoundary
.. autoclass:: DummyBoundary
.. autoclass:: AdiabaticSlipBoundary

Viscous Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: IsothermalNoSlip
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
# from mirgecom.eos import IdealSingleGas
from grudge.symbolic.primitives import TracePair
from mirgecom.fluid import split_conserved, join_conserved


class ViscousBC:
    r"""Abstract interface to viscous boundary conditions.

    .. automethod:: q_flux
    .. automethod:: t_flux
    .. automethod:: inviscid_flux
    .. automethod:: viscous_flux
    .. automethod:: boundary_pair
    """

    def get_q_flux(self, discr, btag, eos, q, **kwargs):
        """Get the flux through boundary *btag* for each scalar in *q*."""
        raise NotImplementedError()

    def get_t_flux(self, discr, btag, eos, q, **kwargs):
        """Get the "temperature flux" through boundary *btag*."""
        raise NotImplementedError()

    def get_inviscid_flux(self, discr, btag, eos, q, **kwargs):
        """Get the inviscid part of the physical flux across the boundary *btag*."""
        raise NotImplementedError()

    def get_viscous_flux(self, discr, btag, eos, q, grad_q, t, grad_t, **kwargs):
        """Get the viscous part of the physical flux across the boundary *btag*."""
        raise NotImplementedError()

    def get_boundary_pair(self, discr, btag, eos, u, **kwargs):
        """Get the interior and exterior solution (*u*) on the boundary."""
        raise NotImplementedError()


class PrescribedBoundary:
    """Boundary condition prescribes boundary soln with user-specified function.

    .. automethod:: __init__
    .. automethod:: boundary_pair
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
        self._userfunc = userfunc

    def boundary_pair(self, discr, q, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        actx = q[0].array_context

        boundary_discr = discr.discr_from_dd(btag)
        nodes = thaw(actx, boundary_discr.nodes())
        ext_soln = self._userfunc(nodes, **kwargs)
        int_soln = discr.project("vol", btag, q)
        return TracePair(btag, interior=int_soln, exterior=ext_soln)


class DummyBoundary:
    """Boundary condition that assigns boundary-adjacent soln as the boundary solution.

    .. automethod:: boundary_pair
    """

    def boundary_pair(self, discr, q, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        dir_soln = discr.project("vol", btag, q)
        return TracePair(btag, interior=dir_soln, exterior=dir_soln)


class AdiabaticSlipBoundary:
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

    .. automethod:: boundary_pair
    """

    def boundary_pair(self, discr, q, btag, **kwargs):
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
        cv = split_conserved(dim, q)
        actx = cv.mass.array_context

        # Grab a unit normal to the boundary
        nhat = thaw(actx, discr.normal(btag))

        # Get the interior/exterior solns
        int_soln = discr.project("vol", btag, q)
        int_cv = split_conserved(dim, int_soln)

        # Subtract out the 2*wall-normal component
        # of velocity from the velocity at the wall to
        # induce an equal but opposite wall-normal (reflected) wave
        # preserving the tangential component
        mom_normcomp = np.dot(int_cv.momentum, nhat)  # wall-normal component
        wnorm_mom = nhat * mom_normcomp  # wall-normal mom vec
        ext_mom = int_cv.momentum - 2.0 * wnorm_mom  # prescribed ext momentum

        # Form the external boundary solution with the new momentum
        bndry_soln = join_conserved(dim=dim, mass=int_cv.mass,
                                    energy=int_cv.energy,
                                    momentum=ext_mom,
                                    species_mass=int_cv.species_mass)

        return TracePair(btag, interior=int_soln, exterior=bndry_soln)


class IsothermalNoSlip(ViscousBC):
    r"""Isothermal no-slip viscous wall boundary.

    This class implements an isothermal no-slip wall by:
    (TBD)
    [Hesthaven_2008]_, Section 6.6, and correspond to the characteristic
    boundary conditions described in detail in [Poinsot_1992]_.
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature

    def get_boundary_pair(self, discr, btag, eos, q, **kwargs):
        """Get the interior and exterior solution (*q*) on the boundary."""
        q_minus = discr.project("vol", btag, q)
        cv_minus = split_conserved(discr.dim, q_minus)

        temperature_plus = self._wall_temp + 0*cv_minus.mass
        velocity_plus = -cv_minus.momentum / cv_minus.mass
        mass_frac_plus = cv_minus.species_mass / cv_minus.mass
        internal_energy_plus = eos.get_internal_energy(
            temperature=temperature_plus, species_fractions=mass_frac_plus
        )
        total_energy_plus = cv_minus.mass*(internal_energy_plus
                                           + .5*np.dot(velocity_plus, velocity_plus))
        q_plus = join_conserved(
            discr.dim, mass=cv_minus.mass, energy=total_energy_plus,
            momentum=-cv_minus.momentum, species_mass=cv_minus.species_mass
        )
        return TracePair(btag, q_minus, q_plus)

    def get_q_flux(self, discr, btag, eos, q, **kwargs):
        """Get the flux through boundary *btag* for each scalar in *q*."""
        bnd_tpair = self.get_boundary_pair(discr, btag, eos, q, **kwargs)
        cv_minus = split_conserved(discr.dim, bnd_tpair.int)
        actx = cv_minus.mass.array_context
        nhat = thaw(actx, discr.normal(btag))
        from mirgecom.flux import central_scalar_flux
        flux_func = central_scalar_flux
        if "numerical_flux_func" in kwargs:
            flux_func = kwargs.get("numerical_flux_func")

        return flux_func(bnd_tpair, nhat)

    def get_t_flux(self, discr, btag, eos, q, **kwargs):
        """Get the "temperature flux" through boundary *btag*."""
        q_minus = discr.project("vol", btag, q)
        cv_minus = split_conserved(discr.dim, q_minus)

        actx = cv_minus.mass.array_context
        nhat = thaw(actx, discr.normal(btag))

        t_minus = eos.temperature(cv_minus)
        t_plus = 0*t_minus + self._wall_temp
        bnd_tpair = TracePair(btag, t_minus, t_plus)

        from mirgecom.flux import central_scalar_flux
        flux_func = central_scalar_flux
        if "numerical_flux_func" in kwargs:
            flux_func = kwargs.get("numerical_flux_func")

        return flux_func(bnd_tpair, nhat)

    def get_inviscid_flux(self, discr, btag, eos, q, **kwargs):
        """Get the inviscid part of the physical flux across the boundary *btag*."""
        bnd_tpair = self.get_boundary_pair(discr, btag, eos, q, **kwargs)
        from mirgecom.inviscid import inviscid_facial_flux
        return inviscid_facial_flux(discr, eos, bnd_tpair)

    def get_viscous_flux(self, discr, btag, eos, q, grad_q, t, grad_t, **kwargs):
        """Get the viscous part of the physical flux across the boundary *btag*."""
        q_tpair = self.get_boundary_pair(discr, btag, eos, q, **kwargs)

        grad_q_minus = discr.project("vol", btag, grad_q)
        grad_q_tpair = TracePair(btag, grad_q_minus, grad_q_minus)

        t_minus = discr.project("vol", btag, t)
        t_plus = 0*t_minus + self._wall_temp
        t_tpair = TracePair(btag, t_minus, t_plus)

        grad_t_minus = discr.project("vol", btag, grad_t)
        grad_t_tpair = TracePair(btag, grad_t_minus, grad_t_minus)

        from mirgecom.viscous import viscous_facial_flux
        return viscous_facial_flux(discr, eos, q_tpair, grad_q_tpair,
                                   t_tpair, grad_t_tpair)
