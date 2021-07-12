""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedBoundary
.. autoclass:: DummyBoundary
.. autoclass:: AdiabaticSlipBoundary
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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
from grudge.trace_pair import TracePair
from mirgecom.fluid import ConservedVars


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

    def boundary_pair(self, discr, cv, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        actx = cv.array_context

        boundary_discr = discr.discr_from_dd(btag)
        nodes = thaw(actx, boundary_discr.nodes())
        ext_soln = self._userfunc(nodes, **kwargs)
        int_soln = discr.project("vol", btag, cv)
        return TracePair(btag, interior=int_soln, exterior=ext_soln)


class DummyBoundary:
    """Boundary condition that assigns boundary-adjacent soln as the boundary solution.

    .. automethod:: boundary_pair
    """

    def boundary_pair(self, discr, cv, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        dir_soln = discr.project("vol", btag, cv)
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

    def boundary_pair(self, discr, cv, btag, **kwargs):
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
        bndry_cv = ConservedVars(mass=int_cv.mass,
                                 energy=int_cv.energy,
                                 momentum=ext_mom,
                                 species_mass=int_cv.species_mass)

        return TracePair(btag, interior=int_cv, exterior=bndry_cv)
