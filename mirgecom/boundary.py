"""Boundary condition implementations.

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
from pytools.obj_array import make_obj_array
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.eos import IdealSingleGas
from grudge.symbolic.primitives import TracePair
from mirgecom.euler import split_conserved, join_conserved


class PrescribedBoundary:
    """Assign the boundary solution with a user-specified function.

    .. automethod:: __init__
    .. automethod:: boundary_pair
    """

    def __init__(self, userfunc):
        """Set the boundary function.

        Parameters
        ----------
        userfunc
            User function must take two parameters: time and nodal
            coordinates, and produce the solution at each node.
        """
        self._userfunc = userfunc

    def boundary_pair(
            self, discr, q, t=0.0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        """Get the interior and exterior solution on the boundary."""
        actx = q[0].array_context

        boundary_discr = discr.discr_from_dd(btag)
        nodes = thaw(actx, boundary_discr.nodes())
        ext_soln = self._userfunc(t, nodes)
        int_soln = discr.project("vol", btag, q)
        return TracePair(btag, interior=int_soln, exterior=ext_soln)


class DummyBoundary:
    """Use the boundary-adjacent solution as the boundary solution.

    .. automethod:: boundary_pair
    """

    def boundary_pair(
        self, discr, q, t=0.0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        """Get the interior and exterior solution on the boundary."""
        dir_soln = discr.project("vol", btag, q)
        return TracePair(btag, interior=dir_soln, exterior=dir_soln)


class AdiabaticSlipBoundary:
    """Adiabatic slip boundary for inviscid flows.

    a.k.a. Reflective inviscid wall boundary

    This class implements an adiabatic reflective slip boundary
    wherein the normal component of velocity at the wall is 0, and
    tangential components are preserved. These perfectly reflecting
    conditions are used by the forward-facing step case in
    JSH/TW Nodal DG Methods, Section 6.6 DOI: 10.1007/978-0-387-72067-8 and
    described in detail by Poinsot and Lele's JCP paper
    http://acoustics.ae.illinois.edu/pdfs/poinsot-lele-1992.pdf

    .. automethod:: boundary_pair
    """

    def boundary_pair(
            self, discr, q, t=0.0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        """Get the interior and exterior solution on the boundary."""
        # Grab some boundary-relevant data
        dim = discr.dim
        cv = split_conserved(dim, q)
        actx = cv.mass.array_context

        # Grab a unit normal to the boundary
        normal = thaw(actx, discr.normal(btag))
        normal_mag = actx.np.sqrt(np.dot(normal, normal))
        nhat_mult = 1.0 / normal_mag
        nhat = normal * make_obj_array([nhat_mult])

        # Get the interior/exterior solns
        int_soln = discr.project("vol", btag, q)
        bndry_cv = split_conserved(dim, int_soln)
        # bpressure = eos.pressure(bndry_cv)

        # Subtract out the 2*wall-normal component
        # of velocity from the velocity at the wall to
        # induce an equal but opposite wall-normal (reflected) wave
        # preserving the tangential component
        wall_velocity = bndry_cv.momentum / make_obj_array([bndry_cv.mass])
        nvel_comp = np.dot(wall_velocity, nhat)  # part of velocity normal to wall
        wnorm_vel = nhat * make_obj_array([nvel_comp])  # wall-normal velocity vec
        wall_velocity = wall_velocity - 2.0 * wnorm_vel  # prescribed ext velocity

        # Re-calculate the boundary solution with the new
        # momentum
        bndry_cv.momentum = wall_velocity * make_obj_array([bndry_cv.mass])
        # bndry_cv.energy = eos.total_energy(bndry_cv, bpressure)
        bndry_soln = join_conserved(dim=dim, mass=bndry_cv.mass,
                                    energy=bndry_cv.energy,
                                    momentum=bndry_cv.momentum)

        return TracePair(btag, interior=int_soln, exterior=bndry_soln)
