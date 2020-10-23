"""Boundary condition implementations.

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedBoundary
.. autoclass:: DummyBoundary
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
from mirgecom.eos import IdealSingleGas
# from mirgecom.euler import split_conserved
from grudge.symbolic.primitives import TracePair


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
    """Adiabatic slip wall boundary for inviscid flows."""
    def boundary_pair(
            self, discr, q, t=0.0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        """Get interior/exterior facial trace pairs."""
        # Grab some boundary-relevant data
        actx = q[0].array_context
        dim = discr.dim
        #       boundary_discr = discr.discr_from_dd(btag)
        normal = thaw(actx, discr.normal(btag))
        normal_mag = actx.np.sqrt(np.dot(normal, normal))
        nhat_mult = 1.0 / normal_mag
        #        print(f"normal={normal}")
        #        print(f"normal_mag={normal_mag}")
        # nhat = normal * make_obj_array(nhat_mult)
        nhat = normal
        for i in range(dim):
            nhat[i] = nhat_mult * nhat[i]

        # Get the interior/exterior solns
        int_soln = discr.project("vol", btag, q)
        boundary_soln = discr.project("vol", btag, q)  # copy?
        bpressure = eos.pressure(boundary_soln)
        #        bsoln = split_conserved(dim, boundary_soln)

        # Subtract out the wall-normal component
        # of velocity from the velocity at the wall
        #        wall_velocity = bsoln.momentum / bsoln.mass
        wall_velocity = 1.0 * boundary_soln[2:]
        wnorm_vel = 1.0 * wall_velocity
        for i in range(dim):
            wall_velocity[i] = wall_velocity[i] / boundary_soln[0]
        nvelhat = np.dot(wall_velocity, nhat)
        for i in range(dim):
            wnorm_vel[i] = nvelhat * normal[i]
            wall_velocity[i] = wall_velocity[i] - 2.0 * wnorm_vel[i]

        # wall_velocity = wall_velocity - 2.0 * wnorm_vel

        # Re-calculate the boundary solution with the new
        # momentum

        #        bsoln.momentum = bsoln.mass * wall_velocity
        #        bsoln.energy = eos.energy(bsoln.mass, bpressure, bsoln.momentum)
        for i in range(dim):
            boundary_soln[2 + i] = boundary_soln[0] * wall_velocity[i]
        boundary_soln[1] = eos.energy(boundary_soln[0], bpressure, boundary_soln[2:])

        return TracePair(btag, interior=int_soln, exterior=boundary_soln)
