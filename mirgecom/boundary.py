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

from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.eos import IdealSingleGas
from grudge.symbolic.primitives import TracePair


class PrescribedBoundary:
    """Boundary condition assigns the boundary solution with a
    user-specified function
    """
    def __init__(self, userfunc):
        self._userfunc = userfunc

    def boundary_pair(
            self, discr, q, t=0.0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        actx = q[0].array_context

        boundary_discr = discr.discr_from_dd(btag)
        nodes = thaw(actx, boundary_discr.nodes())
        ext_soln = self._userfunc(t, nodes)
        int_soln = discr.project("vol", btag, q)
        return TracePair(btag, int_soln, ext_soln)


class DummyBoundary:
    """Simple example boundary condition that interpolates the
    boundary-adjacent volume solution to both sides of a boundary
    face.
    """
    def boundary_pair(
        self, discr, q, t=0.0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        dir_soln = discr.project("vol", btag, q)
        return TracePair(btag, dir_soln, dir_soln)
