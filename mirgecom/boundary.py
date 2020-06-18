__copyright__ = (
    "Copyright (C) 2020 University of Illinois Board of Trustees"
)

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
import numpy.linalg as la  # noqa
from pytools.obj_array import (
    join_fields,
    make_obj_array,
    with_object_array_or_scalar,
)
import pyopencl.clmath as clmath
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

# TODO: Remove grudge dependence?
from grudge.eager import with_queue
from grudge.symbolic.primitives import TracePair


class DummyBoundary:
    def __init__(self, tag=BTAG_ALL):
        self._boundary_tag = tag
        # dummy

    def GetBoundaryFlux(self, discr, w, t=0.0):
        ndim = discr.dim
        dir_soln = discr.interp("vol", self._boundary_tag, w)

        from mirgecom.euler import _facial_flux  # hrm

        return _facial_flux(
            discr,
            w_tpair=TracePair(self._boundary_tag, dir_soln, dir_soln),
        )


class BoundaryBoss:
    def __init__(self):
        self._boundaries = {}

    def AddBoundary(self, boundaryhandler):
        numbnd = len(self._boundaries)
        self._boundaries[numbnd] = boundaryhandler

    def GetBoundaryFlux(self, discr, w, t=0.0):
        queue = w[0].queue
        numbnd = len(self._boundaries)
        numsoln = len(w)

        assert numsoln > 0

        if numbnd == 0:
            self.AddBoundary(DummyBoundary())
            numbnd = 1

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        # Gak!  need help here. how to just calculate on the boundary?
        boundary_flux = discr.interp("vol", "all_faces", w)
        #        boundary_flux = join_fields( [discr.zeros(queue) for i in range(numsoln)] )
        boundary_flux = scalevec(0.0, boundary_flux)

        for bndindex in range(numbnd):
            boundaryhandler = self._boundaries[bndindex]
            this_boundary_flux = boundaryhandler.GetBoundaryFlux(
                discr, w, t
            )
            boundary_flux = boundary_flux + this_boundary_flux

        return boundary_flux
