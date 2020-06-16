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
from mirgecom.eos import IdealSingleGas


class Vortex2D:
    def __init__(
        self,
        eos=IdealSingleGas(),
        center=np.array([0, 0]),
        velocity=np.array([0, 0]),
    ):
        self._beta = 5
        self._eos = eos
        self._gamma = self._eos.Gamma()
        self._center = center  # np.array([5, 0])
        self._velocity = velocity  # np.array([0, 0])
        self._boundary_tag = BTAG_ALL

    def __call__(self, t, x_vec):
        vortex_loc = self._center + t * self._velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]

        # Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
        # also JSH/TW Nodal DG Methods, p. 209

        from math import pi

        r = clmath.sqrt(x_rel ** 2 + y_rel ** 2)
        expterm = self._beta * clmath.exp(1 - r ** 2)
        u = self._velocity[0] - expterm * y_rel / (2 * pi)
        v = self._velocity[1] + expterm * x_rel / (2 * pi)
        rho = (
            1
            - (self._gamma - 1)
            / (16 * self._gamma * pi ** 2)
            * expterm ** 2
        ) ** (1 / (self._gamma - 1))
        p = rho ** self._gamma

        e = p / (self._gamma - 1) + rho / 2 * (u ** 2 + v ** 2)

        return join_fields(rho, e, rho * u, rho * v)

    def SetEOS(self, eos):
        self._eos = eos

    def SetBoundaryTag(self, tag=BTAG_ALL):
        self._boundary_tag = tag

    def GetBoundaryFlux(self, discr, w, t=0.0):
        queue = w[0].queue
        ndim = discr.dim

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        # help - how to make it just the boundary nodes?
        nodes = discr.nodes().with_queue(queue)
        vortex_soln = self.__call__(t, nodes)
        dir_bc = discr.interp("vol", self._boundary_tag, vortex_soln)
        dir_soln = discr.interp("vol", self._boundary_tag, w)
        from mirgecom.euler import _facial_flux  # hrm

        return _facial_flux(
            discr,
            w_tpair=TracePair(self._boundary_tag, dir_soln, dir_bc),
        )


class Lump:
    def __init__(
        self,
        eos=IdealSingleGas(),
        rho0=1.0,
        rhoamp=1.0,
        numdim=2,
        center=[0],
        velocity=[0],
    ):
        if len(center) == numdim:
            self._center = center
        else:
            self._center = np.zeros(shape=(numdim,))

        if len(velocity) == numdim:
            self._velocity = velocity
        else:
            self._velocity = np.zeros(shape=(numdim,))

        print("center shape = ", self._center.shape)
        print("velocity shape = ", self._velocity.shape)

        self._eos = eos
        self._rho0 = rho0
        self._rhoamp = rhoamp
        self._dim = numdim
        self._gamma = eos.Gamma()
        self._boundary_tag = BTAG_ALL

    def __call__(self, t, x_vec):
        lump_loc = self._center + t * self._velocity

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - lump_loc[i] for i in range(self._dim)]
        )
        r = clmath.sqrt(np.dot(rel_center, rel_center))

        from math import pi

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        expterm = self._rhoamp * clmath.exp(1 - r ** 2)
        rho = expterm + self._rho0
        rhoV = scalevec(rho, self._velocity)
        rhoE = (1.0 / (self._gamma - 1.0)) + np.dot(rhoV, rhoV) / (
            2.0 * rho
        )

        return join_fields(rho, rhoE, rhoV)

    def SetEOS(self, eos):
        self._eos = eos

    def ExpectedRHS(self, discr, w, t=0.0):
        queue = w[0].queue
        ndim = discr.dim
        nodes = discr.nodes().with_queue(queue)
        lump_loc = self._center + t * self._velocity
        # coordinates relative to lump center
        rel_center = make_obj_array(
            [nodes[i] - lump_loc[i] for i in range(self._dim)]
        )
        r = clmath.sqrt(np.dot(rel_center, rel_center))

        from math import pi

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        # The expected rhs is:
        # rhorhs  = -2*rho*(r.dot.v)
        # rhoerhs = -rho*v^2*(r.dot.v)
        # rhovrhs = -2*rho*(r.dot.v)*v

        expterm = self._rhoamp * clmath.exp(1 - r ** 2)
        rho = expterm + self._rho0
        rhoV = scalevec(rho, self._velocity)
        rhoE = (1.0 / (self._gamma - 1.0)) + np.dot(rhoV, rhoV) / (
            2.0 * rho
        )

        v = scalevec(1 / rho, rhoV)
        v2 = np.dot(v, v)
        rdotv = np.dot(rel_center, v)
        rhorhs = -2 * rdotv * rho
        rhoErhs = -v2 * rdotv * rho
        rhoVrhs = scalevec(-2 * rho * rdotv, v)

        return join_fields(rhorhs, rhoErhs, rhoVrhs)

    def SetBoundaryTag(self, tag=BTAG_ALL):
        self._boundary_tag = tag

    def GetBoundaryFlux(self, discr, w, t=0.0):
        queue = w[0].queue
        ndim = discr.dim

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        # help - how to make it just the boundary nodes?
        nodes = discr.nodes().with_queue(queue)
        mysoln = self.__call__(t, nodes)
        dir_bc = discr.interp("vol", self._boundary_tag, mysoln)
        dir_soln = discr.interp("vol", self._boundary_tag, w)
        from mirgecom.euler import _facial_flux  # hrm

        return _facial_flux(
            discr,
            w_tpair=TracePair(self._boundary_tag, dir_soln, dir_bc),
        )
