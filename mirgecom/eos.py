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


class IdealSingleGas:
    def __init__(
            self,gamma=1.4,R=287.1):
        self._gamma = gamma
        self._R = R

    def Gamma(self):
        return self._gamma
    
    def R(self):
        return self._R

    def InternalEnergy(self,w):
        queue = w[0].queue
        rho = w[0]
        rhoE = w[1]
        rhoV = w[2:]

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        e = (rhoE - 0.5*np.dot(rhoV,rhoV)/rho)
        return e
    
    def Pressure(self,w):
        queue = w[0].queue
        rho = w[0]
        rhoE = w[1]
        rhoV = w[2:]

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        p = (self._gamma - 1.0)*self.InternalEnergy(w)
        return p

    def Temperature(self,w):
        queue = w[0].queue
        rho = w[0]
        rhoE = w[1]
        rhoV = w[2:]

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])
        T = ((self._gamma - 1.0)/self._R)*self.InternalEnergy(w) / rho
        return T
    
    def __call__(self,w):
        queue = w[0].queue
        rho = w[0]
        rhoE = w[1]
        rhoV = w[2:]

        def scalevec(scalar, vec):
            # workaround for object array behavior
            return make_obj_array([ni * scalar for ni in vec])

        return join_fields ( self.Pressure(w), self.Temperature(w) )

