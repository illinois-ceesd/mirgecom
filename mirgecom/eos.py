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
import numpy.linalg as la  # noqa
from pytools.obj_array import flat_obj_array
import pyopencl.clmath as clmath
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa


class IdealSingleGas:
    """ Implements the ideal gas law (p = rhoRT)
    for a single monatomic gas.
    """

    def __init__(self, gamma=1.4, gas_const=287.1):
        self._gamma = gamma
        self._gas_const = gas_const

    def gamma(self):
        return self._gamma

    def gas_const(self):
        return self._gas_const

    def internal_energy(self, w):

        mass = w[0]
        energy = w[1]
        mom = w[2:]

        e = energy - 0.5 * np.dot(mom, mom) / mass
        return e

    def pressure(self, w):
        p = (self._gamma - 1.0) * self.internal_energy(w)
        return p

    def sound_speed(self, w):
        mass = w[0]
        p = self.pressure(w)
        c2 = self._gamma / mass * p
        c = clmath.sqrt(c2)
        return c

    def temperature(self, w):
        mass = w[0]
        temper = (
            ((self._gamma - 1.0) / self._gas_const)
            * self.internal_energy(w)
            / mass
        )
        return temper

    def __call__(self, w):
        return flat_obj_array(self.pressure(w), self.temperature(w))
