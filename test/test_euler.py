from __future__ import division, absolute_import, print_function

__copyright__ = """Copyright (C) 2020 CEESD"""

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
import pyopencl as cl
import pyopencl.clrandom
from pytools.obj_array import (
    join_fields, make_obj_array,
    with_object_array_or_scalar)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
# TODO: Remove grudge dependence?
from grudge.eager import with_queue
from grudge.symbolic.primitives import TracePair
from mirgecom.euler import _inviscid_flux_2d

# Tests go here

def test_inviscid_flux_2d():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    rho = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoE = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV0 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV1 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    
    rho[:] = 1.0
    rhoE[:] = 2.5
    rhoV0[:] = 0.0
    rhoV1[:] = 0.0

    rhoV = make_obj_array([rhoV0,rhoV1])
    q = join_fields(rho, rhoE, rhoV)
    
    flux = _inviscid_flux_2d(q)
    
    rhoflux = flux[0:1]
    rhoEflux = flux[2:3]
    rhoV0flux = flux[4:5]
    rhoV1flux = flux[6:7]

    print('rhoflux = ',rhoflux)
    print('rhoEflux = ',rhoEflux)
    print('rhoV0flux = ',rhoV0flux)
    print('rhoV1flux = ',rhoV1flux)
    
#    print(flux)
    
