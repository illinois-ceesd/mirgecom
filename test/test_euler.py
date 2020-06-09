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
    gamma = 1.4
    
    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    rho = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoE = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV0 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV1 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    scal1 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    p = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    
    rhoV = make_obj_array([rhoV0,rhoV1])
    ke = 0.5*(rhoV0*rhoV0 + rhoV1*rhoV1)/rho
    p = (gamma-1.0)*(rhoE - ke) # ideal single spec
    scal1 = (rhoE + p)/rho

    expected_mass_flux = rhoV
    expected_energy_flux = scalevec(scal1,rhoV)
    expected_mom_flux1 = rhoV0*rhoV0/rho + p
    expected_mom_flux2 = rhoV0*rhoV1/rho # crossterms
    expected_mom_flux4 = rhoV1*rhoV1/rho + p

    q = join_fields(rho, rhoE, rhoV)
    
    flux = _inviscid_flux_2d(q)
    
    rhoflux = flux[0:2]
    rhoEflux = flux[2:4]
    rhoV0flux = flux[4:6]
    rhoV1flux = flux[6:]

    assert(rhoflux[0].get().all() == expected_mass_flux[0].get().all())
    assert(rhoflux[1].get().all() == expected_mass_flux[1].get().all())
    assert(rhoEflux[0].get().all() == expected_energy_flux[0].get().all())
    assert(rhoEflux[1].get().all() == expected_energy_flux[1].get().all())
    assert(rhoV0flux[0].get().all() == expected_mom_flux1.get().all())
    assert(rhoV0flux[1].get().all() == expected_mom_flux2.get().all()) # crossterms
    assert(rhoV1flux[0].get().all() == expected_mom_flux2.get().all()) # crossterms
    assert(rhoV1flux[1].get().all() == expected_mom_flux4.get().all())
