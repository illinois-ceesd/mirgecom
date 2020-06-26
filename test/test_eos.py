from __future__ import division, absolute_import, print_function

__copyright__ = (
    """Copyright (C) 2020 University of Illinois Board of Trustees"""
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
import pyopencl as cl
import pyopencl.clrandom
import pyopencl.clmath
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from mirgecom.eos import IdealSingleGas
from mirgecom.initializers import Vortex2D
from mirgecom.initializers import Lump
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)


def test_idealsingle_lump():
    #    cl_ctx = ctx_factory()
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = "test_idealsingle_lump: "
    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], n=(nel_1d,) * dim
    )

    order = 3
    print(iotag + "%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    nodes = discr.nodes().with_queue(queue)

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    center[0] = 5
    velocity[0] = 1
    lump = Lump(center=center, velocity=velocity)
    eos = IdealSingleGas()
    lump_soln = lump(0, nodes)

    p = eos.pressure(lump_soln)
    exp_p = 1.0
    errmax = np.max(np.abs(p - exp_p))

    print("lump_soln = ", lump_soln)
    print("pressure = ", p)

    assert errmax < 1e-15


def test_idealsingle_vortex():
    #    cl_ctx = ctx_factory()
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = "test_idealsingle_vortex: "
    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], n=(nel_1d,) * dim
    )

    order = 3
    print(iotag + "%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    nodes = discr.nodes().with_queue(queue)
    eos = IdealSingleGas()
    # Init soln with Vortex
    vortex = Vortex2D()
    vortex_soln = vortex(0, nodes)
    rho = vortex_soln[0]
    gamma = eos.gamma()
    p = eos.pressure(vortex_soln)
    exp_p = rho ** gamma
    errmax = np.max(np.abs(p - exp_p))

    print("vortex_soln = ", vortex_soln)
    print("pressure = ", p)

    assert errmax < 1e-15
