"""Test the EOS interfaces."""

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

import logging
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clrandom
import pyopencl.clmath
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw

from mirgecom.eos import IdealSingleGas
from mirgecom.initializers import Vortex2D
from mirgecom.initializers import Lump
from mirgecom.fluid import split_conserved
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

logger = logging.getLogger(__name__)


def test_idealsingle_lump(ctx_factory):
    """Test EOS with mass lump.

    Tests that the IdealSingleGas EOS returns
    the correct (uniform) pressure for the Lump
    solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], n=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    center[0] = 5
    velocity[0] = 1
    lump = Lump(dim=dim, center=center, velocity=velocity)
    eos = IdealSingleGas()
    lump_soln = lump(nodes)

    cv = split_conserved(dim, lump_soln)
    p = eos.pressure(cv)
    exp_p = 1.0
    errmax = discr.norm(p - exp_p, np.inf)

    exp_ke = 0.5 * cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = discr.norm(ke - exp_ke, np.inf)

    te = eos.total_energy(cv, p)
    terr = discr.norm(te - cv.energy, np.inf)

    logger.info(f"lump_soln = {lump_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15


def test_idealsingle_vortex(ctx_factory):
    r"""Test EOS with isentropic vortex.

    Tests that the IdealSingleGas EOS returns
    the correct pressure (p) for the Vortex2D solution
    field (i.e. $p = \rho^{\gamma}$).
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], n=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    eos = IdealSingleGas()
    # Init soln with Vortex
    vortex = Vortex2D()
    vortex_soln = vortex(nodes)
    cv = split_conserved(dim, vortex_soln)
    gamma = eos.gamma()
    p = eos.pressure(cv)
    exp_p = cv.mass ** gamma
    errmax = discr.norm(p - exp_p, np.inf)

    exp_ke = 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = discr.norm(ke - exp_ke, np.inf)

    te = eos.total_energy(cv, p)
    terr = discr.norm(te - cv.energy, np.inf)

    logger.info(f"vortex_soln = {vortex_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15
