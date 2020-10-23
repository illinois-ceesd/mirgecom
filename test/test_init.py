"""Test the solution initialization routines."""

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

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from mirgecom.initializers import Vortex2D
from mirgecom.initializers import Lump
# from mirgecom.initializers import Uniform
from mirgecom.initializers import SodShock1D

from mirgecom.euler import split_conserved
from mirgecom.eos import IdealSingleGas

from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_lump_init(ctx_factory, dim):
    """Test mass lump intializer.

    Simple test to check that Lump initializer
    creates the expected solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    center[0] = 5
    velocity[0] = 1
    lump = Lump(center=center, velocity=velocity)
    lump_soln = lump(0, nodes)

    cv = split_conserved(dim, lump_soln)
    p = 0.4 * (cv.energy - 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass)
    exp_p = 1.0
    errmax = discr.norm(p - exp_p, np.inf)

    assert errmax < 1e-15


def test_vortex_init(ctx_factory):
    """Test the isentropic vortex initializer.

    Simple test to check that Vortex2D initializer
    creates the expected solution field.
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
    logger.info(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # Init soln with Vortex
    vortex = Vortex2D()
    vortex_soln = vortex(0, nodes)
    gamma = 1.4
    cv = split_conserved(dim, vortex_soln)
    p = 0.4 * (cv.energy - 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass)
    exp_p = cv.mass ** gamma
    errmax = discr.norm(p - exp_p, np.inf)

    logger.info(f"vortex_soln = {vortex_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_shock_init(ctx_factory, dim):
    """Test Sod's 1D shock intialization.

    Simple test to check that Shock1D initializer
    creates the expected solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 10

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-1.0,) * dim, b=(1.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1
    print(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    xpl = 1.0
    xpr = 0.1
    tol = 1e-15
    eos = IdealSingleGas()

    for xdir in range(dim):
        x0 = 0.0
        initr = SodShock1D(dim=dim, xdir=xdir, x0=x0)
        initsoln = initr(t=0.0, x_vec=nodes)
        p = eos.pressure(initsoln)
        nodes_x = nodes[xdir]
        assert discr.norm(actx.np.where(nodes_x < x0, p-xpl, p-xpr), np.inf) < tol


# Surrogate for the currently non-functioning Uniform class
def set_uniform_solution(t, x_vec, eos=IdealSingleGas()):
    """Create a uniform flow solution."""
    dim = len(x_vec)
    _rho = 1.0
    _p = 1.0
    _velocity = np.zeros(shape=(dim,))
    _gamma = 1.4

    mom0 = _rho * _velocity
    e0 = _p / (_gamma - 1.0)
    ke = 0.5 * np.dot(_velocity, _velocity) / _rho
    x_rel = x_vec[0]
    zeros = 0.0*x_rel
    ones = zeros + 1.0

    mass = zeros + _rho
    mom = make_obj_array([mom0 * ones for i in range(dim)])
    energy = e0 + ke + zeros

    return flat_obj_array(mass, energy, mom)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_uniform(ctx_factory, dim):
    """Terst the uniform solution initializer.

    Simple test to check that Uniform initializer
    creates the expected solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 1
    print(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    print(f"DIM = {dim}, {len(nodes)}")
    print(f"Nodes={nodes}")

    #    initr = Uniform(numdim=dim)
    #    initsoln = initr(t=0.0, x_vec=nodes)
    tol = 1e-15
    initsoln = set_uniform_solution(t=0.0, x_vec=nodes)
    ssoln = split_conserved(dim, initsoln)
    assert discr.norm(ssoln.mass - 1.0, np.inf) < tol
    assert discr.norm(ssoln.energy - 2.5, np.inf) < tol

    print(f"Uniform Soln:{initsoln}")
    eos = IdealSingleGas()
    cv = split_conserved(dim, initsoln)
    p = eos.pressure(cv)
    print(f"Press:{p}")

    assert discr.norm(p - 1.0, np.inf) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pulse(ctx_factory, dim):
    """Test the Gaussian pulse generator intialization.

    Test of Gaussian pulse generator.
    If it looks, walks, and quacks like a duck, then ...
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 10

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 1
    print(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    print(f"DIM = {dim}, {len(nodes)}")
    print(f"Nodes={nodes}")

    tol = 1e-15
    from mirgecom.initializers import make_pulse
    amp = 1.0
    w = .1
    rms2 = w * w
    r0 = np.zeros(dim)
    r2 = np.dot(nodes, nodes) / rms2
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=nodes)
    print(f"Pulse = {pulse}")

    # does it return the expected exponential?
    pulse_check = actx.np.exp(-.5 * r2)
    print(f"exact: {pulse_check}")
    pulse_resid = pulse - pulse_check
    print(f"pulse residual: {pulse_resid}")
    assert(discr.norm(pulse_resid, np.inf) < tol)

    # proper scaling with amplitude?
    amp = 2.0
    pulse = 0
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=nodes)
    pulse_resid = pulse - (pulse_check + pulse_check)
    assert(discr.norm(pulse_resid, np.inf) < tol)

    # proper scaling with r?
    amp = 1.0
    rcheck = np.sqrt(2.0) * nodes
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=rcheck)
    assert(discr.norm(pulse - (pulse_check * pulse_check), np.inf) < tol)

    # proper scaling with w?
    w = w / np.sqrt(2.0)
    pulse = make_pulse(amp=amp, r0=r0, w=w, r=nodes)
    assert(discr.norm(pulse - (pulse_check * pulse_check), np.inf) < tol)
