"""Test the generic fluid helper functions."""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
import numpy.random
import numpy.linalg as la  # noqa
import pyopencl.clmath  # noqa
import logging
import pytest

from pytools.obj_array import (
    make_obj_array,
    obj_array_vectorize
)

from meshmode.dof_array import thaw
from mirgecom.euler import split_conserved, join_conserved
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(("mass_exp", "vel_fac"),
                         [(0, 0), (0, 1),
                          (1, 1), (2, 1),
                          (0, 2), (2, 2)])
def test_velocity_gradient_sanity(actx_factory, dim, mass_exp, vel_fac):
    """Test that the grad(v) returns {0, a*I} for v={constant, a*r_xyz}."""
    from mirgecom.fluid import velocity_gradient
    actx = actx_factory()

    nel_1d = 25

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 3
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    mass = 1*ones
    for i in range(mass_exp):
        mass *= (mass + i)
    energy = zeros + 2.5
    velocity = vel_fac * nodes
    mom = mass * velocity

    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
    cv = split_conserved(dim, q)

    grad_q = obj_array_vectorize(discr.grad, q)
    grad_cv = split_conserved(dim, grad_q)

    grad_v = velocity_gradient(discr, cv, grad_cv)

    tol = 1e-11
    exp_result = vel_fac * np.eye(dim) * ones
    grad_v_err = [discr.norm(grad_v[i] - exp_result[i], np.inf)
                  for i in range(dim)]

    assert max(grad_v_err) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_velocity_gradient_eoc(actx_factory, dim):
    """Test the velocity gradient with a trig function."""
    from mirgecom.fluid import velocity_gradient
    actx = actx_factory()

    order = 3

    npts_1d = 25

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(npts_1d,) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    energy = zeros + 2.5

    mass = nodes[dim-1]*nodes[dim-1]
    velocity = make_obj_array([actx.np.cos(nodes[i]) for i in range(dim)])
    mom = mass*velocity

    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
    cv = split_conserved(dim, q)

    grad_q = obj_array_vectorize(discr.grad, q)
    grad_cv = split_conserved(dim, grad_q)

    grad_v = velocity_gradient(discr, cv, grad_cv)

    def exact_grad_row(xdata, gdim, dim):
        exact_grad_row = make_obj_array([zeros for _ in range(dim)])
        exact_grad_row[gdim] = -actx.np.sin(xdata)
        return exact_grad_row

    tol = 1e-5
    comp_err = make_obj_array([
        discr.norm(grad_v[i] - exact_grad_row(nodes[i], i, dim), np.inf)
        for i in range(dim)])
    err_max = comp_err.max()
    assert err_max < tol
