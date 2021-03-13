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

from pytools.obj_array import make_obj_array

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
                          (1, 1), (2, 1)])
def test_velocity_gradient_sanity(actx_factory, dim, mass_exp, vel_fac):
    """Test that the grad(v=r) returns I."""
    from mirgecom.fluid import compute_local_velocity_gradient
    actx = actx_factory()

    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 3
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0
    energy = zeros + 2.5

    mass = ones
    for i in range(mass_exp):
        mass *= (mass + i)

    velocity = vel_fac * nodes
    mom = mass * velocity
    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
    cv = split_conserved(dim, q)
    exp_result = vel_fac * np.eye(dim)

    tol = 1e-12
    grad_v = compute_local_velocity_gradient(discr, cv)
    grad_v_err = discr.norm(grad_v - exp_result, np.inf)
    assert grad_v_err < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_velocity_gradient_eoc(actx_factory, dim):
    """Test that the velocity gradient converges at the proper rate."""
    from mirgecom.fluid import compute_local_velocity_gradient
    actx = actx_factory()

    nel_1d = 16
    order = 3

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    nel_1d_0 = 5
    for hn1 in [1, 2, 3, 4]:

        nel_1d = hn1 * (nel_1d_0 - 1) + 1
        h = 1/(nel_1d-1)

        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(actx, discr.nodes())
        zeros = discr.zeros(actx)
        energy = zeros + 2.5

        mass = nodes[dim-1]*nodes[dim-1]
        velocity = make_obj_array([actx.np.cos(nodes[i]) for i in range(dim)])

        def exact_grad_row(xdata, gdim, dim):
            actx = xdata.array_context
            zeros = 0 * xdata
            exact_grad_row = make_obj_array([zeros for _ in range(dim)])
            exact_grad_row[gdim] = actx.np.sin(xdata)
            return exact_grad_row

        mom = mass*velocity
        q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
        cv = split_conserved(dim, q)
        grad_v = compute_local_velocity_gradient(discr, cv)
        comp_err = make_obj_array([
            discr.norm(grad_v[i] + exact_grad_row(nodes[i], i, dim), np.inf)
            for i in range(dim)])
        err_max = comp_err.max()
        eoc.add_data_point(h, err_max)

    print(eoc)
    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < 1e-9
    )
