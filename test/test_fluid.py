"""Test the generic fluid helper functions."""

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
import numpy.random
import numpy.linalg as la  # noqa
import pyopencl.clmath  # noqa
import logging
import pytest

from pytools.obj_array import make_obj_array

from meshmode.dof_array import thaw
from mirgecom.fluid import split_conserved, join_conserved
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_velocity_gradient(actx_factory, dim):
    """Test that the velocity gradient does the right things."""
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

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # Sanity check: grad(v=0) == 0
    mass = ones
    energy = zeros + 2.5
    mom = make_obj_array([zeros for _ in range(dim)])
    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)

    cv = split_conserved(dim, q)
    grad_v = compute_local_velocity_gradient(discr, cv)

    grad_v_norm = [discr.norm(grad_v[i], np.inf) for i in range(dim)]
    tol = 1e-16
    for i in range(dim):
        assert grad_v_norm[i] < tol

    # Sanity check: grad_j(v_i=r_i) == I
    mom = nodes
    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
    cv = split_conserved(dim, q)
    grad_v = compute_local_velocity_gradient(discr, cv)
    tol = 1e-12
    for i in range(dim):
        grad_v_comp = grad_v[i]
        for j in range(dim):
            if i == j:
                exp_value = 1.0
            else:
                exp_value = 0.0
            assert discr.norm(grad_v_comp[j] - exp_value, np.inf) < tol

    # Sanity check: grad_j(v_i=r_i) == I, constant rho != 1.0
    mass = zeros + 2.0
    mom = mass*nodes
    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
    cv = split_conserved(dim, q)
    grad_v = compute_local_velocity_gradient(discr, cv)
    tol = 1e-12
    for i in range(dim):
        grad_v_comp = grad_v[i]
        for j in range(dim):
            if i == j:
                exp_value = 1.0
            else:
                exp_value = 0.0
            assert discr.norm(grad_v_comp[j] - exp_value, np.inf) < tol

    # Sanity check: grad_j(v_i=r_i) == I, spatially varying rho
    mass = ((nodes[0] + 2.0) * nodes[0])  # quadratic rho
    mom = mass*nodes
    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
    cv = split_conserved(dim, q)
    grad_v = compute_local_velocity_gradient(discr, cv)
    tol = 1e-12
    for i in range(dim):
        grad_v_comp = grad_v[i]
        for j in range(dim):
            if i == j:
                exp_value = 1.0
            else:
                exp_value = 0.0
            assert discr.norm(grad_v_comp[j] - exp_value, np.inf) < tol

    # Test EOC for velocity gradient
    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    nel_1d_0 = 5
    for hn1 in [1, 2, 3, 4]:

        nel_1d = hn1 * (nel_1d_0 - 1) + 1
        h = 1/(nel_1d-1)

        mesh = generate_regular_rect_mesh(
            a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(actx, discr.nodes())

        mass = nodes[dim-1]*nodes[dim-1]
        velocity = make_obj_array([actx.np.cos(nodes[i]) for i in range(dim)])
        mom = mass*velocity
        q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)
        cv = split_conserved(dim, q)
        grad_v = compute_local_velocity_gradient(discr, cv)
        comp_err = make_obj_array([discr.norm(grad_v[i] - discr.grad(velocity[i]),
                                              np.inf) for i in range(dim)])
        max_err = comp_err.max()
        eoc.add_data_point(h, max_err)

    print(eoc)
    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < 1e-9
    )
