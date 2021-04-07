"""Test the viscous fluid helper functions."""

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
import pytest  # noqa

from pytools.obj_array import make_obj_array, obj_array_vectorize

from meshmode.dof_array import thaw
from mirgecom.fluid import split_conserved, join_conserved  # noqa
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


def test_viscous_stress_tensor_structure(actx_factory):
    """Test tau data structure and values."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    mass = 2*ones

    energy = zeros + 2.5
    velocity_x = nodes[0] + 2*nodes[1] + 3*nodes[2]
    velocity_y = 4*nodes[0] + 5*nodes[1] + 6*nodes[2]
    velocity_z = 7*nodes[0] + 8*nodes[1] + 9*nodes[2]
    velocity = make_obj_array([velocity_x, velocity_y, velocity_z])

    mom = mass * velocity

    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)

    grad_q = obj_array_vectorize(discr.grad, q)

    mu_b = 1.0
    mu = 0.5
    from mirgecom.transport import SimpleTransport
    tv_model = SimpleTransport(bulk_viscosity=mu_b, viscosity=mu)

    from mirgecom.eos import IdealSingleGas
    eos = IdealSingleGas(transport_model=tv_model)

    # Exact answer for tau
    exp_grad_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    exp_grad_v_t = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    exp_grad_v_div = 15
    exp_tau = (mu*(exp_grad_v + exp_grad_v_t)
               + (mu_b - 2*mu/3)*exp_grad_v_div*np.eye(3))

    from mirgecom.viscous import viscous_stress_tensor
    tau = viscous_stress_tensor(discr, eos, q, grad_q)

    # The errors come from grad_v
    assert discr.norm(tau - exp_tau, np.inf) < 1e-12
