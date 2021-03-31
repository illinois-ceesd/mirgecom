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

from pytools.obj_array import make_obj_array, obj_array_vectorize

from meshmode.dof_array import thaw
from mirgecom.fluid import split_conserved, join_conserved
from grudge.eager import EagerDGDiscretization
from grudge.symbolic.primitives import DOFDesc
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


# Box grid generator widget lifted from @majosm's diffusion tester
def _get_box_mesh(dim, a, b, n):
    dim_names = ["x", "y", "z"]
    boundary_tag_to_face = {}
    for i in range(dim):
        boundary_tag_to_face["-"+str(i+1)] = ["-"+dim_names[i]]
        boundary_tag_to_face["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh
    return generate_regular_rect_mesh(a=(a,)*dim, b=(b,)*dim, n=(n,)*dim,
        boundary_tag_to_face=boundary_tag_to_face)


# Simple obj array vectorized weak grad call
def _vector_weak_grad(discr, q):
    return obj_array_vectorize(discr.weak_grad, q)


# DG grad tester works only for continuous functions
def _vector_dg_grad(discr, q):
    ncomp = 1
    if isinstance(q, np.ndarray):
        actx = q[0].array_context
        ncomp = len(q)
    else:
        actx = q.array_context

    vol_part = _vector_weak_grad(discr, q)
    q_minus = discr.project("vol", "all_faces", q)
    dd = DOFDesc("all_faces")
    normal = thaw(actx, discr.normal(dd))
    if ncomp > 1:
        facial_flux = make_obj_array([q_minus[i]*normal for i in range(ncomp)])
    else:
        facial_flux = q_minus*normal
    return -discr.inverse_mass(vol_part - discr.face_mass(facial_flux))


# Get the grudge internal grad for *q*
def _grad(discr, q):
    return obj_array_vectorize(discr.grad, q)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(("mass_exp", "vel_fac"),
                         [(0, 0), (0, 1),
                          (1, 1), (2, 1)])
def test_velocity_gradient_sanity(actx_factory, dim, mass_exp, vel_fac):
    """Test that the grad(v) returns {0, I} for v={constant, r_xyz}."""
    from mirgecom.fluid import velocity_gradient
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

    tol = 1e-12
    exp_result = vel_fac * np.eye(dim) * ones
    grad_v_err = [discr.norm(grad_v[i] - exp_result[i], np.inf)
                  for i in range(dim)]

    assert max(grad_v_err) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_velocity_gradient_eoc(actx_factory, dim):
    """Test that the velocity gradient converges at the proper rate."""
    from mirgecom.fluid import velocity_gradient
    actx = actx_factory()

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

        mass = nodes[dim-1]*nodes[dim-1]
        energy = zeros + 2.5
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

        comp_err = make_obj_array([
            discr.norm(grad_v[i] - exact_grad_row(nodes[i], i, dim), np.inf)
            for i in range(dim)])
        err_max = comp_err.max()
        eoc.add_data_point(h, err_max)

    print(eoc)
    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < 1e-9
    )
