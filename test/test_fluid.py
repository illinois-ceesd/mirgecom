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

from mirgecom.fluid import make_conserved
from mirgecom.discretization import create_discretization_collection
import grudge.op as op
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


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
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3

    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    mass = 1*ones
    for i in range(mass_exp):
        mass *= (mass + i)

    energy = zeros + 2.5
    velocity = vel_fac * nodes
    mom = mass * velocity

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    from grudge.op import local_grad
    grad_cv = local_grad(discr, cv)

    grad_v = velocity_gradient(cv, grad_cv)

    tol = 1e-11
    exp_result = vel_fac * np.eye(dim) * ones
    grad_v_err = [actx.to_numpy(op.norm(discr, grad_v[i] - exp_result[i], np.inf))
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

    nel_1d_0 = 4
    for hn1 in [1, 2, 3, 4]:

        nel_1d = hn1 * nel_1d_0
        h = 1/nel_1d

        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
        )

        discr = create_discretization_collection(actx, mesh, order=order)
        nodes = actx.thaw(discr.nodes())
        zeros = discr.zeros(actx)

        mass = nodes[dim-1]*nodes[dim-1]
        energy = zeros + 2.5
        velocity = make_obj_array([actx.np.cos(nodes[i]) for i in range(dim)])
        mom = mass*velocity

        cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
        from grudge.op import local_grad
        grad_cv = local_grad(discr, cv)
        grad_v = velocity_gradient(cv, grad_cv)

        def exact_grad_row(xdata, gdim, dim):
            exact_grad_row = make_obj_array([zeros for _ in range(dim)])
            exact_grad_row[gdim] = -actx.np.sin(xdata)
            return exact_grad_row

        comp_err = make_obj_array([
            actx.to_numpy(
                op.norm(discr, grad_v[i] - exact_grad_row(nodes[i], i, dim), np.inf))
            for i in range(dim)])
        err_max = comp_err.max()
        eoc.add_data_point(h, err_max)

    logger.info(eoc)
    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < 1e-9
    )


def test_velocity_gradient_structure(actx_factory):
    """Test gradv data structure, verifying usability with other helper routines."""
    from mirgecom.fluid import velocity_gradient
    actx = actx_factory()
    dim = 3
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 1

    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    mass = 2*ones

    energy = zeros + 2.5
    velocity_x = nodes[0] + 2*nodes[1] + 3*nodes[2]
    velocity_y = 4*nodes[0] + 5*nodes[1] + 6*nodes[2]
    velocity_z = 7*nodes[0] + 8*nodes[1] + 9*nodes[2]
    velocity = make_obj_array([velocity_x, velocity_y, velocity_z])

    mom = mass * velocity

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    from grudge.op import local_grad
    grad_cv = local_grad(discr, cv)
    grad_v = velocity_gradient(cv, grad_cv)

    tol = 1e-11
    exp_result = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    exp_trans = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    exp_trace = 15
    assert grad_v.shape == (dim, dim)
    from meshmode.dof_array import DOFArray
    assert type(grad_v[0, 0]) == DOFArray

    def inf_norm(x):
        return actx.to_numpy(op.norm(discr, x, np.inf))

    assert inf_norm(grad_v - exp_result) < tol
    assert inf_norm(grad_v.T - exp_trans) < tol
    assert inf_norm(np.trace(grad_v) - exp_trace) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_species_mass_gradient(actx_factory, dim):
    """Test gradY structure and values against exact."""
    actx = actx_factory()
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 1

    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1

    nspecies = 2*dim
    mass = 2*ones  # make mass != 1
    energy = zeros + 2.5
    velocity = make_obj_array([ones for _ in range(dim)])
    mom = mass * velocity
    # assemble y so that each one has simple, but unique grad components
    y = make_obj_array([ones for _ in range(nspecies)])
    for idim in range(dim):
        ispec = 2*idim
        y[ispec] = ispec*(idim*dim+1)*sum([(iidim+1)*nodes[iidim]
                                           for iidim in range(dim)])
        y[ispec+1] = -y[ispec]
    species_mass = mass*y

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)
    from grudge.op import local_grad
    grad_cv = local_grad(discr, cv)

    from mirgecom.fluid import species_mass_fraction_gradient
    grad_y = species_mass_fraction_gradient(cv, grad_cv)

    assert grad_y.shape == (nspecies, dim)
    from meshmode.dof_array import DOFArray
    assert type(grad_y[0, 0]) == DOFArray

    def inf_norm(x):
        return actx.to_numpy(op.norm(discr, x, np.inf))

    tol = 1e-11
    for idim in range(dim):
        ispec = 2*idim
        exact_grad = np.array([(ispec*(idim*dim+1))*(iidim+1)
                                for iidim in range(dim)])
        assert inf_norm(grad_y[ispec] - exact_grad) < tol
        assert inf_norm(grad_y[ispec+1] + exact_grad) < tol
