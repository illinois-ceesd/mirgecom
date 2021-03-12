"""Test the artificial viscosity functions."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""


__license__ = """
Permission is hereby granted,  free of charge,  to any person obtaining a copy
of this software and associated documentation files (the "Software"),  to deal
in the Software without restriction,  including without limitation the rights
to use,  copy,  modify,  merge,  publish,  distribute,  sublicense,  and/or sell
copies of the Software,  and to permit persons to whom the Software is
furnished to do so,  subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS",  WITHOUT WARRANTY OF ANY KIND,  EXPRESS OR
IMPLIED,  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,  DAMAGES OR OTHER
LIABILITY,  WHETHER IN AN ACTION OF CONTRACT,  TORT OR OTHERWISE,  ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
import numpy as np
import pyopencl as cl
import pytest
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL

from mirgecom.tag_cells import smoothness_indicator
from mirgecom.artificial_viscosity import artificial_viscosity
from mirgecom.boundary import DummyBoundary
from grudge.eager import EagerDGDiscretization
from pytools.obj_array import flat_obj_array
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim",  [1, 2, 3])
@pytest.mark.parametrize("order",  [1, 5])
def test_tag_cells(ctx_factory, dim, order):
    """Test tag_cells.

    Tests that the cells tagging properly tags cells
    given a prescirbed solutions.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 2
    tolerance = 1.e-16

    def norm_indicator(expected, discr, soln, **kwargs):
        return(discr.norm(expected-smoothness_indicator(soln, discr, **kwargs),
                          np.inf))

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-1.0, )*dim,  b=(1.0, )*dim,  n=(nel_1d, ) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    nele = mesh.nelements
    zeros = 0.0*nodes[0]

    # test jump discontinuity
    soln = actx.np.where(nodes[0] > 0.0+zeros, 1.0+zeros, zeros)
    err = norm_indicator(1.0, discr, soln)

    assert err < tolerance,  "Jump discontinuity should trigger indicator (1.0)"

    # get meshmode polynomials
    group = discr.discr_from_dd("vol").groups[0]
    basis = group.basis()  # only one group
    unit_nodes = group.unit_nodes
    modes = group.mode_ids()
    order = group.order

    # loop over modes and check smoothness
    for i, mode in enumerate(modes):
        ele_soln = basis[i](unit_nodes)
        soln[0].set(np.tile(ele_soln, (nele, 1)))
        if sum(mode) == order:
            expected = 1.0
        else:
            expected = 0.0
        err = norm_indicator(expected, discr, soln)
        assert err < tolerance,  "Only highest modes should trigger indicator (1.0)"

    # Test s0
    s0 = -1.
    eps = 1.0e-6

    phi_n_p = np.sqrt(np.power(10, s0))
    phi_n_pm1 = np.sqrt(1. - np.power(10, s0))

    # pick a polynomial of order n_p, n_p-1
    n_p = np.array(np.nonzero((np.sum(modes, axis=1) == order))).flat[0]
    n_pm1 = np.array(np.nonzero((np.sum(modes, axis=1) == order-1))).flat[0]

    # create test soln perturbed around
    # Solution above s0
    ele_soln = ((phi_n_p+eps)*basis[n_p](unit_nodes)
                + phi_n_pm1*basis[n_pm1](unit_nodes))
    soln[0].set(np.tile(ele_soln, (nele, 1)))
    err = norm_indicator(1.0, discr, soln, s0=s0, kappa=0.0)
    assert err < tolerance,  (
        "A function with an indicator >s0 should trigger indicator")

    # Solution below s0
    ele_soln = ((phi_n_p-eps)*basis[n_p](unit_nodes)
                + phi_n_pm1*basis[n_pm1](unit_nodes))
    soln[0].set(np.tile(ele_soln, (nele, 1)))
    err = norm_indicator(0.0, discr, soln, s0=s0, kappa=0.0)
    assert err < tolerance, (
        "A function with an indicator <s0 should not trigger indicator")

    # Test kappa
    # non-perturbed solution
    # test middle value
    kappa = 0.5
    ele_soln = (phi_n_p*basis[n_p](unit_nodes)
                + phi_n_pm1*basis[n_pm1](unit_nodes))
    soln[0].set(np.tile(ele_soln, (nele, 1)))
    err = norm_indicator(0.5, discr, soln, s0=s0, kappa=kappa)
    assert err < 1.0e-10,  "A function with s_e=s_0 should return 0.5"

    # test bounds
    # lower bound
    shift = 1.0e-5
    err = norm_indicator(0.0, discr, soln, s0=s0+kappa+shift, kappa=kappa)
    assert err < tolerance,  "s_e<s_0-kappa should not trigger indicator"
    err = norm_indicator(0.0, discr, soln, s0=s0+kappa-shift, kappa=kappa)
    assert err > tolerance,  "s_e>s_0-kappa should trigger indicator"

    # lower bound
    err = norm_indicator(1.0, discr, soln, s0=s0-(kappa+shift), kappa=kappa)
    assert err < tolerance,  "s_e>s_0+kappa should fully trigger indicator (1.0)"
    err = norm_indicator(1.0, discr, soln, s0=s0-(kappa-shift), kappa=kappa)
    assert err > tolerance,  "s_e<s_0+kappa should not fully trigger indicator (1.0)"


@pytest.mark.parametrize("dim",  [1, 2, 3])
@pytest.mark.parametrize("order",  [2, 3])
def test_artificial_viscosity(ctx_factory, dim, order):
    """Test artificial_viscosity.

    Tests the application on a few simple functions
    to confirm artificial viscosity returns the analytical result.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 10
    tolerance = 1.e-8

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-1.0, )*dim,  b=(1.0, )*dim,  n=(nel_1d, ) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)

    boundaries = {BTAG_ALL: DummyBoundary()}

    # Uniform field return 0 rhs
    fields = flat_obj_array(zeros+1.0)
    rhs = artificial_viscosity(discr, t=0, eos=None, boundaries=boundaries,
                               r=fields, alpha=1.0, s0=-np.inf)
    err = discr.norm(rhs, np.inf)
    assert err < tolerance

    # Linar field return 0 rhs
    fields = flat_obj_array(nodes[0])
    rhs = artificial_viscosity(discr, t=0, eos=None, boundaries=boundaries,
                               r=fields, alpha=1.0, s0=-np.inf)
    err = discr.norm(rhs, np.inf)
    assert err < tolerance

    # Quadratic field return constant 2
    fields = flat_obj_array(np.dot(nodes, nodes))
    rhs = artificial_viscosity(discr, t=0, eos=None, boundaries=boundaries,
                               r=fields, alpha=1.0, s0=-np.inf)
    err = discr.norm(2.*dim-rhs, np.inf)
    assert err < tolerance
