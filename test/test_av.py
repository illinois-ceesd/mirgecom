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

from mirgecom.artificial_viscosity import (
    av_laplacian_operator,
    smoothness_indicator
)
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    project_fluid_state
)
from mirgecom.eos import IdealSingleGas

from grudge.eager import EagerDGDiscretization

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

from pytools.obj_array import make_obj_array

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim",  [1, 2, 3])
@pytest.mark.parametrize("order",  [1, 5])
def test_tag_cells(ctx_factory, dim, order):
    """Test tag_cells.

    Tests that the cells tagging properly tags cells
    given prescribed solutions.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 2
    tolerance = 1.e-16

    def norm_indicator(expected, discr, soln, **kwargs):
        return(discr.norm(expected-smoothness_indicator(discr, soln, **kwargs),
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
        a=(-1.0, )*dim, b=(1.0, )*dim, nelements_per_axis=(nel_1d, )*dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)

    class TestBoundary:
        def soln_gradient_flux(self, disc, btag, fluid_state, gas_model, **kwargs):
            fluid_state_int = project_fluid_state(disc, "vol", btag, fluid_state,
                                                  gas_model)
            cv_int = fluid_state_int.cv
            from grudge.trace_pair import TracePair
            bnd_pair = TracePair(btag,
                                 interior=cv_int,
                                 exterior=cv_int)
            nhat = thaw(actx, disc.normal(btag))
            from mirgecom.flux import gradient_flux_central
            flux_weak = gradient_flux_central(bnd_pair, normal=nhat)
            return disc.project(btag, "all_faces", flux_weak)

        def av_flux(self, disc, btag, diffusion, **kwargs):
            nhat = thaw(actx, disc.normal(btag))
            diffusion_minus = discr.project("vol", btag, diffusion)
            diffusion_plus = diffusion_minus
            from grudge.trace_pair import TracePair
            bnd_grad_pair = TracePair(btag, interior=diffusion_minus,
                                      exterior=diffusion_plus)
            from mirgecom.flux import divergence_flux_central
            flux_weak = divergence_flux_central(bnd_grad_pair, normal=nhat)
            return disc.project(btag, "all_faces", flux_weak)

    boundaries = {BTAG_ALL: TestBoundary()}
    # Uniform field return 0 rhs
    soln = zeros + 1.0
    cv = make_conserved(
        dim,
        mass=soln,
        energy=soln,
        momentum=make_obj_array([soln for _ in range(dim)]),
        species_mass=make_obj_array([soln for _ in range(dim)])
    )
    gas_model = GasModel(eos=IdealSingleGas())
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
    boundary_kwargs = {"gas_model": gas_model}
    rhs = av_laplacian_operator(discr, boundaries=boundaries,
                                boundary_kwargs=boundary_kwargs,
                                fluid_state=fluid_state, alpha=1.0, s0=-np.inf)
    err = discr.norm(rhs, np.inf)
    assert err < tolerance

    # Linear field return 0 rhs
    soln = nodes[0]
    cv = make_conserved(
        dim,
        mass=soln,
        energy=soln,
        momentum=make_obj_array([soln for _ in range(dim)]),
        species_mass=make_obj_array([soln for _ in range(dim)])
    )
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
    rhs = av_laplacian_operator(discr, boundaries=boundaries,
                                boundary_kwargs=boundary_kwargs,
                                fluid_state=fluid_state, alpha=1.0, s0=-np.inf)
    err = discr.norm(rhs, np.inf)
    assert err < tolerance

    # Quadratic field return constant 2
    soln = np.dot(nodes, nodes)
    cv = make_conserved(
        dim,
        mass=soln,
        energy=soln,
        momentum=make_obj_array([soln for _ in range(dim)]),
        species_mass=make_obj_array([soln for _ in range(dim)])
    )
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
    rhs = av_laplacian_operator(discr, boundaries=boundaries,
                                boundary_kwargs=boundary_kwargs,
                                fluid_state=fluid_state, alpha=1.0, s0=-np.inf)
    err = discr.norm(2.*dim-rhs, np.inf)
    assert err < tolerance
