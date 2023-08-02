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
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
)
from meshmode.mesh import BTAG_ALL
from meshmode.discretization.connection import FACE_RESTR_ALL
import grudge.op as op
from mirgecom.artificial_viscosity import (
    av_laplacian_operator,
    smoothness_indicator,
    AdiabaticNoSlipWallAV,
    PrescribedFluidBoundaryAV
)
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.eos import IdealSingleGas

from mirgecom.discretization import create_discretization_collection
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from mirgecom.simutil import get_box_mesh
from pytools.obj_array import make_obj_array

logger = logging.getLogger(__name__)


# NOTE:  Testing of this av_laplacian_operator is currently
# pretty limited.  This fact is somewhat indicative of the
# limitations and shortcomings of this operator.  We intend
# to soon replace our shock-handling approach with one that is
# more robust in the presence of discontinuous coeffcients
# and in which we understand the required boundary conditions.
# Tracking the replacement endeavor:
# https://github.com/illinois-ceesd/mirgecom/issues/684

@pytest.mark.parametrize("dim",  [1, 2, 3])
@pytest.mark.parametrize("order",  [1, 5])
def test_tag_cells(ctx_factory, dim, order):
    """Test tag_cells.

    This test checks that tag_cells properly tags cells near
    discontinuous or nearly-discontinuous features in 1d test solutions.
    The following tests/functions are used:
    - Discontinuity detection/Heaviside step discontinuity
    - Detection smoothness/Element basis functions for each mode
    - Detection thresholding/(p, p-1) polynomials greater/less/equal to s0
    - Detection bounds checking for s_e = s_0 +/- delta
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 2
    tolerance = 1.e-16

    def norm_indicator(expected, dcoll, soln, **kwargs):
        return (op.norm(dcoll, expected-smoothness_indicator(dcoll, soln, **kwargs),
                        np.inf))

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-1.0, )*dim,  b=(1.0, )*dim,  n=(nel_1d, ) * dim
    )

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    nele = mesh.nelements
    zeros = 0.0*nodes[0]

    # Test jump discontinuity
    soln = actx.np.where(nodes[0] > 0.0+zeros, 1.0+zeros, zeros)
    err = norm_indicator(1.0, dcoll, soln)
    assert err < tolerance,  "Jump discontinuity should trigger indicator (1.0)"

    # get meshmode polynomials
    group = dcoll.discr_from_dd("vol").groups[0]
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
        err = norm_indicator(expected, dcoll, soln)
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
    err = norm_indicator(1.0, dcoll, soln, s0=s0, kappa=0.0)
    assert err < tolerance,  (
        "A function with an indicator >s0 should trigger indicator")

    # Solution below s0
    ele_soln = ((phi_n_p-eps)*basis[n_p](unit_nodes)
                + phi_n_pm1*basis[n_pm1](unit_nodes))
    soln[0].set(np.tile(ele_soln, (nele, 1)))
    err = norm_indicator(0.0, dcoll, soln, s0=s0, kappa=0.0)
    assert err < tolerance, (
        "A function with an indicator <s0 should not trigger indicator")

    # Test kappa
    # non-perturbed solution
    # test middle value
    kappa = 0.5
    ele_soln = (phi_n_p*basis[n_p](unit_nodes)
                + phi_n_pm1*basis[n_pm1](unit_nodes))
    soln[0].set(np.tile(ele_soln, (nele, 1)))
    err = norm_indicator(0.5, dcoll, soln, s0=s0, kappa=kappa)
    assert err < 1.0e-10,  "A function with s_e=s_0 should return 0.5"

    # test bounds
    # lower bound
    shift = 1.0e-5
    err = norm_indicator(0.0, dcoll, soln, s0=s0+kappa+shift, kappa=kappa)
    assert err < tolerance,  "s_e<s_0-kappa should not trigger indicator"
    err = norm_indicator(0.0, dcoll, soln, s0=s0+kappa-shift, kappa=kappa)
    assert err > tolerance,  "s_e>s_0-kappa should trigger indicator"

    # upper bound
    err = norm_indicator(1.0, dcoll, soln, s0=s0-(kappa+shift), kappa=kappa)
    # s_e>s_0+kappa should fully trigger indicator (1.0)
    assert err < tolerance
    err = norm_indicator(1.0, dcoll, soln, s0=s0-(kappa-shift), kappa=kappa)
    # s_e<s_0+kappa should not fully trigger indicator (1.0)
    assert err > tolerance


@pytest.mark.parametrize("dim",  [1, 2, 3])
@pytest.mark.parametrize("order",  [2, 3])
def test_artificial_viscosity(ctx_factory, dim, order):
    """Test artificial_viscosity.

    Test AV operator for some test functions to verify artificial viscosity
    returns the analytical result.
    - 1d x^n (expected rhs = n*(n-1)*x^(n-2)
    - x^2 + y^2 + z^2  (expected rhs = 2*dim)
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 10
    tolerance = 1.e-6

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(1.0, )*dim, b=(2.0, )*dim, nelements_per_axis=(nel_1d, )*dim
    )

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    class TestBoundary:
        def cv_gradient_flux(self, dcoll, dd_bdry, state_minus, gas_model, **kwargs):
            cv_int = state_minus.cv
            from grudge.trace_pair import TracePair
            bnd_pair = TracePair(dd_bdry,
                                 interior=cv_int,
                                 exterior=cv_int)
            nhat = actx.thaw(dcoll.normal(dd_bdry))
            from mirgecom.flux import num_flux_central
            from arraycontext import outer
            # Do not project to "all_faces" as now we use built-in grad_cv_operator
            return outer(num_flux_central(bnd_pair.int, bnd_pair.ext), nhat)

        def av_flux(self, dcoll, dd_bdry, diffusion, **kwargs):
            nhat = actx.thaw(dcoll.normal(dd_bdry))
            diffusion_minus = op.project(dcoll, "vol", dd_bdry, diffusion)
            diffusion_plus = diffusion_minus
            from grudge.trace_pair import TracePair
            bnd_grad_pair = TracePair(dd_bdry, interior=diffusion_minus,
                                      exterior=diffusion_plus)
            from mirgecom.flux import num_flux_central
            flux_weak = num_flux_central(bnd_grad_pair.int, bnd_grad_pair.ext)@nhat
            return op.project(dcoll, dd_bdry, "all_faces", flux_weak)

    boundaries = {BTAG_ALL: TestBoundary()}

    # Up to quadratic 1d
    for nsol in range(3):
        soln = nodes[0]**nsol
        exp_rhs_1d = nsol * (nsol-1) * nodes[0]**(nsol - 2)
        print(f"{nsol=},{soln=},{exp_rhs_1d=}")
        cv = make_conserved(
            dim,
            mass=soln,
            energy=soln,
            momentum=make_obj_array([soln for _ in range(dim)]),
            species_mass=make_obj_array([soln for _ in range(dim)])
        )
        gas_model = GasModel(eos=IdealSingleGas())
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
        rhs = av_laplacian_operator(dcoll, boundaries=boundaries,
                                    gas_model=gas_model,
                                    fluid_state=fluid_state, alpha=1.0, s0=-np.inf)
        print(f"{rhs=}")
        err = op.norm(dcoll, rhs-exp_rhs_1d, np.inf)
        assert err < tolerance

    # Quadratic field return constant 2*dim
    soln = np.dot(nodes, nodes)
    cv = make_conserved(
        dim,
        mass=soln,
        energy=soln,
        momentum=make_obj_array([soln for _ in range(dim)]),
        species_mass=make_obj_array([soln for _ in range(dim)])
    )
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
    rhs = av_laplacian_operator(dcoll, boundaries=boundaries,
                                gas_model=gas_model,
                                fluid_state=fluid_state, alpha=1.0, s0=-np.inf)
    err = op.norm(dcoll, 2.*dim-rhs, np.inf)
    assert err < tolerance


@pytest.mark.parametrize("order",  [2, 3])
@pytest.mark.parametrize("dim",  [1, 2])
def test_trig(ctx_factory, dim, order):
    """Test artificial_viscosity.

    Test AV operator for some test functions to verify artificial viscosity
    returns the analytical result.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()
    nel_1d_base = 4

    for nfac in [1, 2, 4]:
        nel_1d = nfac * nel_1d_base

        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(0.0, )*dim, b=(1.0, )*dim, nelements_per_axis=(nel_1d, )*dim,
            periodic=(True,)*dim
        )

        dcoll = create_discretization_collection(actx, mesh, order=order)
        nodes = actx.thaw(dcoll.nodes())

        boundaries = {}

        # u = sin(2 pi x)sin(2 pi y)sin(2 pi z)
        # rhs  = -dim pi^2 u
        soln = 1.0
        exp_rhs = -dim * 4. * np.pi**2
        for w in nodes:
            soln = soln * actx.np.sin(w * 2 * np.pi)
            exp_rhs = exp_rhs * actx.np.sin(w * 2 * np.pi)

        gas_model = GasModel(eos=IdealSingleGas())
        cv = make_conserved(
            dim,
            mass=soln,
            energy=soln,
            momentum=make_obj_array([soln for _ in range(dim)]),
            species_mass=make_obj_array([soln for _ in range(dim)])
        )
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
        rhs = av_laplacian_operator(dcoll, boundaries=boundaries,
                                    gas_model=gas_model,
                                    fluid_state=fluid_state, alpha=1.0, s0=-np.inf)

        err_rhs = actx.to_numpy(op.norm(dcoll, rhs-exp_rhs, np.inf))
        eoc_rec.add_data_point(1.0/nel_1d, err_rhs)

    logger.info(
        f"{dim=}, {order=}"
        f"Errors:\n{eoc_rec}"
    )
    # For RHS-only checks we expect order - 1.
    expected_order = order - 1
    assert (
        eoc_rec.order_estimate() >= expected_order - .5
        or eoc_rec.max_error() < 1e-9
    )


class _VortexSoln:

    def __init__(self, **kwargs):
        self._dim = 2
        from mirgecom.initializers import Vortex2D
        origin = np.zeros(2)
        velocity = origin + 1
        self._initializer = Vortex2D(center=origin, velocity=velocity)

    def dim(self):
        return self._dim

    def __call__(self, r, eos, **kwargs):
        return self._initializer(x_vec=r, eos=eos)


class _TestSoln:

    def __init__(self, dim=2):
        self._dim = dim
        from mirgecom.initializers import Uniform
        origin = np.zeros(self._dim)
        velocity = origin + 1
        self._initializer = Uniform(dim=self._dim, velocity=velocity)

    def dim(self):
        return self._dim

    def __call__(self, r, eos, **kwargs):
        return self._initializer(x_vec=r, eos=eos)


# This test makes sure that the fluid boundaries work mechanically for AV,
# and give an expected result when using the AV boundary interface.
@pytest.mark.parametrize("prescribed_soln", [_TestSoln(dim=1),
                                             _TestSoln(dim=2),
                                             _TestSoln(dim=3)])
@pytest.mark.parametrize("order", [2, 3])
def test_fluid_av_boundaries(ctx_factory, prescribed_soln, order):
    """Check fluid boundary AV interface."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = prescribed_soln.dim()

    kappa = 3.0
    sigma = 5.0

    from mirgecom.transport import SimpleTransport
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa)

    gas_model = GasModel(eos=IdealSingleGas(gas_const=1.0),
                         transport=transport_model)

    def _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        return make_fluid_state(prescribed_soln(r=nodes, eos=gas_model.eos,
                                            **kwargs), gas_model)

    nels_geom = 16
    a = 1.0
    b = 2.0
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    cv = prescribed_soln(r=nodes, eos=gas_model.eos)
    fluid_state = make_fluid_state(cv, gas_model)

    boundary_nhat = actx.thaw(dcoll.normal(BTAG_ALL))

    prescribed_boundary = \
        PrescribedFluidBoundaryAV(boundary_state_func=_boundary_state_func)
    adiabatic_noslip = AdiabaticNoSlipWallAV()

    fluid_boundaries = {BTAG_ALL: prescribed_boundary}
    from mirgecom.navierstokes import grad_cv_operator
    fluid_grad_cv = \
        grad_cv_operator(dcoll, gas_model, fluid_boundaries, fluid_state)

    # Put in a testing field for AV - doesn't matter what - as long as it
    # is spatially-dependent in all dimensions.
    av_diffusion = 0. * fluid_grad_cv + np.dot(nodes, nodes)
    av_diffusion_boundary = op.project(dcoll, "vol", BTAG_ALL, av_diffusion)

    # Prescribed boundaries are used for inflow/outflow-type boundaries
    # where we expect to _preserve_ the soln gradient
    from grudge.dof_desc import as_dofdesc
    dd_bdry = as_dofdesc(BTAG_ALL)
    dd_allfaces = dd_bdry.with_boundary_tag(FACE_RESTR_ALL)
    expected_av_flux_prescribed_boundary = av_diffusion_boundary@boundary_nhat
    print(f"{expected_av_flux_prescribed_boundary=}")
    exp_av_flux = op.project(dcoll, dd_bdry, dd_allfaces,
                                expected_av_flux_prescribed_boundary)
    print(f"{exp_av_flux=}")

    prescribed_boundary_av_flux = \
        prescribed_boundary.av_flux(dcoll, BTAG_ALL, av_diffusion)
    print(f"{prescribed_boundary_av_flux=}")

    bnd_flux_resid = (prescribed_boundary_av_flux - exp_av_flux)
    print(f"{bnd_flux_resid=}")
    assert actx.np.equal(bnd_flux_resid, 0)

    # Solid wall boundaries are expected to have 0 AV flux
    wall_bnd_flux = \
        adiabatic_noslip.av_flux(dcoll, BTAG_ALL, av_diffusion)
    print(f"adiabatic_noslip: {wall_bnd_flux=}")
    assert actx.np.equal(wall_bnd_flux, 0)
