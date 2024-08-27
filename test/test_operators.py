"""Test the operator module for sanity."""

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

import numpy as np  # noqa
import pytest  # noqa
import logging
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
)
from pytools.obj_array import make_obj_array
import pymbolic as pmbl  # noqa
import pymbolic.primitives as prim
from meshmode.mesh import BTAG_ALL
# from grudge.dof_desc import as_dofdesc
# from grudge.dt_utils import characteristic_lengthscales
from mirgecom.flux import num_flux_central
from mirgecom.fluid import (
    make_conserved
)
import mirgecom.symbolic as sym
import grudge.op as op
from grudge.trace_pair import interior_trace_pair, bv_trace_pair
from mirgecom.discretization import create_discretization_collection
from functools import partial
from mirgecom.simutil import get_box_mesh
from grudge import geometry
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    DISCR_TAG_QUAD,
    DTAG_VOLUME_ALL,
    FACE_RESTR_ALL,
    VTAG_ALL,
    BoundaryDomainTag,
    as_dofdesc,
)
logger = logging.getLogger(__name__)


def _elbnd_flux(dcoll, compute_interior_flux, compute_boundary_flux,
                int_tpair, boundaries):
    return (
        compute_interior_flux(int_tpair)
        + sum(compute_boundary_flux(as_dofdesc(bdtag)) for bdtag in boundaries))


def _coord_test_func(dim, order=1):
    """Make a coordinate-based test function.

    1d: x^order
    2d: x^order * y^order
    3d: x^order * y^order * z^order
    """
    sym_coords = prim.make_sym_vector("x", dim)

    sym_result = 1
    for i in range(dim):
        sym_result *= sym_coords[i]**order

    return sym_result


def _poly_test_func(x_vec, order=1, a=None):
    """Make a coordinate-based polynomial test function.

    1d: x^order
    2d: x^order + y^order
    3d: x^order + y^order + z^order
    """
    dim = len(x_vec)
    if a is None:
        a = 1
    if np.isscalar(a):
        a = make_obj_array([a for _ in range(dim)])
    if len(a) != dim:
        raise ValueError("Coefficients *a* have wrong dimension.")

    result = 0
    for i in range(dim):
        result += a[i]*x_vec[i]**order

    return result


def _dpoly_test_func(x_vec, order=1, a=None):
    """Make derivative of a coordinate-based polynomial test function.

    1d: x^order
    2d: x^order + y^order
    3d: x^order + y^order + z^order
    """
    dim = len(x_vec)
    if a is None:
        a = 1
    if np.isscalar(a):
        a = make_obj_array([a for _ in range(dim)])
    if len(a) != dim:
        raise ValueError("Coefficients *a* have wrong dimension.")
    if order == 0:
        return 0*x_vec

    return make_obj_array([order * a[i] * x_vec[i]**(order-1)
                           for i in range(dim)])


def _ipoly_test_func(x_vec, order=1, a=None):
    """Make a coordinate-based polynomial test function.

    1d: x^order
    2d: x^order + y^order
    3d: x^order + y^order + z^order
    """
    dim = len(x_vec)
    if a is None:
        a = 1
    if np.isscalar(a):
        a = make_obj_array([a for _ in range(dim)])
    if len(a) != dim:
        raise ValueError("Coefficients *a* have wrong dimension.")

    result = 0
    for i in range(dim):
        term = 1.
        for j in range(dim):
            if i == j:
                term = term * (1./(order+1))*a[i]*x_vec[i]**(order+1)
            else:
                term = term * x_vec[i]
        result = result + term

    return result


def _trig_test_func(dim):
    """Make trig test function.

    1d: cos(pi x/2)
    2d: sin(pi x/2)cos(pi y/2)
    3d: sin(pi x/2)sin(pi y/2)cos(pi z/2)
    """
    sym_coords = prim.make_sym_vector("x", dim)

    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    sym_result = sym_cos(np.pi*sym_coords[dim-1]/2)
    for i in range(dim-1):
        sym_result = sym_result * sym_sin(np.pi*sym_coords[i]/2)

    return sym_result


def _cv_test_func(dim):
    """Make a CV array container for testing.

    There is no need for this CV to be physical, we are just testing whether the
    operator machinery still gets the right answers when operating on a CV array
    container.

    mass = constant
    energy = _trig_test_func
    momentum = <(dim_index+1)*_coord_test_func(order=2)>
    """
    sym_mass = 1
    sym_energy = _trig_test_func(dim)
    sym_momentum = make_obj_array([
        (i+1)*_coord_test_func(dim, order=2)
        for i in range(dim)])
    return make_conserved(
        dim=dim, mass=sym_mass, energy=sym_energy, momentum=sym_momentum)


def central_flux_interior(actx, dcoll, int_tpair):
    """Compute a central flux for interior faces."""
    normal = actx.thaw(dcoll.normal(int_tpair.dd))
    from arraycontext import outer
    flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
    dd_allfaces = int_tpair.dd.with_dtag("all_faces")
    return op.project(dcoll, int_tpair.dd, dd_allfaces, flux_weak)


def central_flux_boundary(actx, dcoll, soln_func, dd_bdry):
    """Compute a central flux for boundary faces."""
    boundary_discr = dcoll.discr_from_dd(dd_bdry)
    bnd_nodes = actx.thaw(boundary_discr.nodes())
    soln_bnd = soln_func(x_vec=bnd_nodes)
    bnd_nhat = actx.thaw(dcoll.normal(dd_bdry))
    from grudge.trace_pair import TracePair
    bnd_tpair = TracePair(dd_bdry, interior=soln_bnd, exterior=soln_bnd)
    from arraycontext import outer
    flux_weak = outer(num_flux_central(bnd_tpair.int, bnd_tpair.ext), bnd_nhat)
    dd_allfaces = bnd_tpair.dd.with_dtag("all_faces")
    flux = op.project(dcoll, bnd_tpair.dd, dd_allfaces, flux_weak)
    return flux


@pytest.mark.parametrize(("dim", "mesh_name", "rot_axis", "wonk"),
                         [
                             (1, "tet_box1", None, False),
                             (2, "tet_box2", None, False),
                             (3, "tet_box3", None, False),
                             (2, "hex_box2", None, False),
                             (3, "hex_box3", None, False),
                             (2, "tet_box2_rot", np.array([0, 0, 1]), False),
                             (3, "tet_box3_rot1", np.array([0, 0, 1]), False),
                             (3, "tet_box3_rot2", np.array([0, 1, 1]), False),
                             (3, "tet_box3_rot3", np.array([1, 1, 1]), False),
                             (2, "hex_box2_rot", np.array([0, 0, 1]), False),
                             (3, "hex_box3_rot1", np.array([0, 0, 1]), False),
                             (3, "hex_box3_rot2", np.array([0, 1, 1]), False),
                             (3, "hex_box3_rot3", np.array([1, 1, 1]), False),
                             ])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("sym_test_func_factory", [
    partial(_coord_test_func, order=0),
    partial(_coord_test_func, order=1),
    lambda dim: 2*_coord_test_func(dim, order=1),
    partial(_coord_test_func, order=2),
    _trig_test_func,
    _cv_test_func
])
def test_grad_operator(actx_factory, dim, mesh_name, rot_axis, wonk,
                       order, sym_test_func_factory):
    """Test the gradient operator for sanity.

    Check whether we get the right answers for gradients of analytic functions with
    some simple input fields and states:
    - constant
    - multilinear funcs
    - quadratic funcs
    - trig funcs
    - :class:`~mirgecom.fluid.ConservedVars` composed of funcs from above
    """
    import pyopencl as cl
    from grudge.array_context import PyOpenCLArrayContext
    from meshmode.mesh.processing import rotate_mesh_around_axis
    from grudge.dt_utils import h_max_from_volume
    from mirgecom.simutil import componentwise_norms
    from arraycontext import flatten
    from mirgecom.operators import grad_operator
    from meshmode.mesh.processing import map_mesh

    def add_wonk(x: np.ndarray) -> np.ndarray:
        wonk_field = np.empty_like(x)
        if len(x) >= 2:
            wonk_field[0] = (
                1.5*x[0] + np.cos(x[0])
                + 0.1*np.sin(10*x[1]))
            wonk_field[1] = (
                0.05*np.cos(10*x[0])
                + 1.3*x[1] + np.sin(x[1]))
        else:
            wonk_field[0] = 1.5*x[0] + np.cos(x[0])

        if len(x) >= 3:
            wonk_field[2] = x[2] + np.sin(x[0] / 2) / 2

        return wonk_field

    tpe = mesh_name.startswith("hex_")
    rotation_angle = 32.0
    theta = rotation_angle/180.0 * np.pi

    # This comes from array_context
    actx = actx_factory()

    if tpe:  # TPE requires *grudge* array context, not array_context
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        actx = PyOpenCLArrayContext(queue)

    sym_test_func = sym_test_func_factory(dim)

    tol = 1e-10 if dim < 3 else 2e-9

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for nfac in [2, 4, 8]:

        # Make non-uniform spacings
        b = tuple((2 ** n) for n in range(dim))

        mesh = get_box_mesh(dim, a=0, b=b, n=nfac*3, tensor_product_elements=tpe)
        if wonk:
            mesh = map_mesh(mesh, add_wonk)
        if rot_axis is not None:
            mesh = rotate_mesh_around_axis(mesh, theta=theta, axis=rot_axis)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        dcoll = create_discretization_collection(actx, mesh, order=order)

        # compute max element size
        h_max = h_max_from_volume(dcoll)

        def sym_eval(expr, x_vec):
            mapper = sym.EvaluationMapper({"x": x_vec})
            from arraycontext import rec_map_array_container
            return rec_map_array_container(
                # If expressions don't depend on coords (e.g., order 0), evaluated
                # result will be scalar-valued, so promote to DOFArray
                lambda comp_expr: mapper(comp_expr) + 0*x_vec[0],
                expr)

        test_func = partial(sym_eval, sym_test_func)
        grad_test_func = partial(sym_eval, sym.grad(dim, sym_test_func))

        nodes = actx.thaw(dcoll.nodes())
        int_flux = partial(central_flux_interior, actx, dcoll)
        bnd_flux = partial(central_flux_boundary, actx, dcoll, test_func)

        test_data = test_func(nodes)
        exact_grad = grad_test_func(nodes)
        # print(f"{test_data=}")
        # print(f"{exact_grad=}")

        err_scale = max(flatten(componentwise_norms(dcoll, exact_grad, np.inf),
                                actx))

        print(f"{err_scale=}")
        if err_scale <= 1e-10:
            err_scale = 1
            print(f"Rescaling: {err_scale=}")

        # print(f"{test_data=}")
        # print(f"{exact_grad=}")

        test_data_int_tpair = interior_trace_pair(dcoll, test_data)
        boundaries = [BTAG_ALL]
        test_data_flux_bnd = _elbnd_flux(dcoll, int_flux, bnd_flux,
                                         test_data_int_tpair, boundaries)

        dd_vol = as_dofdesc("vol")
        dd_allfaces = as_dofdesc("all_faces")
        test_grad = grad_operator(dcoll, dd_vol, dd_allfaces,
                                  test_data, test_data_flux_bnd)

        # print(f"{test_grad=}")
        grad_err = \
            max(flatten(
                componentwise_norms(dcoll, test_grad - exact_grad, np.inf),
                actx) / err_scale)
        # print(f"{actx.to_numpy(h_max)=}\n{actx.to_numpy(grad_err)=}")

        eoc.add_data_point(actx.to_numpy(h_max), actx.to_numpy(grad_err))

    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < tol
    )

#                             (2, "tet_box2_rot", np.array([0, 0, 1]), False),
#                             (3, "tet_box3_rot1", np.array([0, 0, 1]), False),
#                             (3, "tet_box3_rot2", np.array([0, 1, 1]), False),
#                             (3, "tet_box3_rot3", np.array([1, 1, 1]), False),
#                             (2, "hex_box2_rot", np.array([0, 0, 1]), False),
#                             (3, "hex_box3_rot1", np.array([0, 0, 1]), False),
#                             (3, "hex_box3_rot2", np.array([0, 1, 1]), False),
#                             (3, "hex_box3_rot3", np.array([1, 1, 1]), False),


#@pytest.mark.parametrize(("dim", "mesh_name", "rot_axis", "wonk"),
#                         [
#                             (1, "tet_box1", None, False),
#                             (2, "tet_box2", None, False),
#                             (3, "tet_box3", None, False),
#                             (2, "hex_box2", None, False),
#                             (3, "hex_box3", None, False),
#                             ])
@pytest.mark.parametrize(("dim", "mesh_name", "rot_axis", "wonk"),
                         [
                             (1, "tet_box1", None, False),])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("overint", [False, True])
def test_overintegration(actx_factory, dim, mesh_name, rot_axis, wonk, order,
                         overint):
    """Test overintegration with grad operator.

    Check whether we get the right answers for gradients of polynomial
    expressions up to quadrature order P', and proper error behavior for
    expressions of order higher than P'.
    - P <= P'
    - P > P'
    """
    import pyopencl as cl
    from grudge.array_context import PyOpenCLArrayContext
    from meshmode.mesh.processing import rotate_mesh_around_axis
    from grudge.dt_utils import h_min_from_volume
    from mirgecom.simutil import componentwise_norms
    from arraycontext import flatten
    from mirgecom.operators import grad_operator
    from meshmode.mesh.processing import map_mesh

    def add_wonk(x: np.ndarray) -> np.ndarray:
        wonk_field = np.empty_like(x)
        if len(x) >= 2:
            wonk_field[0] = (
                1.5*x[0] + np.cos(x[0])
                + 0.1*np.sin(10*x[1]))
            wonk_field[1] = (
                0.05*np.cos(10*x[0])
                + 1.3*x[1] + np.sin(x[1]))
        else:
            wonk_field[0] = 1.5*x[0] + np.cos(x[0])

        if len(x) >= 3:
            wonk_field[2] = x[2] + np.sin(x[0] / 2) / 2

        return wonk_field

    p = order
    p_prime = 2*p + 1
    print(f"{mesh_name=}")
    print(f"{dim=}, {p=}, {p_prime=}")
    # print(f"{p=},{p_prime=}")

    tpe = mesh_name.startswith("hex_")
    # print(f"{tpe=}")
    rotation_angle = 32.0
    theta = rotation_angle/180.0 * np.pi

    # This comes from array_context
    actx = actx_factory()

    if tpe:  # TPE requires *grudge* array context, not array_context
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        actx = PyOpenCLArrayContext(queue)

    tol = 1e-11 if tpe else 5e-12
    test_passed = True

    for f_order in range(p_prime+1):
        # print(f"{f_order=}")
        test_func = partial(_poly_test_func, order=f_order)
        grad_test_func = partial(_dpoly_test_func, order=f_order)
        integ_test_func = partial(_ipoly_test_func, order=f_order)

        a = (-1,)*dim
        b = (1,)*dim

        # Make non-uniform spacings
        # b = tuple((2 ** n) for n in range(dim))

        mesh = get_box_mesh(dim, a=a, b=b, n=8, tensor_product_elements=tpe)
        if wonk:
            mesh = map_mesh(mesh, add_wonk)
        if rot_axis is not None:
            mesh = rotate_mesh_around_axis(mesh, theta=theta, axis=rot_axis)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        dcoll = create_discretization_collection(
            actx, mesh, order=p, quadrature_order=p_prime)

        # compute min element size
        h_min = actx.to_numpy(h_min_from_volume(dcoll))[()]
        # lenscale = actx.to_numpy(min(flatten(
        #    characteristic_lengthscales(actx, dcoll), actx)))

        if overint:
            quad_discr_tag = DISCR_TAG_QUAD
        else:
            quad_discr_tag = DISCR_TAG_BASE

        allfaces_dd_quad = as_dofdesc(FACE_RESTR_ALL, quad_discr_tag)
        vol_dd_base = as_dofdesc(DTAG_VOLUME_ALL)
        vol_dd_quad = vol_dd_base.with_discr_tag(quad_discr_tag)
        bdry_dd_base = as_dofdesc(BTAG_ALL)
        bdry_dd_quad = bdry_dd_base.with_discr_tag(quad_discr_tag)

        x_base = actx.thaw(dcoll.nodes(dd=vol_dd_base))
        bdry_x_base = actx.thaw(dcoll.nodes(bdry_dd_base))

        def get_flux(u_tpair, dcoll=dcoll):
            dd = u_tpair.dd
            dd_allfaces = dd.with_domain_tag(
                BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL)
                )
            normal = geometry.normal(actx, dcoll, dd)
            u_avg = u_tpair.avg
            flux = u_avg * normal
            return op.project(dcoll, dd, dd_allfaces, flux)

        test_tol = tol * dim * p / h_min

        # nodes = actx.thaw(dcoll.nodes())
        # print(f"{nodes=}")
        # int_flux = partial(central_flux_interior, actx, dcoll)
        # bnd_flux = partial(central_flux_boundary, actx, dcoll, test_func)

        u_base = test_func(x_vec=x_base)
        u_bnd_base = test_func(x_vec=bdry_x_base)
        u_quad = op.project(dcoll, vol_dd_base, vol_dd_quad, u_base)

        u_bnd_flux_quad = (
            sum(get_flux(op.project_tracepair(dcoll, allfaces_dd_quad, itpair))
                for itpair in op.interior_trace_pairs(dcoll, u_base,
                                                      volume_dd=vol_dd_base))
            + get_flux(op.project_tracepair(
                dcoll, bdry_dd_quad,
                bv_trace_pair(dcoll, bdry_dd_base, u_base, u_bnd_base)))
        )

        exact_grad = grad_test_func(x_vec=x_base)
        exact_integ = integ_test_func(x_vec=x_base)

        err_scale = actx.to_numpy(
            max(flatten(componentwise_norms(dcoll, exact_grad, np.inf),
                                              actx)))[()]
        ierr_scale = actx.to_numpy(
            max(flatten(componentwise_norms(dcoll, exact_integ, np.inf),
                                              actx)))[()]
        # print(f"{err_scale=}")
        if err_scale <= test_tol:
            err_scale = 1
            print(f"Rescaling: {err_scale=}")
        if ierr_scale <= test_tol:
            ierr_scale = 1
            print(f"Rescaling: {ierr_scale=}")

        # print(f"{actx.to_numpy(test_data)=}")
        # print(f"{actx.to_numpy(exact_grad)=}")

        # test_data_int_tpair = op.project_tracepair(
        #    dcoll, allfaces_dd_quad, interior_trace_pair(dcoll, u_base))

        # boundaries = [BTAG_ALL]
        # test_data_flux_bnd = _elbnd_flux(dcoll, int_flux, bnd_flux,
        #                                 test_data_int_tpair, boundaries)
        # ubfq = actx.to_numpy(u_bnd_flux_quad)
        # tdfb = actx.to_numpy(test_data_flux_bnd)
        # print(f"{ubfq=}")
        # print(f"{tdfb=}")

        # flux_diff = u_bnd_flux_quad - test_data_flux_bnd
        # flux_diff = actx.to_numpy(flux_diff)
        # print(f"{flux_diff=}")
        # assert False
        # dd_vol = as_dofdesc("vol")
        # dd_allfaces = as_dofdesc("all_faces")
        # local_grad = op.weak_local_grad(dcoll, dd_vol, test_data)
        # print(f"{actx.to_numpy(local_grad)=}")
        test_grad = grad_operator(dcoll, vol_dd_quad, allfaces_dd_quad,
                                  u_quad, u_bnd_flux_quad)
        test_integ_base = op.elementwise_integral(dcoll, vol_dd_base, u_base)
        test_integ_quad = op.elementwise_integral(dcoll, vol_dd_quad, u_quad)

        print(f"{actx.to_numpy(test_integ_base)=}")
        print(f"{actx.to_numpy(test_integ_quad)=}")
        # print(f"{actx.to_numpy(test_grad)=}")

        grad_err = \
            max(flatten(
                componentwise_norms(dcoll, test_grad - exact_grad, np.inf),
                actx) / err_scale)
        grad_err = actx.to_numpy(grad_err)
        print(f"{p=},{h_min=},{f_order=},{grad_err=},{test_tol=}")
        this_test = grad_err < test_tol

        print(f"{test_passed=}, {overint=}")
        # ensure it fails with no overintegration for function order > p
        if f_order > p and not overint:
            this_test = not this_test

        test_passed = test_passed and this_test
    assert test_passed
