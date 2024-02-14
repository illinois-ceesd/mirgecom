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
from grudge.dof_desc import as_dofdesc
from mirgecom.flux import num_flux_central
from mirgecom.fluid import (
    make_conserved
)
import mirgecom.symbolic as sym
import grudge.op as op
from grudge.trace_pair import interior_trace_pair
from mirgecom.discretization import create_discretization_collection
from functools import partial
from mirgecom.simutil import get_box_mesh
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


def _trig_test_func(dim):
    """Make trig test function.

    1d: cos(2pi x)
    2d: sin(2pi x)cos(2pi y)
    3d: sin(2pi x)sin(2pi y)cos(2pi z)
    """
    sym_coords = prim.make_sym_vector("x", dim)

    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    sym_result = sym_cos(2*np.pi*sym_coords[dim-1])
    for i in range(dim-1):
        sym_result = sym_result * sym_sin(2*np.pi*sym_coords[i])

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
    return op.project(dcoll, bnd_tpair.dd, dd_allfaces, flux_weak)


@pytest.mark.parametrize("tpe", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("sym_test_func_factory", [
    partial(_coord_test_func, order=0),
    partial(_coord_test_func, order=1),
    lambda dim: 2*_coord_test_func(dim, order=1),
    partial(_coord_test_func, order=2),
    _trig_test_func,
    _cv_test_func
])
def test_grad_operator(actx_factory, tpe, dim, order, sym_test_func_factory):
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

    # This comes from array_context
    actx = actx_factory()

    if tpe:  # TPE requires *grudge* array context, not array_context
        if dim == 1:  # TPE does not support dim=1
            pytest.skip()
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        actx = PyOpenCLArrayContext(queue)

    sym_test_func = sym_test_func_factory(dim)

    tol = 1e-10 if dim < 3 else 1e-9
    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for nfac in [1, 2, 4]:

        mesh = get_box_mesh(dim, a=0, b=1, n=nfac*3, tensor_product_elements=tpe)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        dcoll = create_discretization_collection(actx, mesh, order=order,
                                                 tensor_product_elements=tpe)

        # compute max element size
        from grudge.dt_utils import h_max_from_volume
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

        from mirgecom.simutil import componentwise_norms

        from arraycontext import flatten
        err_scale = max(flatten(componentwise_norms(dcoll, exact_grad, np.inf),
                                actx))

        if err_scale <= 1e-16:
            err_scale = 1

        print(f"{test_data=}")
        print(f"{exact_grad=}")

        test_data_int_tpair = interior_trace_pair(dcoll, test_data)
        boundaries = [BTAG_ALL]
        test_data_flux_bnd = _elbnd_flux(dcoll, int_flux, bnd_flux,
                                         test_data_int_tpair, boundaries)

        from mirgecom.operators import grad_operator
        dd_vol = as_dofdesc("vol")
        dd_allfaces = as_dofdesc("all_faces")
        test_grad = grad_operator(dcoll, dd_vol, dd_allfaces,
                                  test_data, test_data_flux_bnd)

        print(f"{test_grad=}")
        grad_err = \
            max(flatten(
                componentwise_norms(dcoll, test_grad - exact_grad, np.inf),
                actx) / err_scale)

        eoc.add_data_point(actx.to_numpy(h_max), actx.to_numpy(grad_err))

    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < tol
    )
