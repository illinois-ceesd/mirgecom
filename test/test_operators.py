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
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL
from mirgecom.flux import gradient_flux_central
from mirgecom.fluid import (
    ConservedVars,
    make_conserved
)
from grudge.eager import (
    EagerDGDiscretization,
    interior_trace_pair
)
from functools import partial

logger = logging.getLogger(__name__)


def _elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                int_tpair, boundaries):
    return (compute_interior_flux(int_tpair)
            + sum(compute_boundary_flux(btag) for btag in boundaries))


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None):
    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, npoints_per_axis=n, boundary_tag_to_face=bttf, mesh_type=t)


def _coord_test_func(actx=None, x_vec=None, order=1, fac=1.0, grad=False):
    """Make a coordinate-based test function or its gradient.

    Test Function
    -------------
    1d: fac*x^order
    2d: fac*x^order * y^order
    3d: fac*x^order * y^order * z^order

    Test Function Gradient
    ----------------------
    1d: fac*order*x^(order-1)
    2d: fac*order*<x^(order-1)*y^order, x^order*y^(order-1)>
    3d: fac*order*<x^(order-1)*y^order*z^order,
                   x^order*y^(order-1)*z^order,
                   x^order*y^order*z^(order-1)>
    """
    if x_vec is None:
        return 0

    dim = len(x_vec)
    if grad:
        ret_ary = fac*order*make_obj_array([0*x_vec[0]+1.0 for _ in range(dim)])
        for i in range(dim):
            for j in range(dim):
                termpow = (order - 1) if order and j == i else order
                ret_ary[i] = ret_ary[i] * (x_vec[j]**termpow)
    else:
        ret_ary = fac*(0*x_vec[0]+1.0)
        for i in range(dim):
            ret_ary = ret_ary * (x_vec[i]**order)

    return ret_ary


def _trig_test_func(actx=None, x_vec=None, grad=False):
    """Make trig test function or its gradient.

    Test Function
    -------------
    1d: cos(2pi x)
    2d: sin(2pi x)cos(2pi y)
    3d: sin(2pi x)sin(2pi y)cos(2pi x)

    Grad Test Function
    ------------------
    1d: 2pi * -sin(2pi x)
    2d: 2pi * <cos(2pi x)cos(2pi y), -sin(2pi x)sin(2pi y)>
    3d: 2pi * <cos(2pi x)sin(2pi y)cos(2pi z),
               sin(2pi x)cos(2pi y)cos(2pi z),
               -sin(2pi x)sin(2pi y)sin(2pi z)>
    """
    if x_vec is None:
        return 0
    dim = len(x_vec)
    if grad:
        ret_ary = make_obj_array([0*x_vec[0] + 1.0 for _ in range(dim)])
        for i in range(dim):  # component & derivative for ith dir
            for j in range(dim):  # form term for jth dir in ith component
                if j == i:  # then this is a derivative term
                    if j == (dim-1):  # deriv of cos term
                        ret_ary[i] = ret_ary[i] * -actx.np.sin(2*np.pi*x_vec[j])
                    else:  # deriv of sin term
                        ret_ary[i] = ret_ary[i] * actx.np.cos(2*np.pi*x_vec[j])
                    ret_ary[i] = 2*np.pi*ret_ary[i]
                else:  # non-derivative term
                    if j == (dim-1):  # cos term
                        ret_ary[i] = ret_ary[i] * actx.np.cos(2*np.pi*x_vec[j])
                    else:  # sin term
                        ret_ary[i] = ret_ary[i] * actx.np.sin(2*np.pi*x_vec[j])
    else:
        # return _make_trig_term(actx, r=x_vec, term=dim-1)
        ret_ary = actx.np.cos(2*np.pi*x_vec[dim-1])
        for i in range(dim-1):
            ret_ary = ret_ary * actx.np.sin(2*np.pi*x_vec[i])

    return ret_ary


def _cv_test_func(actx, x_vec, grad=False):
    """Make a CV array container for testing.

    There is no need for this CV to be physical, we are just testing whether the
    operator machinery still gets the right answers when operating on a CV array
    container.

    Testing CV
    ----------
    mass = constant
    energy = _trig_test_func
    momentum = <_coord_test_func(order=2, fac=dim_index+1)>

    Testing CV Gradient
    -------------------
    mass = <0>
    energy = <_trig_test_func(grad=True)>
    momentum = <_coord_test_func(grad=True)>
    """
    zeros = 0*x_vec[0]
    dim = len(x_vec)
    momentum = make_obj_array([zeros+1.0 for _ in range(dim)])
    if grad:
        dm = make_obj_array([zeros+0.0 for _ in range(dim)])
        de = _trig_test_func(actx, x_vec, grad=True)
        dp = make_obj_array([np.empty(0) for _ in range(dim)])
        dy = (
            momentum * make_obj_array([np.empty(0) for _ in range(0)]).reshape(-1, 1)
        )
        for i in range(dim):
            dp[i] = _coord_test_func(actx, x_vec, order=2, fac=(i+1), grad=True)
        return make_conserved(dim=dim, mass=dm, energy=de, momentum=np.stack(dp),
                              species_mass=dy)
    else:
        mass = zeros + 1.0
        energy = _trig_test_func(actx, x_vec)
        for i in range(dim):
            momentum[i] = _coord_test_func(actx, x_vec, order=2, fac=(i+1))
        return make_conserved(dim=dim, mass=mass, energy=energy, momentum=momentum)


def central_flux_interior(actx, discr, int_tpair):
    """Compute a central flux for interior faces."""
    normal = thaw(actx, discr.normal(int_tpair.dd))
    flux_weak = gradient_flux_central(int_tpair, normal)
    return discr.project(int_tpair.dd, "all_faces", flux_weak)


def central_flux_boundary(actx, discr, soln_func, btag):
    """Compute a central flux for boundary faces."""
    boundary_discr = discr.discr_from_dd(btag)
    bnd_nodes = thaw(actx, boundary_discr.nodes())
    soln_bnd = soln_func(actx=actx, x_vec=bnd_nodes)
    bnd_nhat = thaw(actx, discr.normal(btag))
    from grudge.trace_pair import TracePair
    bnd_tpair = TracePair(btag, interior=soln_bnd, exterior=soln_bnd)
    flux_weak = gradient_flux_central(bnd_tpair, bnd_nhat)
    return discr.project(bnd_tpair.dd, "all_faces", flux_weak)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("test_func", [partial(_coord_test_func, order=0),
                                       partial(_coord_test_func, order=1),
                                       partial(_coord_test_func, order=1, fac=2),
                                       partial(_coord_test_func, order=2),
                                       _trig_test_func,
                                       _cv_test_func])
def test_grad_operator(actx_factory, dim, order, test_func):
    """Test the gradient operator for sanity.

    Check whether we get the right answers for gradients of analytic functions with
    some simple input fields and states:
    - constant
    - multilinear funcs
    - quadratic funcs
    - trig funcs
    - :class:`~mirgecom.fluid.ConservedVars` composed of funcs from above
    """
    actx = actx_factory()

    tol = 1e-10 if dim < 3 else 1e-9
    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for nfac in [1, 2, 4]:

        npts_axis = (nfac*4,)*dim
        box_ll = (0,)*dim
        box_ur = (1,)*dim
        mesh = _get_box_mesh(dim, a=box_ll, b=box_ur, n=npts_axis)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        # compute max element size
        from grudge.dt_utils import h_max_from_volume
        h_max = h_max_from_volume(discr)

        nodes = thaw(actx, discr.nodes())
        int_flux = partial(central_flux_interior, actx, discr)
        bnd_flux = partial(central_flux_boundary, actx, discr, test_func)

        test_data = test_func(actx=actx, x_vec=nodes)
        exact_grad = test_func(actx=actx, x_vec=nodes, grad=True)

        if isinstance(test_data, ConservedVars):
            err_scale = discr.norm(exact_grad.join(), np.inf)
        else:
            err_scale = discr.norm(exact_grad, np.inf)
        if err_scale <= 1e-16:
            err_scale = 1

        print(f"{test_data=}")
        print(f"{exact_grad=}")

        test_data_int_tpair = interior_trace_pair(discr, test_data)
        boundaries = [BTAG_ALL]
        test_data_flux_bnd = _elbnd_flux(discr, int_flux, bnd_flux,
                                         test_data_int_tpair, boundaries)

        from mirgecom.operators import grad_operator
        if isinstance(test_data, ConservedVars):
            test_grad = make_conserved(
                dim=dim, q=grad_operator(discr, test_data.join(),
                                         test_data_flux_bnd.join())
                )
            grad_err = discr.norm((test_grad - exact_grad).join(), np.inf)/err_scale
        else:
            test_grad = grad_operator(discr, test_data, test_data_flux_bnd)
            grad_err = discr.norm(test_grad - exact_grad, np.inf)/err_scale

        print(f"{test_grad=}")
        eoc.add_data_point(h_max, grad_err)

    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < tol
    )
