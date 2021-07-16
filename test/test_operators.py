"""Test the generic operator helper functions."""

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

from pytools.obj_array import make_obj_array, obj_array_vectorize  # noqa
from meshmode.dof_array import thaw
from mirgecom.fluid import split_conserved, join_conserved  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.symbolic import DTAG_BOUNDARY
from grudge.symbolic.primitives import DOFDesc
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from mirgecom.flux import central_scalar_flux
from mirgecom.operators import (
    dg_grad,
    element_local_grad,
    weak_grad,
)

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


# DG grad tester works only for continuous functions
def _vector_dg_grad(discr, q):
    ncomp = 1
    if isinstance(q, np.ndarray):
        actx = q[0].array_context
        ncomp = len(q)
    else:
        actx = q.array_context

    vol_part = weak_grad(discr, q)
    q_minus = discr.project("vol", "all_faces", q)
    dd = DOFDesc("all_faces")
    normal = thaw(actx, discr.normal(dd))
    if ncomp > 1:
        facial_flux = make_obj_array([q_minus[i]*normal for i in range(ncomp)])
    else:
        facial_flux = q_minus*normal
    return -discr.inverse_mass(vol_part - discr.face_mass(facial_flux))


# scalar flux - multiple disparate scalar components OK
def _my_scalar_flux(discr, trace_pair):
    if isinstance(trace_pair.int, np.ndarray):
        actx = trace_pair.int[0].array_context
    else:
        actx = trace_pair.int.array_context
    normal = thaw(actx, discr.normal(trace_pair.dd))
    my_flux = central_scalar_flux(trace_pair, normal)
    return discr.project(trace_pair.dd, "all_faces", my_flux)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_dg_gradient_of_scalar_function(actx_factory, dim):
    """Test that DG gradient produces expected results for scalar funcs."""
    actx = actx_factory()

    npts_geom = 17
    a = 1.0
    b = 2.0
    mesh = _get_box_mesh(dim=dim, a=a, b=b, n=npts_geom)
    boundaries = {}
    for i in range(dim):
        boundaries[DTAG_BOUNDARY("-"+str(i+1))] = 0
        boundaries[DTAG_BOUNDARY("+"+str(i+1))] = 0

    order = 3
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    tol = 1e-9

    def internal_flux(trace_pair):
        return _my_scalar_flux(discr, trace_pair)

    def boundary_flux(btag, u):
        u_minus = discr.project("vol", btag, u)
        normal = thaw(actx, discr.normal(btag))
        my_flux = u_minus*normal
        return discr.project(btag, "all_faces", my_flux)

    # Test gradient of scalar functions
    for test_component in range(dim):
        test_func = nodes[test_component]

        def bnd_flux(btag):
            return boundary_flux(btag, test_func)

        test_grad = dg_grad(discr, internal_flux, bnd_flux,
                            boundaries, test_func)
        cont_grad = element_local_grad(discr, test_func)
        print(f"{test_grad=}")
        print(f"{cont_grad=}")
        assert discr.norm(test_grad - cont_grad, np.inf) < tol

    for test_component in range(dim):
        test_func = actx.np.cos(nodes[test_component])

        def bnd_flux(btag):
            return boundary_flux(btag, test_func)

        test_grad = dg_grad(discr, internal_flux, bnd_flux,
                            boundaries, test_func)
        cont_grad = element_local_grad(discr, test_func)
        print(f"{test_grad=}")
        print(f"{cont_grad=}")
        assert discr.norm(test_grad - cont_grad, np.inf) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_dg_gradient_of_vector_function(actx_factory, dim):
    """Test that DG gradient produces expected results for vector funcs."""
    actx = actx_factory()

    npts_geom = 17
    a = 1.0
    b = 2.0
    mesh = _get_box_mesh(dim=dim, a=a, b=b, n=npts_geom)
    boundaries = {}
    for i in range(dim):
        boundaries[DTAG_BOUNDARY("-"+str(i+1))] = 0
        boundaries[DTAG_BOUNDARY("+"+str(i+1))] = 0

    order = 3
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    tol = 1e-9

    def internal_flux(trace_pair):
        return _my_scalar_flux(discr, trace_pair)

    def boundary_flux(btag, u):
        u_minus = discr.project("vol", btag, u)
        normal = thaw(actx, discr.normal(btag))
        ncomp = 1
        if isinstance(u, np.ndarray):
            ncomp = len(u)
        if ncomp > 1:
            my_flux = make_obj_array([u_minus[i]*normal for i in range(ncomp)])
        else:
            my_flux = u_minus*normal
        return discr.project(btag, "all_faces", my_flux)

    # Test gradient of vector functions
    test_func = nodes

    def bnd_flux(btag):
        return boundary_flux(btag, test_func)

    test_grad = dg_grad(discr, internal_flux, bnd_flux,
                            boundaries, test_func)

    # manually verified "right" answer given by discr.grad
    cont_grad = make_obj_array([element_local_grad(discr, nodes[i])
                                for i in range(dim)])
    print(f"{test_grad=}")
    print(f"{cont_grad=}")
    assert discr.norm(test_grad - cont_grad, np.inf) < tol

    test_func = actx.np.cos(nodes)
    test_grad = dg_grad(discr, internal_flux, bnd_flux,
                        boundaries, test_func)
    # manually verified "right" answer given by discr.grad
    cont_grad = make_obj_array([element_local_grad(discr, actx.np.cos(nodes[i]))
                                for i in range(dim)])
    print(f"{test_grad=}")
    print(f"{cont_grad=}")
    assert discr.norm(test_grad - cont_grad, np.inf) < tol
