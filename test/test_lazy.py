__copyright__ = """Copyright (C) 2021 University of Illinois Board of Trustees"""

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
from functools import partial
from pytools.obj_array import make_obj_array, obj_array_vectorize_n_args
import pyopencl as cl
import pyopencl.tools as cl_tools
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath  # noqa
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from arraycontext.container.traversal import freeze, thaw

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest  # noqa

import logging
logger = logging.getLogger(__name__)


@pytest.fixture
def op_test_data(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    eager_actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    lazy_actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    def get_discr(order):
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2,
            b=(0.5,)*2,
            nelements_per_axis=(4,)*2,
            boundary_tag_to_face={
                "x": ["-x", "+x"],
                "y": ["-y", "+y"]
            })

        from grudge.eager import EagerDGDiscretization
        return EagerDGDiscretization(eager_actx, mesh, order=order)

    return eager_actx, lazy_actx, get_discr


def _rel_linf_error(discr, actual, expected):
    from mirgecom.fluid import ConservedVars
    if isinstance(actual, ConservedVars):
        return _rel_linf_error(discr, actual.join(), expected)
    if isinstance(expected, ConservedVars):
        return _rel_linf_error(discr, actual, expected.join())
    return np.max(obj_array_vectorize_n_args(
        lambda a, b: discr.norm(a - b, np.inf) / discr.norm(b, np.inf),
        actual, expected))


# FIXME: Re-enable and fix up if/when standalone gradient operator exists
# def test_lazy_op_gradient(ctx_factory):
#     cl_ctx = ctx_factory()
#     actx, discr = _op_test_fixture(cl_ctx)
#
#     from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE
#     from mirgecom.diffusion import (
#         _gradient_operator,
#         DirichletDiffusionBoundary,
#         NeumannDiffusionBoundary)
#
#     boundaries = {
#         DTAG_BOUNDARY("x"): DirichletDiffusionBoundary(0),
#         DTAG_BOUNDARY("y"): NeumannDiffusionBoundary(0)
#     }
#
#     def op(alpha, u):
#         return _gradient_operator(
#             discr, DISCR_TAG_BASE, alpha, boundaries, u)

#     compiled_op = actx.compile(op)
#     alpha = discr.zeros(actx) + 1
#     u = discr.zeros(actx)
#     compiled_op(alpha, u)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_lazy_op_divergence(op_test_data, order):
    eager_actx, lazy_actx, get_discr = op_test_data
    discr = get_discr(order)

    from grudge.trace_pair import interior_trace_pair
    from mirgecom.operators import div_operator

    def get_flux(u_tpair):
        dd = u_tpair.dd
        dd_allfaces = dd.with_dtag("all_faces")
        normal = thaw(discr.normal(dd), u_tpair.int[0].array_context)
        flux = u_tpair.avg @ normal
        return discr.project(dd, dd_allfaces, flux)

    def op(u):
        return div_operator(discr, u, get_flux(interior_trace_pair(discr, u)))

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        u = make_obj_array([actx.np.sin(np.pi*nodes[i]) for i in range(2)])
        return u,

    rel_linf_error = partial(_rel_linf_error, discr)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    eager_result = op(*get_inputs(eager_actx))
    lazy_result = lazy_to_eager(lazy_op(*get_inputs(lazy_actx)))
    assert rel_linf_error(lazy_result, eager_result) < 1e-12


@pytest.mark.parametrize("order", [1, 2, 3])
def test_lazy_op_diffusion(op_test_data, order):
    eager_actx, lazy_actx, get_discr = op_test_data
    discr = get_discr(order)

    from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE
    from mirgecom.diffusion import (
        diffusion_operator,
        DirichletDiffusionBoundary,
        NeumannDiffusionBoundary)

    boundaries = {
        DTAG_BOUNDARY("x"): DirichletDiffusionBoundary(0),
        DTAG_BOUNDARY("y"): NeumannDiffusionBoundary(0)
    }

    def op(alpha, u):
        return diffusion_operator(
            discr, DISCR_TAG_BASE, alpha, boundaries, u)

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        alpha = discr.zeros(actx) + 1
        u = actx.np.cos(np.pi*nodes[0])
        return alpha, u

    rel_linf_error = partial(_rel_linf_error, discr)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    eager_result = op(*get_inputs(eager_actx))
    lazy_result = lazy_to_eager(lazy_op(*get_inputs(lazy_actx)))
    assert rel_linf_error(lazy_result, eager_result) < 1e-12


@pytest.mark.parametrize("order", [1, 2, 3])
def test_lazy_op_euler(op_test_data, order):
    eager_actx, lazy_actx, get_discr = op_test_data
    discr = get_discr(order)

    from grudge.dof_desc import DTAG_BOUNDARY
    from mirgecom.eos import IdealSingleGas
    from mirgecom.boundary import AdiabaticSlipBoundary
    from mirgecom.euler import euler_operator

    eos = IdealSingleGas()

    boundaries = {
        DTAG_BOUNDARY("x"): AdiabaticSlipBoundary(),
        DTAG_BOUNDARY("y"): AdiabaticSlipBoundary()
    }

    def op(state):
        return euler_operator(discr, eos, boundaries, state)

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        from mirgecom.initializers import MulticomponentLump
        init = MulticomponentLump(
            dim=2, nspecies=3, velocity=np.ones(2), spec_y0s=np.ones(3),
            spec_amplitudes=np.ones(3))
        state = init(nodes)
        return state,

    rel_linf_error = partial(_rel_linf_error, discr)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    eager_result = op(*get_inputs(eager_actx))
    lazy_result = lazy_to_eager(lazy_op(*get_inputs(lazy_actx)))
    assert rel_linf_error(lazy_result, eager_result) < 1e-12


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
