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
import pyopencl as cl
import pyopencl.tools as cl_tools
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath  # noqa
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from meshmode.dof_array import thaw

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest  # noqa

import logging
logger = logging.getLogger(__name__)


def _op_test_fixture(cl_ctx):
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    n = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,)*2,
        b=(0.5,)*2,
        nelements_per_axis=(n,)*2,
        boundary_tag_to_face={
            "x": ["-x", "+x"],
            "y": ["-y", "+y"]
        })

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=2)

    return actx, discr


def _rel_linf_error(discr, x, y):
    from mirgecom.fluid import ConservedVars
    if isinstance(x, ConservedVars):
        return _rel_linf_error(discr, x.join(), y)
    if isinstance(y, ConservedVars):
        return _rel_linf_error(discr, x, y.join())
    return discr.norm(x - y, np.inf) / discr.norm(y, np.inf)


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
#         DTAG_BOUNDARY("x"): DirichletDiffusionBoundary(0.),
#         DTAG_BOUNDARY("y"): NeumannDiffusionBoundary(0.)
#     }
#
#     def op(alpha, u):
#         return _gradient_operator(
#             discr, DISCR_TAG_BASE, alpha, boundaries, u)

#     compiled_op = actx.compile(op)
#     alpha = discr.zeros(actx) + 1
#     u = discr.zeros(actx)
#     compiled_op(alpha, u)


# FIXME: Re-enable and fix up if/when standalone divergence operator exists
# def test_lazy_op_divergence(ctx_factory):
#     cl_ctx = ctx_factory()
#     actx, discr = _op_test_fixture(cl_ctx)
#
#     from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE
#     from mirgecom.diffusion import (
#         _divergence_alpha_operator,
#         DirichletDiffusionBoundary,
#         NeumannDiffusionBoundary)
#
#     boundaries = {
#         DTAG_BOUNDARY("x"): DirichletDiffusionBoundary(0.),
#         DTAG_BOUNDARY("y"): NeumannDiffusionBoundary(0.)
#     }
#
#     def op(alpha, u):
#         return _divergence_alpha_operator(
#             discr, DISCR_TAG_BASE, alpha, boundaries, u)

#     compiled_op = actx.compile(op)
#     alpha = discr.zeros(actx) + 1
#     u = make_obj_array([discr.zeros(actx) for _ in range(2)])
#     compiled_op(alpha, u)


def test_lazy_op_diffusion(ctx_factory):
    cl_ctx = ctx_factory()
    actx, discr = _op_test_fixture(cl_ctx)
    nodes = thaw(actx, discr.nodes())

    from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE
    from mirgecom.diffusion import (
        diffusion_operator,
        DirichletDiffusionBoundary,
        NeumannDiffusionBoundary)

    boundaries = {
        DTAG_BOUNDARY("x"): DirichletDiffusionBoundary(0.),
        DTAG_BOUNDARY("y"): NeumannDiffusionBoundary(0.)
    }

    def op(alpha, u):
        return diffusion_operator(
            discr, DISCR_TAG_BASE, alpha, boundaries, u)

    compiled_op = actx.compile(op)

    alpha = discr.zeros(actx) + 1
    u = actx.np.cos(np.pi*nodes[0])

    rel_linf_error = partial(_rel_linf_error, discr)

    eager_result = op(alpha, u)
    lazy_result = compiled_op(alpha, u)
    assert rel_linf_error(lazy_result, eager_result) < 1e-12


def test_lazy_op_euler(ctx_factory):
    cl_ctx = ctx_factory()
    actx, discr = _op_test_fixture(cl_ctx)
    nodes = thaw(actx, discr.nodes())

    from grudge.dof_desc import DTAG_BOUNDARY
    from mirgecom.eos import IdealSingleGas
    from mirgecom.boundary import AdiabaticSlipBoundary
    from mirgecom.euler import euler_operator

    eos = IdealSingleGas()

    boundaries = {
        DTAG_BOUNDARY("x"): AdiabaticSlipBoundary(),
        DTAG_BOUNDARY("y"): AdiabaticSlipBoundary()
    }

    def op(cv):
        return euler_operator(discr, eos, boundaries, cv)

    compiled_op = actx.compile(op)

    from mirgecom.initializers import MulticomponentLump
    init = MulticomponentLump(
        dim=2, velocity=np.ones(2), spec_y0s=np.ones(3), spec_amplitudes=np.ones(3))
    state = init(nodes)

    rel_linf_error = partial(_rel_linf_error, discr)

    eager_result = op(state)
    lazy_result = compiled_op(state)
    assert rel_linf_error(lazy_result, eager_result) < 1e-12


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
