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
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath # noqa
from pytools.obj_array import make_obj_array
import pymbolic as pmbl
from meshmode.dof_array import thaw
from meshmode.mesh.generation import generate_regular_rect_mesh
import mirgecom.symbolic as sym

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest

import logging
logger = logging.getLogger(__name__)


def _const_diff_pair():
    """Return a constant ($1$) and its derivative ($0$)."""
    return 1, 0


def _poly_diff_pair():
    """Return a polynomial ($x^2$) and its derivative ($2x$)."""
    sym_x = pmbl.var("x")
    return sym_x**2, 2*sym_x


def _cos_diff_pair():
    r"""Return a cosine function ($\cos(2x)$ and its derivative ($-2\sin(2x)$)."""
    sym_x = pmbl.var("x")
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    # Put the factor of 2 after to match how pymbolic computes the product rule
    return sym_cos(2*sym_x), -sym_sin(2*sym_x)*2


def _sin_diff_pair():
    r"""Return a sine function ($\sin(2x)$ and its derivative ($2\cos(2x)$)."""
    sym_x = pmbl.var("x")
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    # Put the factor of 2 after to match how pymbolic computes the product rule
    return sym_sin(2*sym_x), sym_cos(2*sym_x)*2


def _exp_diff_pair():
    r"""
    Return an exponential function ($\exp(2x)$ and its derivative ($2\exp(2x)$).
    """
    sym_x = pmbl.var("x")
    sym_exp = pmbl.var("exp")
    # Put the factor of 2 after to match how pymbolic computes the product rule
    return sym_exp(2*sym_x), sym_exp(2*sym_x)*2


@pytest.mark.parametrize(("sym_f", "expected_sym_df"), [
    _const_diff_pair(),
    _poly_diff_pair(),
    _cos_diff_pair(),
    _sin_diff_pair(),
    _exp_diff_pair(),
])
def test_symbolic_diff(sym_f, expected_sym_df):
    """
    Compute the symbolic derivative of an expression and compare it to an
    expected result.
    """
    sym_df = sym.diff(pmbl.var("x"))(sym_f)
    assert sym_df == expected_sym_df


def test_symbolic_div():
    """
    Compute the symbolic divergence of a vector expression and compare it to an
    expected result.
    """
    # (Equivalent to make_obj_array([pmbl.var("x")[i] for i in range(3)]))
    sym_coords = pmbl.make_sym_vector("x", 3)
    sym_x = sym_coords[0]
    sym_y = sym_coords[1]

    sym_f = make_obj_array([
        sym_x,
        sym_x * sym_y,
        sym_y])

    sym_div_f = sym.div(sym_f)
    expected_sym_div_f = 1 + sym_x + 0
    assert sym_div_f == expected_sym_div_f


def test_symbolic_grad():
    """
    Compute the symbolic gradient of an expression and compare it to an expected
    result.
    """
    sym_coords = pmbl.make_sym_vector("x", 3)
    sym_x = sym_coords[0]
    sym_y = sym_coords[1]

    sym_f = sym_x**2 * sym_y

    sym_grad_f = sym.grad(3, sym_f)
    expected_sym_grad_f = make_obj_array([
        sym_y * 2*sym_x,
        sym_x**2,
        0])
    assert (sym_grad_f == expected_sym_grad_f).all()


def test_symbolic_evaluation(actx_factory):
    """
    Evaluate a symbolic expression by plugging in numbers and
    :class:`~meshmode.dof_array.DOFArray`s and compare the result to the equivalent
    quantity computed explicitly.
    """
    actx = actx_factory()

    mesh = generate_regular_rect_mesh(
        a=(-np.pi/2,)*2,
        b=(np.pi/2,)*2,
        nelements_per_axis=(4,)*2)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=2)

    nodes = thaw(actx, discr.nodes())

    sym_coords = pmbl.make_sym_vector("x", 2)

    sym_f = (
        pmbl.var("exp")(-pmbl.var("t"))
        * pmbl.var("cos")(sym_coords[0])
        * pmbl.var("sin")(sym_coords[1]))

    t = 0.5

    f = sym.EvaluationMapper({"t": t, "x": nodes})(sym_f)

    expected_f = np.exp(-t) * actx.np.cos(nodes[0]) * actx.np.sin(nodes[1])

    assert discr.norm(f - expected_f)/discr.norm(expected_f) < 1e-12


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
