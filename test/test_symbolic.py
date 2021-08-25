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
import pymbolic.primitives as prim
from meshmode.dof_array import thaw
from meshmode.mesh.generation import generate_regular_rect_mesh
import mirgecom.symbolic as sym

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest

import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("sym_f", "expected_sym_df"), [
    (1, 0),
    (pmbl.var("x")**2, 2*pmbl.var("x")),
    (pmbl.var("cos")(2*pmbl.var("x")), -pmbl.var("sin")(2*pmbl.var("x"))*2),
    (pmbl.var("sin")(2*pmbl.var("x")), pmbl.var("cos")(2*pmbl.var("x"))*2),
    (pmbl.var("exp")(2*pmbl.var("x")), pmbl.var("exp")(2*pmbl.var("x"))*2),
])
def test_symbolic_diff(sym_f, expected_sym_df):
    sym_df = sym.diff(pmbl.var("x"))(sym_f)
    assert sym_df == expected_sym_df


def test_symbolic_div():
    sym_coords = prim.make_sym_vector("x", 3)
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
    sym_coords = prim.make_sym_vector("x", 3)
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
    actx = actx_factory()

    mesh = generate_regular_rect_mesh(
        a=(-np.pi/2,)*2,
        b=(np.pi/2,)*2,
        nelements_per_axis=(4,)*2)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=2)

    nodes = thaw(actx, discr.nodes())

    sym_coords = prim.make_sym_vector("x", 2)

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
