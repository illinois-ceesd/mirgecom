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


def _const_deriv_pair():
    """Return a constant ($1$) and its derivative ($0$)."""
    return 1, 0


def _poly_deriv_pair():
    """Return a polynomial ($x^2$) and its derivative ($2x$)."""
    sym_x = pmbl.var("x")
    return sym_x**2, 2*sym_x


def _cos_deriv_pair():
    r"""Return a cosine function ($\cos(2x)$ and its derivative ($-2\sin(2x)$)."""
    sym_x = pmbl.var("x")
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    # Put the factor of 2 after to match how pymbolic computes the product rule
    return sym_cos(2*sym_x), -sym_sin(2*sym_x)*2


def _sin_deriv_pair():
    r"""Return a sine function ($\sin(2x)$ and its derivative ($2\cos(2x)$)."""
    sym_x = pmbl.var("x")
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    # Put the factor of 2 after to match how pymbolic computes the product rule
    return sym_sin(2*sym_x), sym_cos(2*sym_x)*2


def _exp_deriv_pair():
    r"""
    Return an exponential function ($\exp(2x)$ and its derivative ($2\exp(2x)$).
    """
    sym_x = pmbl.var("x")
    sym_exp = pmbl.var("exp")
    # Put the factor of 2 after to match how pymbolic computes the product rule
    return sym_exp(2*sym_x), sym_exp(2*sym_x)*2


def _obj_array_deriv_pair():
    """
    Return a pair of object arrays containing expressions and their derivatives.
    """
    expr_deriv_pairs = [
        _const_deriv_pair(),
        _poly_deriv_pair(),
        _cos_deriv_pair()]
    return (
        make_obj_array([expr for expr, _ in expr_deriv_pairs]),
        make_obj_array([deriv for _, deriv in expr_deriv_pairs]))


def _array_container_deriv_pair():
    """
    Return a pair of array containers containing expressions and their derivatives.

    Returns
    -------
    A pair of :class:`mirgecom.fluid.ConservedVars` instances.
    """
    expr_deriv_pairs = [
        _const_deriv_pair(),
        _poly_deriv_pair(),
        _cos_deriv_pair(),
        _sin_deriv_pair()]
    from mirgecom.fluid import make_conserved
    return (
        make_conserved(
            dim=2,
            mass=expr_deriv_pairs[0][0],
            energy=expr_deriv_pairs[1][0],
            momentum=make_obj_array([
                expr_deriv_pairs[2][0],
                expr_deriv_pairs[3][0]])),
        make_conserved(
            dim=2,
            mass=expr_deriv_pairs[0][1],
            energy=expr_deriv_pairs[1][1],
            momentum=make_obj_array([
                expr_deriv_pairs[2][1],
                expr_deriv_pairs[3][1]])))


@pytest.mark.parametrize(("sym_f", "expected_sym_df"), [
    _const_deriv_pair(),
    _poly_deriv_pair(),
    _cos_deriv_pair(),
    _sin_deriv_pair(),
    _exp_deriv_pair(),
    _obj_array_deriv_pair(),
    _array_container_deriv_pair(),
])
def test_symbolic_diff(sym_f, expected_sym_df):
    """
    Compute the symbolic derivative of an expression and compare it to an
    expected result.
    """
    sym_df = sym.diff(pmbl.var("x"))(sym_f)
    if isinstance(sym_f, np.ndarray):
        assert (sym_df == expected_sym_df).all()
    else:
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
    sym_z = sym_coords[2]

    # Vector
    sym_f = make_obj_array([
        sym_x,
        sym_x * sym_y,
        sym_y])

    sym_div_f = sym.div(3, sym_f)
    expected_sym_div_f = 1 + sym_x + 0
    assert sym_div_f == expected_sym_div_f

    # Object array of vectors
    sym_f = make_obj_array([
        make_obj_array([sym_x, sym_x * sym_y, sym_y]),
        make_obj_array([sym_y, sym_y * sym_z, sym_z]),
        make_obj_array([sym_z, sym_z * sym_x, sym_x])])

    sym_div_f = sym.div(3, sym_f)
    expected_sym_div_f = make_obj_array([
        1 + sym_x + 0,
        0 + sym_z + 1,
        0 + 0 + 0])
    assert (sym_div_f == expected_sym_div_f).all()

    # Tensor
    sym_f = np.outer(sym_coords, sym_coords)

    sym_div_f = sym.div(3, sym_f)
    expected_sym_div_f = make_obj_array([
        sym_x + sym_x + sym_x + sym_x,
        sym_y + sym_y + sym_y + sym_y,
        sym_z + sym_z + sym_z + sym_z])
    assert (sym_div_f == expected_sym_div_f).all()

    # Array container
    from mirgecom.fluid import make_conserved
    sym_f = make_conserved(
        dim=3,
        mass=make_obj_array([sym_x, sym_x * sym_y, sym_y]),
        momentum=np.outer(sym_coords, sym_coords),
        energy=make_obj_array([sym_y, sym_y * sym_z, sym_z]),
        species_mass=np.empty((0, 3), dtype=object))

    sym_div_f = sym.div(3, sym_f)
    expected_sym_div_f = make_conserved(
        dim=3,
        mass=1 + sym_x + 0,
        momentum=make_obj_array([
            sym_x + sym_x + sym_x + sym_x,
            sym_y + sym_y + sym_y + sym_y,
            sym_z + sym_z + sym_z + sym_z]),
        energy=0 + sym_z + 1)
    assert sym_div_f.mass == expected_sym_div_f.mass
    assert (sym_div_f.momentum == expected_sym_div_f.momentum).all()
    assert sym_div_f.energy == expected_sym_div_f.energy
    assert sym_div_f.species_mass.shape == expected_sym_div_f.species_mass.shape


def test_symbolic_grad():
    """
    Compute the symbolic gradient of an expression and compare it to an expected
    result.
    """
    sym_coords = pmbl.make_sym_vector("x", 3)
    sym_x = sym_coords[0]
    sym_y = sym_coords[1]
    sym_z = sym_coords[2]

    # Scalar
    sym_f = sym_x**2 * sym_y

    sym_grad_f = sym.grad(3, sym_f)
    expected_sym_grad_f = make_obj_array([
        sym_y * 2*sym_x,
        sym_x**2,
        0])
    assert (sym_grad_f == expected_sym_grad_f).all()

    # Vector (nested)
    sym_f = make_obj_array([
        sym_x,
        sym_x * sym_y,
        sym_z])

    sym_grad_f = sym.grad(3, sym_f, nested=True)
    expected_sym_grad_f = make_obj_array([
        make_obj_array([1, 0, 0]),
        make_obj_array([sym_y, sym_x, 0]),
        make_obj_array([0, 0, 1])])
    assert (sym_grad_f[0] == expected_sym_grad_f[0]).all()
    assert (sym_grad_f[1] == expected_sym_grad_f[1]).all()
    assert (sym_grad_f[2] == expected_sym_grad_f[2]).all()

    # Vector (not nested)
    sym_f = make_obj_array([
        sym_x,
        sym_x * sym_y,
        sym_z])

    sym_grad_f = sym.grad(3, sym_f, nested=False)
    expected_sym_grad_f = np.stack(
        make_obj_array([
            make_obj_array([1, 0, 0]),
            make_obj_array([sym_y, sym_x, 0]),
            make_obj_array([0, 0, 1])]))
    assert (sym_grad_f == expected_sym_grad_f).all()

    # Array container
    from mirgecom.fluid import make_conserved
    sym_f = make_conserved(
        dim=3,
        mass=sym_x * sym_y,
        momentum=make_obj_array([sym_x, sym_x * sym_y, sym_z]),
        energy=sym_z * sym_x)

    sym_grad_f = sym.grad(3, sym_f)
    expected_sym_grad_f = make_conserved(
        dim=3,
        mass=make_obj_array([sym_y, sym_x, 0]),
        momentum=np.stack(
            make_obj_array([
                make_obj_array([1, 0, 0]),
                make_obj_array([sym_y, sym_x, 0]),
                make_obj_array([0, 0, 1])])),
        energy=make_obj_array([sym_z, 0, sym_x]),
        species_mass=np.empty((0, 3), dtype=object))
    assert (sym_grad_f.mass == expected_sym_grad_f.mass).all()
    assert (sym_grad_f.momentum == expected_sym_grad_f.momentum).all()
    assert (sym_grad_f.energy == expected_sym_grad_f.energy).all()
    assert sym_grad_f.species_mass.shape == expected_sym_grad_f.species_mass.shape


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

    # Scalar
    sym_f = (
        pmbl.var("exp")(-pmbl.var("t"))
        * pmbl.var("cos")(sym_coords[0])
        * pmbl.var("sin")(sym_coords[1]))
    f = sym.evaluate(sym_f, t=0.5, x=nodes)

    expected_f = np.exp(-0.5) * actx.np.cos(nodes[0]) * actx.np.sin(nodes[1])
    assert actx.to_numpy(discr.norm(f - expected_f)/discr.norm(expected_f)) < 1e-12

    # Vector
    sym_f = make_obj_array([
        pmbl.var("exp")(-pmbl.var("t")) * (0*sym_coords[0] + 1),
        pmbl.var("cos")(sym_coords[0]),
        pmbl.var("sin")(sym_coords[1])])
    f = sym.evaluate(sym_f, t=0.5, x=nodes)

    expected_f = make_obj_array([
        np.exp(-0.5) * (0*nodes[0] + 1),
        actx.np.cos(nodes[0]),
        actx.np.sin(nodes[1])])
    assert actx.to_numpy(discr.norm(f - expected_f)/discr.norm(expected_f)) < 1e-12

    # Array container
    from mirgecom.fluid import make_conserved
    sym_f = make_conserved(
        dim=2,
        mass=pmbl.var("exp")(-pmbl.var("t")) * (sym_coords[0] - sym_coords[0] + 1),
        momentum=make_obj_array([
            pmbl.var("cos")(sym_coords[0]),
            pmbl.var("cos")(sym_coords[1])]),
        energy=pmbl.var("sin")(sym_coords[0]))
    f = sym.evaluate(sym_f, t=0.5, x=nodes)

    expected_f = make_conserved(
        dim=2,
        mass=np.exp(-0.5) * (0*nodes[0] + 1),
        momentum=make_obj_array([
            actx.np.cos(nodes[0]),
            actx.np.cos(nodes[1])]),
        energy=actx.np.sin(nodes[0]))
    assert actx.to_numpy(discr.norm(f - expected_f)/discr.norm(expected_f)) < 1e-12


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
