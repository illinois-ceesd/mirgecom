"""Test floating point array comparison tools."""

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

import pytest  # noqa

from grudge.eager import EagerDGDiscretization
from mirgecom.float_comparisons import within_tol

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)


def test_within_tol_absolute(actx_factory):
    """Test absolute mode of tolerance checking."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    assert within_tol(discr, ones, ones + 1e-7, tol=1e-6, relative=False)
    assert not within_tol(discr, ones, ones + 1e-5, tol=1e-6, relative=False)


def test_within_tol_relative(actx_factory):
    """Test relative mode of tolerance checking."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    assert within_tol(discr, 30 * ones, 30 * ones + 1e-5, tol=1e-6)
    assert not within_tol(discr, .01 * ones, .01 * ones + 1e-7, tol=1e-6)


def test_within_tol_errors_around_zero(actx_factory):
    """Test relative mode's checks when zero values are present."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    zeros = discr.zeros(actx)

    assert within_tol(discr, zeros, zeros + 1e-100, tol=1e-6)
    assert not within_tol(
        discr, zeros, zeros + 1e-100, tol=1e-6,
        correct_for_eps_differences_from_zero=False
    )
