__copyright__ = """Copyright (C) 2020 University of Illinois Board of Trustees"""

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
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
import grudge.op as op
from mirgecom.limiter import (
    cell_characteristic_size,
    bound_preserving_limiter
)
from mirgecom.discretization import create_discretization_collection
import pytest


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_positivity_preserving_limiter(actx_factory, order, dim):
    """Testing positivity-preserving limiter."""
    actx = actx_factory()

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.1,) * dim, b=(0.1,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    discr = create_discretization_collection(actx, mesh, order=order)

    # create cells with negative values
    nodes = actx.thaw(actx.freeze(discr.nodes()))
    eps = 0.1
    field = nodes[0]*eps

    cell_area = cell_characteristic_size(actx, discr)

    limited_field = bound_preserving_limiter(discr, cell_area, field, mmin=0.0)

    error = actx.np.linalg.norm(op.elementwise_min(discr, limited_field), np.inf)

    assert error > -1.0e-13

@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_bound_preserving_limiter(actx_factory, order, dim):
    """Testing upper bound limiting."""
    actx = actx_factory()

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.1,) * dim, b=(0.1,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    discr = create_discretization_collection(actx, mesh, order=order)

    # create cells with values larger than 1.0
    nodes = actx.thaw(actx.freeze(discr.nodes()))
    eps = 0.1
    field = 1.0 + nodes[0]*eps

    cell_area = cell_characteristic_size(actx, discr)

    limited_field = -1.0 + bound_preserving_limiter(discr, cell_area, field,
                                             mmin=0.0, mmax=1.0)

    error = actx.np.linalg.norm(op.elementwise_min(discr, limited_field), np.inf)

    assert error > 1.0e-13
