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
import math
import pytest
import numpy as np
import pyopencl as cl

from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw  # noqa
from mirgecom.filter import get_spectral_filter


@pytest.mark.parametrize("cutoff", [3, 4, 8])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_filter_coeff(ctx_factory, cutoff, dim, order):
    """
    Tests that the filter coefficients have the right shape
    and a quick check that the values at the filter
    band limits have the expected values.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    vol_discr = discr.discr_from_dd("vol")

    # number of polynomials
    numpoly = 1
    for d in range(1, dim+1):
        numpoly *= (order + d)
    numpoly /= math.factorial(int(dim))
    numpoly = int(numpoly)

    # number of filtered modes
    nfilt = numpoly - cutoff
    # alpha = f(machine eps)
    alpha = -1.0*np.log(np.finfo(float).eps)

    # expected values @ filter band limits
    expected_high_coeff = np.exp(-1.0*alpha)
    low_index = cutoff - 1
    high_index = numpoly - 1
    if nfilt <= 0:
        expected_high_coeff = 1.0
        low_index = 0

    from modepy import vandermonde
    for group in vol_discr.groups:
        vander = vandermonde(group.basis(), group.unit_nodes)
        vanderm1 = np.linalg.inv(vander)
        filter_coeff = get_spectral_filter(dim, order, cutoff, 2)
        assert(filter_coeff.shape == vanderm1.shape)
        assert(filter_coeff[high_index][high_index] == expected_high_coeff)
        assert(filter_coeff[low_index][low_index] == 1.0)
