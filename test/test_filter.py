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

from mirgecom.filter import get_spectral_filter
from mirgecom.filter import SpectralFilter
from mirgecom.initializers import Uniform
from mirgecom.error import compare_states


@pytest.mark.parametrize("cutoff", [3, 4, 8])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_filter_coeff(ctx_factory, cutoff, dim, order):
    """
    Tests that the filter coefficients have the right shape
    and a quick sanity check on the values at the filter
    band limits.
    """
    cl_ctx = ctx_factory()
    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
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
    expected_cutoff_coeff = 1.0
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
        assert(filter_coeff[low_index][low_index] == expected_cutoff_coeff)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_filter_class(ctx_factory, dim, order):
    """
    Tests that the SpectralFilter class performs the
    correct operation on the input fields. Several
    test input fields are (will be) tested.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    nodes = discr.nodes().with_queue(queue)

    npoly = int(1)
    for i in range(dim):
        npoly *= int(order + dim + 1)
    npoly /= math.factorial(int(dim))
    cutoff = int(npoly / 2)

    vol_discr = discr.discr_from_dd("vol")
    filter_mat = get_spectral_filter(dim, order, cutoff, 2)
    spectral_filter = SpectralFilter(vol_discr, filter_mat)

    # First test a uniform field, which should pass through
    # the filter unharmed.
    initr = Uniform(numdim=dim)
    uniform_soln = initr(t=0, x_vec=nodes)
    filtered_soln = spectral_filter(vol_discr, uniform_soln)
    max_errors = compare_states(uniform_soln, filtered_soln)
    tol = 1e-14

    print(f'Max Errors = {max_errors}')
    assert(np.max(max_errors) < tol)
