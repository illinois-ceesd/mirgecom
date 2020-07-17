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
import logging
import math
import pytest
import numpy as np

import pyopencl as cl
#import pyopencl.clmath as clmath
import pyopencl.array as clarray
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from pytools.obj_array import make_obj_array
from mirgecom.filter import get_spectral_filter
from mirgecom.filter import SpectralFilter
from mirgecom.filter import apply_linear_operator
from mirgecom.initializers import Uniform
from mirgecom.error import compare_states


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("filter_order", [1, 2, 3])
def test_filter_coeff(ctx_factory, filter_order, order, dim):
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

    eta = .5
    # number of points
    npt = 1
    for d in range(1, dim+1):
        npt *= (order + d)
    npt /= math.factorial(int(dim))
    npt = int(npt)
    cutoff = int(eta * order)

    # number of filtered modes
    nfilt = order - cutoff
    # alpha = f(machine eps)
    alpha = -1.0*np.log(np.finfo(float).eps)

    # expected values @ filter band limits
    expected_high_coeff = np.exp(-1.0*alpha)
    expected_cutoff_coeff = 1.0
    if dim == 1:
        cutoff_indices = [cutoff]
        high_indices = [order]
    elif dim == 2:
        sk = 0
        cutoff_indices = []
        high_indices = []
        for i in range(order + 1):
            for j in range(order - i + 1):
                if (i + j) == cutoff:
                    cutoff_indices.append(sk)
                if (i + j) == order:
                    high_indices.append(sk)
                sk += 1
    elif dim == 3:
        sk = 0
        cutoff_indices = []
        high_indices = []
        for i in range(order + 1):
            for j in range(order - i + 1):
                for k in range(order - (i + j) + 1):
                    if (i + j + k) == cutoff:
                        cutoff_indices.append(sk)
                    if (i + j + k) == order:
                        high_indices.append(sk)
                    sk += 1

    if nfilt <= 0:
        expected_high_coeff = 1.0

    from modepy import vandermonde
    for group in vol_discr.groups:

        vander = vandermonde(group.basis(), group.unit_nodes)
        vanderm1 = np.linalg.inv(vander)
        filter_coeff = get_spectral_filter(dim, alpha, order, cutoff, filter_order)
        assert(filter_coeff.shape == vanderm1.shape)
        for high_index in high_indices:
            assert(filter_coeff[high_index][high_index] == expected_high_coeff)
        for low_index in cutoff_indices:
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
    logger = logging.getLogger(__name__)
    filter_order = 1
    nel_1d = 16
    eta = .5
    alpha = -1.0*np.log(np.finfo(float).eps)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim, b=(1.0,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    nodes = discr.nodes().with_queue(queue)

    npt = int(1)
    for i in range(dim):
        npt *= int(order + dim + 1)
    npt /= math.factorial(int(dim))
    cutoff = int(eta * order)

    vol_discr = discr.discr_from_dd("vol")
    filter_mat = get_spectral_filter(dim, alpha, order, cutoff, filter_order)
    spectral_filter = SpectralFilter(vol_discr, filter_mat)

    # First test a uniform field, which should pass through
    # the filter unharmed.
    initr = Uniform(numdim=dim)
    uniform_soln = initr(t=0, x_vec=nodes)
    filtered_soln = spectral_filter(discr=vol_discr, fields=uniform_soln)
    max_errors = compare_states(uniform_soln, filtered_soln)
    tol = 1e-14

    logger.info(f'Max Errors (uniform field) = {max_errors}')
    assert(np.max(max_errors) < tol)

    # construct polynomial field:
    # a0 + a1*x + a2*x*x + ....
    def polyfn(coeff, x_vec):
        my_x = make_obj_array(
            [x_vec[i] for i in range(dim)]
        )
        #        r = clmath.sqrt(np.dot(my_x, my_x))
        r = my_x[0]
        result = clarray.zeros(r.queue, shape=r.shape, dtype=np.float64)
        for n, a in enumerate(coeff):
            result += a * r ** n
        return result

    # Any order {cutoff} and below fields should be unharmed
    tol = 1e-14
    field_order = int(cutoff)
    coeff = [1.0 / (i + 1) for i in range(field_order + 1)]
    field = polyfn(coeff=coeff, x_vec=nodes)
    field = make_obj_array([field])
    filtered_field = spectral_filter(vol_discr, field)
    max_errors = compare_states(field, filtered_field)
    logger.info(f'Field = {field}')
    logger.info(f'Filtered = {filtered_field}')
    logger.info(f'Max Errors (poly) = {max_errors}')
    assert(np.max(max_errors) < tol)

    # Any order > cutoff fields should have higher modes attenuated
    tol = 1e-3
    from modepy import vandermonde
    for field_order in range(cutoff+1, cutoff+4):
        coeff = [1.0 / (i + 1) for i in range(field_order+1)]
        field = polyfn(coeff=coeff, x_vec=nodes)
        field = make_obj_array([field])
        for group in vol_discr.groups:
            vander = vandermonde(group.basis(), group.unit_nodes)
            vanderm1 = np.linalg.inv(vander)
            unfiltered_spectrum = apply_linear_operator(vol_discr, vanderm1, field)
            filtered_field = spectral_filter(vol_discr, field)
            filtered_spectrum = apply_linear_operator(vol_discr, vanderm1,
                                                      filtered_field)
            max_errors = compare_states(unfiltered_spectrum, filtered_spectrum)
            logger.info(f'Field = {field}')
            logger.info(f'Filtered = {filtered_field}')
            logger.info(f'Max Errors (poly) = {max_errors}')
            assert(np.max(max_errors) < tol)
