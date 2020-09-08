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
from functools import partial

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from pytools.obj_array import (
    make_obj_array
)
from meshmode.dof_array import thaw  # noqa
from mirgecom.filter import (
    make_spectral_filter,
    SpectralFilter,
    apply_linear_operator,
)
# Uncomment if you want to inspect results in VTK
# from grudge.shortcuts import make_visualizer


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("filter_order", [1, 2, 3])
def test_filter_coeff(ctx_factory, filter_order, order, dim):
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

    eta = .5
    # number of points
    nmodes = 1
    for d in range(1, dim+1):
        nmodes *= (order + d)
    nmodes /= math.factorial(int(dim))
    nmodes = int(nmodes)

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

    from mirgecom.filter import exponential_mode_response_function as xmrfunc
    frfunc = partial(xmrfunc, alpha=alpha, filter_order=filter_order)

    from modepy import vandermonde
    for group in vol_discr.groups:
        mode_ids = group.mode_ids()
        #        print(f"mode_ids = {mode_ids}")
        #        for mode_index, mode_id in enumerate(mode_ids):
        #            mode = sum(modelist)
        #            print(f"mode = {mode}")
        #            print(f"list(mode) = {list(mode)}")
        vander = vandermonde(group.basis(), group.unit_nodes)
        vanderm1 = np.linalg.inv(vander)
        filter_coeff = make_spectral_filter(group, cutoff=cutoff,
                                            mode_response_function=frfunc)
        assert(filter_coeff.shape == vanderm1.shape)
        for mode_index, mode_id in enumerate(mode_ids):
            mode = mode_id
            if dim > 1:
                mode = sum(mode_id)
            if mode == cutoff:
                assert(filter_coeff[mode_index][mode_index] == expected_cutoff_coeff)
            if mode == order:
                assert(filter_coeff[mode_index][mode_index] == expected_high_coeff)


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
    actx = PyOpenCLArrayContext(queue)

    logger = logging.getLogger(__name__)
    filter_order = 1
    nel_1d = 2
    eta = .5
    alpha = -1.0*np.log(np.finfo(float).eps)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim, b=(1.0,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    nummodes = int(1)
    for i in range(dim):
        nummodes *= int(order + dim + 1)
    nummodes /= math.factorial(int(dim))
    cutoff = int(eta * order)

    from mirgecom.filter import exponential_mode_response_function as xmrfunc
    frfunc = partial(xmrfunc, alpha=alpha, filter_order=filter_order)

    vol_discr = discr.discr_from_dd("vol")
    groups = vol_discr.groups
    group = groups[0]
    filter_mat = make_spectral_filter(group, cutoff, mode_response_function=frfunc)
    spectral_filter = SpectralFilter(vol_discr, filter_mat)

    # First test a uniform field, which should pass through
    # the filter unharmed.
    from mirgecom.initializers import Uniform
    initr = Uniform(numdim=dim)
    uniform_soln = initr(t=0, x_vec=nodes)

    filtered_soln = spectral_filter(discr=vol_discr, fields=uniform_soln)
    soln_resid = uniform_soln - filtered_soln
    max_errors = [discr.norm(v, np.inf) for v in soln_resid]

    tol = 1e-14

    logger.info(f"Max Errors (uniform field) = {max_errors}")
    assert(np.max(max_errors) < tol)

    # construct polynomial field:
    # a0 + a1*x + a2*x*x + ....
    def polyfn(coeff):  # , x_vec):
        #        r = actx.np.sqrt(np.dot(nodes, nodes))
        r = nodes[0]
        result = 0
        for n, a in enumerate(coeff):
            result += a * r ** n
        return make_obj_array([result])

    # Any order {cutoff} and below fields should be unharmed
    tol = 1e-14
    field_order = int(cutoff)
    coeff = [1.0 / (i + 1) for i in range(field_order + 1)]
    field = polyfn(coeff=coeff)
    filtered_field = spectral_filter(vol_discr, field)
    soln_resid = field - filtered_field
    max_errors = [discr.norm(v, np.inf) for v in soln_resid]
    logger.info(f"Field = {field}")
    logger.info(f"Filtered = {filtered_field}")
    logger.info(f"Max Errors (poly) = {max_errors}")
    assert(np.max(max_errors) < tol)

    # Any order > cutoff fields should have higher modes attenuated
    threshold = 1e-3
    tol = 1e-1
    # Uncomment for visualization/inspection of test fields and spectra
    #    vis = make_visualizer(discr, discr.order)
    from modepy import vandermonde
    for field_order in range(cutoff+1, cutoff+4):
        coeff = [1.0 / (i + 1) for i in range(field_order+1)]
        field = polyfn(coeff=coeff)
        filtered_field = spectral_filter(vol_discr, field)
        for group in vol_discr.groups:
            vander = vandermonde(group.basis(), group.unit_nodes)
            vanderm1 = np.linalg.inv(vander)
            unfiltered_spectrum = apply_linear_operator(vol_discr, vanderm1, field)
            filtered_spectrum = apply_linear_operator(vol_discr, vanderm1,
                                                      filtered_field)
            # Uncomment judiciously to visually inspect fields & spectra
            # spectrum_resid = unfiltered_spectrum - filtered_spectrum
            # spectrum_scale = filtered_spectrum / unfiltered_spectrum
            # io_fields = [
            #   ('unfiltered', field),
            #   ('filtered', filtered_field),
            #   ('unfiltered_spectrum', unfiltered_spectrum),
            #   ('filtered_spectrum', filtered_spectrum),
            #   ('residual', spectrum_resid)
            # ]
            # vis.write_vtk_file(f"filter_test_{field_order}.vtu", io_fields)
            field_resid = unfiltered_spectrum - filtered_spectrum
            max_errors = [discr.norm(v, np.inf) for v in field_resid]
            # fields should be different, but not too different
            assert(tol > np.max(max_errors) > threshold)
