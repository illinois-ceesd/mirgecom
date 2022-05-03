"""Test filter-related functions and constructs."""

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
from functools import partial

from meshmode.dof_array import thaw
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from pytools.obj_array import (
    make_obj_array
)
from meshmode.dof_array import thaw  # noqa
from mirgecom.filter import make_spectral_filter


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("filter_order", [1, 2, 3])
def test_filter_coeff(actx_factory, filter_order, order, dim):
    """
    Test the construction of filter coefficients.

    Tests that the filter coefficients have the right values
    at the imposed band limits of the filter.  Also tests that
    the created filter operator has the expected shape:
    (nummodes x nummodes) matrix, and the filter coefficients
    in the expected positions corresponding to mode ids.
    """
    actx = actx_factory()

    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    vol_discr = discr.discr_from_dd("vol")

    eta = .5  # just filter half the modes
    # number of modes see e.g.:
    # JSH/TW Nodal DG Methods, Section 10.1
    # DOI: 10.1007/978-0-387-72067-8
    nmodes = 1
    for d in range(1, dim+1):
        nmodes *= (order + d)
    nmodes /= math.factorial(int(dim))
    nmodes = int(nmodes)

    cutoff = int(eta * order)

    # number of filtered modes
    nfilt = order - cutoff
    # alpha = f(machine eps)
    # Alpha value suggested by:
    # JSH/TW Nodal DG Methods, Section 5.3
    # DOI: 10.1007/978-0-387-72067-8
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

    for group in vol_discr.groups:
        mode_ids = group.mode_ids()
        filter_coeff = actx.thaw(
            make_spectral_filter(
                actx, group, cutoff=cutoff,
                mode_response_function=frfunc
            )
        )
        for mode_index, mode_id in enumerate(mode_ids):
            mode = mode_id
            if dim > 1:
                mode = sum(mode_id)
            if mode == cutoff:
                assert(filter_coeff[mode_index] == expected_cutoff_coeff)
            if mode == order:
                assert(filter_coeff[mode_index] == expected_high_coeff)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_filter_function(actx_factory, dim, order, do_viz=False):
    """
    Test the stand-alone procedural interface to spectral filtering.

    Tests that filtered fields have expected attenuated higher modes.
    """
    actx = actx_factory()

    logger = logging.getLogger(__name__)
    filter_order = 1
    nel_1d = 1
    eta = .5   # filter half the modes
    # Alpha value suggested by:
    # JSH/TW Nodal DG Methods, Seciton 5.3
    # DOI: 10.1007/978-0-387-72067-8
    alpha = -1.0*np.log(np.finfo(float).eps)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim, b=(1.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # number of modes see e.g.:
    # JSH/TW Nodal DG Methods, Section 10.1
    # DOI: 10.1007/978-0-387-72067-8
    nummodes = int(1)
    for _ in range(dim):
        nummodes *= int(order + dim + 1)
    nummodes /= math.factorial(int(dim))
    cutoff = int(eta * order)

    from mirgecom.filter import exponential_mode_response_function as xmrfunc
    frfunc = partial(xmrfunc, alpha=alpha, filter_order=filter_order)

    # First test a uniform field, which should pass through
    # the filter unharmed.
    from mirgecom.initializers import Uniform
    initr = Uniform(dim=dim)
    uniform_soln = initr(t=0, x_vec=nodes)

    from mirgecom.filter import filter_modally
    filtered_soln = filter_modally(discr, "vol", cutoff,
                                   frfunc, uniform_soln)
    soln_resid = uniform_soln - filtered_soln
    from mirgecom.simutil import componentwise_norms
    max_errors = componentwise_norms(discr, soln_resid, np.inf)

    tol = 1e-14

    logger.info(f"Max Errors (uniform field) = {max_errors}")
    assert actx.np.less(np.max(max_errors), tol)

    # construct polynomial field:
    # a0 + a1*x + a2*x*x + ....
    def polyfn(coeff):  # , x_vec):
        # r = actx.np.sqrt(np.dot(nodes, nodes))
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
    filtered_field = filter_modally(discr, "vol", cutoff,
                                    frfunc, field)
    soln_resid = field - filtered_field
    max_errors = [actx.to_numpy(discr.norm(v, np.inf)) for v in soln_resid]
    logger.info(f"Field = {field}")
    logger.info(f"Filtered = {filtered_field}")
    logger.info(f"Max Errors (poly) = {max_errors}")
    assert(np.max(max_errors) < tol)

    # Any order > cutoff fields should have higher modes attenuated
    threshold = 1e-3
    tol = 1e-1
    if do_viz is True:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(discr, discr.order)

    from grudge.dof_desc import DD_VOLUME_MODAL, DD_VOLUME

    modal_map = discr.connection_from_dds(DD_VOLUME, DD_VOLUME_MODAL)

    for field_order in range(cutoff+1, cutoff+4):
        coeff = [1.0 / (i + 1) for i in range(field_order+1)]
        field = polyfn(coeff=coeff)
        filtered_field = filter_modally(discr, "vol", cutoff,
                                        frfunc, field)

        unfiltered_spectrum = modal_map(field)
        filtered_spectrum = modal_map(filtered_field)
        if do_viz is True:
            spectrum_resid = unfiltered_spectrum - filtered_spectrum
            io_fields = [
                ("unfiltered", field),
                ("filtered", filtered_field),
                ("unfiltered_spectrum", unfiltered_spectrum),
                ("filtered_spectrum", filtered_spectrum),
                ("residual", spectrum_resid)
            ]
            vis.write_vtk_file(f"filter_test_{field_order}.vtu", io_fields)
        field_resid = unfiltered_spectrum - filtered_spectrum
        max_errors = [actx.to_numpy(discr.norm(v, np.inf)) for v in field_resid]
        # fields should be different, but not too different
        assert(tol > np.max(max_errors) > threshold)
