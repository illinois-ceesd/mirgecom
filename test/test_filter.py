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
import numpy as np
import logging
import pytest
#import pyopencl as cl

#from pytools.obj_array import (
#    flat_obj_array,
#    make_obj_array,
#)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_vandermonde(ctx_factory, dim, order):
    cl_ctx = ctx_factory()
    #    queue = cl.CommandQueue(cl_ctx)
    logger = logging.getLogger(__name__)
    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    #    for dim in [1, 2, 3]:
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    vol_discr = discr.discr_from_dd("vol")
    ngroups = len(vol_discr.groups)
    logger.info(f'Number of groups: {ngroups}')

    from modepy import vandermonde
    for group in vol_discr.groups:
        vander = vandermonde(group.basis(), group.unit_nodes)
        vanderm1 = np.linalg.inv(vander)
        imaybe = np.matmul(vander, vanderm1)
        iresid = np.abs(imaybe - np.identity(len(group.unit_nodes[0])))
        maxerr = np.max(iresid)
        assert(maxerr <= 1e-15)

