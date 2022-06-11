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

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


def test_actx_np_where(actx_factory):
    """Test whether scalars work in array context fake numpy where."""

    actx = actx_factory()
    order = 1
    dim = 3
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(11,) * dim
    )
    from mirgecom.discretization import create_discretization_collection
    discr = create_discretization_collection(actx, mesh, order)
    from arraycontext import thaw
    nodes = thaw(discr.nodes(), actx)
    zero_ary = discr.zeros(actx)
    one_ary = 1.0 + zero_ary
    yesno_ary = actx.np.greater(nodes[0], 0.)
    try_this = actx.np.where(yesno_ary, zero_ary, one_ary)
    print(f"{try_this=}")
    try_this = actx.np.where(yesno_ary, zero_ary, 1.0)
    print(f"{try_this=}")
    try_this = actx.np.where(yesno_ary, 0.0, one_ary)
    print(f"{try_this=}")
