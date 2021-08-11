__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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

import os
import math
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pytest

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import DOFArray, thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge import sym as grudge_sym
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_BASE, DTAG_BOUNDARY
from mirgecom.sampling import query_eval
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from mirgecom.mpi import mpi_entry_point
from pytools.obj_array import make_obj_array
import pyopencl.tools as cl_tools


@pytest.mark.parametrize("query_point", [
    (0, 0),
    (0, 1e-5),
    (0, -1e-5),
    (0.1, 0.2),
])
def test_query_eval(actx_factory, query_point, visualize=False):
    actx = actx_factory()

    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(0,)*dim,
        b=(1,)*dim,
        nelements_per_axis=(4,)*dim)

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)

    nodes = thaw(actx, discr.nodes())

    def simple_poly(x,y):
        return (x**3) + (y**2)

    def simple_poly_nodes(nodes):
        return simple_poly(nodes[0], nodes[1])

    u = simple_poly_nodes(nodes)

    if visualize:
        vis = make_visualizer(discr, order)
        vis.write_vtk_file("test_query_eval_mesh.vtu",
            [
                ("u", u)
            ], overwrite=True)

    tol = 1e-4

    q_mapped, q_elem = query_eval(np.array(query_point), actx, discr, dim, tol)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
