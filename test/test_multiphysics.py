__copyright__ = """Copyright (C) 2022 University of Illinois Board of Trustees"""

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
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath # noqa
from pytools.obj_array import make_obj_array
import grudge.op as op
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from grudge.dof_desc import DOFDesc, VolumeDomainTag, DISCR_TAG_BASE
from mirgecom.discretization import create_discretization_collection
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
import pytest

import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_independent_volumes(actx_factory, order, visualize=False):
    """Check multi-volume machinery by setting up two independent volumes."""
    actx = actx_factory()

    n = 8

    dim = 2

    dim_names = ["x", "y", "z"]
    boundary_tag_to_face = {}
    for i in range(dim):
        boundary_tag_to_face["-"+str(i)] = ["-"+dim_names[i]]
        boundary_tag_to_face["+"+str(i)] = ["+"+dim_names[i]]

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-1,)*dim, b=(1,)*dim,
        nelements_per_axis=(n,)*dim, boundary_tag_to_face=boundary_tag_to_face)

    volume_meshes = {
        "vol1": mesh,
        "vol2": mesh,
    }

    dcoll = create_discretization_collection(actx, volume_meshes, order)

    dd_vol1 = DOFDesc(VolumeDomainTag("vol1"), DISCR_TAG_BASE)
    dd_vol2 = DOFDesc(VolumeDomainTag("vol2"), DISCR_TAG_BASE)

    nodes1 = actx.thaw(dcoll.nodes(dd=dd_vol1))
    nodes2 = actx.thaw(dcoll.nodes(dd=dd_vol2))

    boundaries1 = {
        dd_vol1.trace("-0").domain_tag: DirichletDiffusionBoundary(-1.),
        dd_vol1.trace("+0").domain_tag: DirichletDiffusionBoundary(1.),
        dd_vol1.trace("-1").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol1.trace("+1").domain_tag: NeumannDiffusionBoundary(0.),
    }

    boundaries2 = {
        dd_vol2.trace("-0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol2.trace("+0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol2.trace("-1").domain_tag: DirichletDiffusionBoundary(-1.),
        dd_vol2.trace("+1").domain_tag: DirichletDiffusionBoundary(1.),
    }

    u1 = nodes1[0]
    u2 = nodes2[1]

    u = make_obj_array([u1, u2])

    def get_rhs(t, u):
        return make_obj_array([
            diffusion_operator(
                dcoll, kappa=1, boundaries=boundaries1, u=u[0],
                volume_dd=dd_vol1),
            diffusion_operator(
                dcoll, kappa=1, boundaries=boundaries2, u=u[1],
                volume_dd=dd_vol2)])

    rhs = get_rhs(0, u)

    if visualize:
        from grudge.shortcuts import make_visualizer
        viz1 = make_visualizer(dcoll, order+3, volume_dd=dd_vol1)
        viz2 = make_visualizer(dcoll, order+3, volume_dd=dd_vol2)
        viz1.write_vtk_file(
            f"multiphysics_independent_volumes_{order}_1.vtu", [
                ("u", u[0]),
                ("rhs", rhs[0]),
                ])
        viz2.write_vtk_file(
            f"multiphysics_independent_volumes_{order}_2.vtu", [
                ("u", u[1]),
                ("rhs", rhs[1]),
                ])

    linf_err1 = actx.to_numpy(op.norm(dcoll, rhs[0], np.inf, dd=dd_vol1))
    linf_err2 = actx.to_numpy(op.norm(dcoll, rhs[1], np.inf, dd=dd_vol2))

    assert linf_err1 < 1e-9
    assert linf_err2 < 1e-9


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
