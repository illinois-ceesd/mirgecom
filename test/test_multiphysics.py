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

    global_mesh1 = generate_regular_rect_mesh(
        a=(-1, -2), b=(1, -1),
        nelements_per_axis=(n,)*dim, boundary_tag_to_face=boundary_tag_to_face)

    global_mesh2 = generate_regular_rect_mesh(
        a=(-1, 1), b=(1, 2),
        nelements_per_axis=(n,)*dim, boundary_tag_to_face=boundary_tag_to_face)

    volume_meshes = {
        "Lower": global_mesh1,
        "Upper": global_mesh2,
    }

    dcoll = create_discretization_collection(actx, volume_meshes, order)

    dd_vol_lower = DOFDesc(VolumeDomainTag("Lower"), DISCR_TAG_BASE)
    dd_vol_upper = DOFDesc(VolumeDomainTag("Upper"), DISCR_TAG_BASE)

    lower_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_lower))
    upper_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_upper))

    lower_boundaries = {
        dd_vol_lower.trace("-0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol_lower.trace("+0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol_lower.trace("-1").domain_tag: DirichletDiffusionBoundary(-2.),
        dd_vol_lower.trace("+1").domain_tag: DirichletDiffusionBoundary(-1.),
    }

    upper_boundaries = {
        dd_vol_upper.trace("-0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol_upper.trace("+0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol_upper.trace("-1").domain_tag: DirichletDiffusionBoundary(1.),
        dd_vol_upper.trace("+1").domain_tag: DirichletDiffusionBoundary(2.),
    }

    lower_u = lower_nodes[1]
    upper_u = upper_nodes[1]

    u = make_obj_array([lower_u, upper_u])

    def get_rhs(t, u):
        return make_obj_array([
            diffusion_operator(
                dcoll, kappa=1, boundaries=lower_boundaries, u=u[0],
                volume_dd=dd_vol_lower),
            diffusion_operator(
                dcoll, kappa=1, boundaries=upper_boundaries, u=u[1],
                volume_dd=dd_vol_upper)])

    rhs = get_rhs(0, u)

    if visualize:
        from grudge.shortcuts import make_visualizer
        viz_lower = make_visualizer(dcoll, order+3, volume_dd=dd_vol_lower)
        viz_upper = make_visualizer(dcoll, order+3, volume_dd=dd_vol_upper)
        viz_lower.write_vtk_file(
            f"multiphysics_independent_volumes_{order}_lower.vtu", [
                ("u", u[0]),
                ("rhs", rhs[0]),
                ])
        viz_upper.write_vtk_file(
            f"multiphysics_independent_volumes_{order}_upper.vtu", [
                ("u", u[1]),
                ("rhs", rhs[1]),
                ])

    linf_err_lower = actx.to_numpy(op.norm(dcoll, rhs[0], np.inf, dd=dd_vol_lower))
    linf_err_upper = actx.to_numpy(op.norm(dcoll, rhs[1], np.inf, dd=dd_vol_upper))

    assert linf_err_lower < 1e-9
    assert linf_err_upper < 1e-9


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
