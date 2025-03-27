"""Test built-in callback routines."""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
import pytest  # noqa

from arraycontext import flatten

from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts

from mirgecom.fluid import make_conserved
from mirgecom.eos import IdealSingleGas
from mirgecom.discretization import create_discretization_collection

pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory])


def test_basic_cfd_healthcheck(actx_factory):
    """Quick test of some health checking utilities."""
    actx = actx_factory()
    nel_1d = 4
    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = dcoll.zeros(actx)
    ones = zeros + 1.0

    # Let's make a very bad state (negative mass)
    mass = -1*ones
    velocity = 2 * nodes
    mom = mass * velocity
    energy = zeros + .5*np.dot(mom, mom)/mass

    eos = IdealSingleGas()
    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    pressure = eos.pressure(cv)

    from mirgecom.simutil import check_range_local
    assert check_range_local(dcoll, "vol", mass, min_value=0, max_value=np.inf)
    assert check_range_local(dcoll, "vol", pressure, min_value=1e-6,
                             max_value=np.inf)

    # Let's make another very bad state (nans)
    mass = 1*ones
    energy = zeros + 2.5
    velocity = np.nan * nodes
    mom = mass * velocity

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    pressure = eos.pressure(cv)

    from mirgecom.simutil import check_naninf_local
    assert check_naninf_local(dcoll, "vol", pressure)

    # Let's make one last very bad state (inf)
    mass = 1*ones
    energy = np.inf * ones
    velocity = 2 * nodes
    mom = mass * velocity

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    pressure = eos.pressure(cv)

    assert check_naninf_local(dcoll, "vol", pressure)

    # What the hey, test a good one
    energy = 2.5 + .5*np.dot(mom, mom)
    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    pressure = eos.pressure(cv)

    assert not check_naninf_local(dcoll, "vol", pressure)
    assert not check_range_local(dcoll, "vol", pressure, min_value=0,
                                 max_value=np.inf)


def test_analytic_comparison(actx_factory):
    """Quick test of state comparison routine."""
    from mirgecom.initializers import Vortex2D
    from mirgecom.simutil import compare_fluid_solutions, componentwise_norms

    actx = actx_factory()
    nel_1d = 4
    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 2
    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = dcoll.zeros(actx)
    ones = zeros + 1.0
    mass = ones
    energy = ones
    velocity = 2 * nodes
    mom = mass * velocity
    vortex_init = Vortex2D()
    vortex_soln = vortex_init(x_vec=nodes, eos=IdealSingleGas())

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    resid = vortex_soln - cv

    expected_errors = actx.to_numpy(
        flatten(componentwise_norms(dcoll, resid, order=np.inf), actx)).tolist()

    errors = compare_fluid_solutions(dcoll, cv, cv)
    assert max(errors) == 0

    errors = compare_fluid_solutions(dcoll, cv, vortex_soln)
    assert errors == expected_errors


@pytest.mark.parametrize("dim", [2, 3])
def test_mesh_to_mesh_xfer(actx_factory, dim):
    """Quick test mesh to mesh data transfer."""
    from meshmode.mesh.generation import generate_regular_rect_mesh
    from meshmode.mesh import TensorProductElementGroup
    from mirgecom.simutil import (
        inverse_element_connectivity,
        compute_vertex_averages,
        scatter_vertex_values_to_dofarray,
        build_elemental_interpolation_info,
        recover_interp_fallbacks,
        apply_elemental_interpolation,
        remap_dofarrays_in_structure
    )

    actx = actx_factory()
    nel_1d_1 = 4
    nel_1d_2 = 7
    group_cls = TensorProductElementGroup

    mesh_1 = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d_1,) * dim,
        group_cls=group_cls
    )

    mesh_2 = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d_2,) * dim,
        group_cls=group_cls
    )

    dcoll_1 = create_discretization_collection(actx, mesh_1, order=1)
    dcoll_2 = create_discretization_collection(actx, mesh_2, order=1)

    nodes_1 = actx.thaw(dcoll_1.nodes())
    nodes_2 = actx.thaw(dcoll_2.nodes())

    src_x = nodes_1[0]

    # The element connectivity (econn) for an unstructured mesh is an object
    # that maps the element index to the ordered list of vertex indices
    # that make up that element.
    #
    # The inverse econn is an object that maps the vertex indices to
    # a list of elements that touch that vertex. Here we collect the
    # additional info: (grp_id, el_ind, vert_pos)
    # grp_id: Element group ID
    # el_ind: Element index
    # vert_pos: Element-local position for the vertex
    #
    # This data structure is key as it allows us to quickly gather
    # information from every element touching a node, and scatter
    # nodal attributes to all the elements that touch it.
    iconn_1 = inverse_element_connectivity(mesh_1)

    # This call gets the average value of the nodal coordinates over
    # all elements that touch this vertex. Since all elements agree
    # on the nodal coordinates, it should *match* the value from
    # mesh.vertices.
    src_av_coords = [compute_vertex_averages(actx, x, iconn_1)
                     for x in nodes_1]

    from pytools.obj_array import make_obj_array
    # Test the scatter back to dofarray vals
    # This bit scatters the averaged coordinates back to the elements
    # and makes a vector attribute just like "nodes".
    xf_av_coords = make_obj_array([scatter_vertex_values_to_dofarray(
        actx, src_x, iconn_1, src_x_av) for src_x_av in src_av_coords])

    # Verify that the averaged vals are *identical* to the originals
    xf_resid = nodes_1 - xf_av_coords
    for dim_resid in xf_resid:
        resid = actx.to_numpy(dim_resid)[0]
        assert np.max(np.abs(resid)) == 0.0

    # ------ Test mesh-to-mesh interp ------------

    # The interp info tells us for each vertex in mesh2,
    # the element id and element-local natural coordinates
    # of the element that contains the mesh2 vertex.
    # This information can be used to evaluate mesh1's
    # element shape functions to get a field value at
    # the mesh2 vertex location.
    interp_info_11 = build_elemental_interpolation_info(mesh_1, mesh_1)
    interp_info_22 = build_elemental_interpolation_info(mesh_2, mesh_2)

    # These self-to-self interps should work flawlessly
    assert np.all(interp_info_11.src_element_ids >= 0)
    assert np.all(interp_info_22.src_element_ids >= 0)
    assert interp_info_11.ref_coords.shape == (mesh_1.vertices.shape[1],
                                               mesh_1.ambient_dim)
    assert interp_info_22.ref_coords.shape == (mesh_2.vertices.shape[1],
                                               mesh_2.ambient_dim)

    # Test building interp wghts for xfer mesh1-->mesh2
    interp_info_12 = build_elemental_interpolation_info(mesh_1, mesh_2)

    # This next bit checks to see if any mesh2 vertices were
    # unfound by the build_elemental_interpolation_info call.
    print("Total (1->2) fallback points(before):",
          len(interp_info_12.fallback_indices))
    for i in interp_info_12.fallback_indices[:5]:
        print(f"Point {i}:",
              mesh_2.vertices[:, i],
              " -> Not found in mesh_1")

    # This bit tries to use a Gauss-Newton solver to
    # find an element, natural coordinate for the
    # unfound points (if any)
    if np.any(interp_info_12.fallback_mask):
        interp_info_12 = recover_interp_fallbacks(
            interp_info_12, mesh_1, mesh_2)

    # Check again to make sure they were all got
    # by the GaussNewton
    print("Total (1->2) fallback points(after):",
          len(interp_info_12.fallback_indices))
    for i in interp_info_12.fallback_indices[:5]:
        print(f"Point {i}:",
              mesh_2.vertices[:, i],
              " -> Not found in mesh_1")

    # Don't allow any nodes to be unfound, or fail the test
    assert np.all(interp_info_12.src_element_ids >= 0)
    assert interp_info_12.ref_coords.shape == (mesh_2.vertices.shape[1],
                                                   mesh_1.ambient_dim)

    # Test building interp wghts for xfer mesh2-->mesh1
    interp_info_21 = build_elemental_interpolation_info(mesh_2, mesh_1)

    print("Total (2->1) fallback points(before):",
          len(interp_info_21.fallback_indices))
    for i in interp_info_21.fallback_indices[:5]:
        print(f"Point {i}:",
              mesh_1.vertices[:, i],
              " -> Not found in mesh_1")

    if np.any(interp_info_21.fallback_mask):
        interp_info_12 = recover_interp_fallbacks(
            interp_info_21, mesh_2, mesh_1)

    print("Total (2->1) fallback points(after):",
          len(interp_info_21.fallback_indices))
    for i in interp_info_21.fallback_indices[:5]:
        print(f"Point {i}:",
              mesh_1.vertices[:, i],
              " -> Not found in mesh_1")

    assert np.all(interp_info_21.src_element_ids >= 0)
    assert interp_info_21.ref_coords.shape == (mesh_1.vertices.shape[1],
                                               mesh_2.ambient_dim)

    # Try doing some interpolation (1-->2) then (2-->1)

    # In these tests, we transfer the nodal coordinates from
    # one mesh to the other.
    vert1 = mesh_1.vertices
    vert2 = mesh_2.vertices

    # Shoud be good to a few 1e-16
    interp_tol = 2e-15

    for idim in range(dim):
        x_src_data = actx.to_numpy(nodes_1[idim])[0]
        x_trg_truth = vert2[idim]
        xf_x = apply_elemental_interpolation(x_src_data,
                                             interp_info_12)
        xf_resid = xf_x - x_trg_truth
        max_resid = np.max(np.abs(xf_resid))
        assert max_resid < interp_tol

    # Test the other direction
    for idim in range(dim):
        x_src_data = actx.to_numpy(nodes_2[idim])[0]
        x_trg_truth = vert1[idim]
        xf_x = apply_elemental_interpolation(x_src_data,
                                             interp_info_21)
        xf_resid = xf_x - x_trg_truth
        max_resid = np.max(np.abs(xf_resid))
        assert max_resid < interp_tol

    # All the low-level routines are tested, this next test
    # demonstrates how to use the high-level wrapper to
    # map something like a restart dataset from one mesh to
    # another.

    # make a bunch of fake data on mesh1
    cv = make_conserved(dim=dim, mass=nodes_1[0], momentum=nodes_1,
                        energy=nodes_1[1])
    restart_data = {"x": nodes_1[0],
                    "nodes": nodes_1,
                    "cv": cv,
                    "string": "hello",
                    "numba": 5}

    # map it to mesh2
    rst_msh2 = remap_dofarrays_in_structure(
        actx, restart_data, mesh_1, mesh_2, interp_info=interp_info_12)

    intrp_tol = 1e-15
    assert np.max(np.abs(actx.to_numpy(nodes_2[0] - rst_msh2["x"])[0])) < intrp_tol

    for idim in range(dim):
        assert np.max(np.abs(actx.to_numpy(
            nodes_2[idim] - rst_msh2["nodes"][idim])[0])) < intrp_tol

    cv2 = rst_msh2["cv"]
    assert np.max(np.abs(actx.to_numpy(nodes_2[0] - cv2.mass)[0])) < intrp_tol
    for idim in range(dim):
        assert np.max(np.abs(actx.to_numpy(
            nodes_2[idim] - cv2.momentum[idim])[0])) < intrp_tol
    assert np.max(np.abs(actx.to_numpy(nodes_2[1] - cv2.energy)[0])) < intrp_tol

    assert rst_msh2["string"] == "hello"
    assert rst_msh2["numba"] == 5
