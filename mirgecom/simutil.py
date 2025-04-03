"""Provide some utilities for building simulation applications.

General utilities
-----------------

.. autofunction:: check_step
.. autofunction:: get_sim_timestep
.. autofunction:: write_visfile
.. autofunction:: global_reduce
.. autofunction:: get_reasonable_memory_pool

Diagnostic utilities
--------------------

.. autofunction:: compare_fluid_solutions
.. autofunction:: componentwise_norms
.. autofunction:: max_component_norm
.. autofunction:: check_naninf_local
.. autofunction:: check_range_local
.. autofunction:: boundary_report

Mesh and element utilities
--------------------------

.. autofunction:: geometric_mesh_partitioner
.. autofunction:: distribute_mesh
.. autofunction:: get_number_of_tetrahedron_nodes
.. autofunction:: get_box_mesh
.. autofunction:: interdecomposition_mapping
.. autofunction:: interdecomposition_overlap
.. autofunction:: invert_decomp
.. autofunction:: multivolume_interdecomposition_overlap
.. autofunction:: copy_mapped_dof_array_data
.. autofunction:: inverse_element_connectivity
.. autofunction:: apply_elemental_interpolation
.. autofunction:: remap_dofarrays_in_structure

Simulation support utilities
----------------------------

.. autofunction:: configurate

File comparison utilities
-------------------------

.. autofunction:: compare_files_vtu
.. autofunction:: compare_files_xdmf
.. autofunction:: compare_files_hdf5

Exceptions
----------

.. autoclass:: SimulationConfigurationError
.. autoclass:: ApplicationOptionsError
.. autoclass:: SimulationRuntimeError
"""

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
import logging
import os
import pickle
from functools import partial
from typing import Dict, List, Optional
from contextlib import contextmanager
from dataclasses import (
    dataclass, fields,
    is_dataclass
)

from logpyle import IntervalTimer

import grudge.op as op
import numpy as np
import pyopencl as cl

from arraycontext import tag_axes
from meshmode.transform_metadata import (
    DiscretizationElementAxisTag,
    DiscretizationDOFAxisTag
)
from arraycontext import flatten, map_array_container
from grudge.discretization import (
    DiscretizationCollection,
    PartID
)
from grudge.dof_desc import DD_VOLUME_ALL
from meshmode.dof_array import DOFArray
from pytools.obj_array import make_obj_array
from collections import defaultdict, OrderedDict
from mirgecom.utils import normalize_boundaries
from mirgecom.viscous import get_viscous_timestep
from scipy.spatial import cKDTree


logger = logging.getLogger(__name__)


class SimulationConfigurationError(RuntimeError):
    """Simulation physics configuration or parameters error."""

    pass


class SimulationRuntimeError(RuntimeError):
    """General simulation runtime error."""

    pass


class ApplicationOptionsError(RuntimeError):
    """Application command-line options error."""

    pass


def get_number_of_tetrahedron_nodes(dim, order, include_faces=False):
    """Get number of nodes (modes) in *dim* Tetrahedron of *order*."""
    # number of {nodes, modes} see e.g.:
    # JSH/TW Nodal DG Methods, Section 10.1
    # DOI: 10.1007/978-0-387-72067-8
    from math import factorial
    nnodes = int(factorial(dim+order) / (factorial(dim) * factorial(order)))
    if include_faces:
        nnodes = nnodes + (dim+1)*get_number_of_tetrahedron_nodes(dim-1, order)
    return nnodes


def inverse_element_connectivity(mesh):
    """
    Invert the element connectivity of a mesh to map global vertex IDs
    to the elements and local indices that reference them.

    Parameters
    ----------
    mesh: meshmode.mesh.Mesh
        The mesh containing element groups and their vertex indices.

    Returns
    -------
    dict
        A mapping from global vertex ID to a list of (grp_id, el_id, local_vertex_id)
        tuples indicating all locations in the mesh where the vertex is used.
    """
    from collections import defaultdict

    inverted = defaultdict(list)
    for grp_id, grp in enumerate(mesh.groups):
        # shape: (nelements, nvertices_per_element)
        vertex_indices = grp.vertex_indices
        for el_id, element_vertices in enumerate(vertex_indices):
            for local_vertex_id, global_vertex_id in enumerate(element_vertices):
                inverted[global_vertex_id].append((grp_id, el_id, local_vertex_id))

    return dict(inverted)


def compute_vertex_averages(actx, dofarray, inverted_conn):
    """
    Compute average values at each global vertex from DOFArray data.

    Parameters
    ----------
        actx: ArrayContext
            The array context for accessing DOFArray data.
        dofarray: DOFArray
            The data to average.
        inverted_conn: dict[int, list[tuple[int, int, int]]]
            Maps global vertex IDs to (group_id, element_id, local_vertex_id).

    Returns
    -------
        np.ndarray:
            Averaged values per global vertex.
            Indexed by global vertex ID.
    """
    # N-Tuple of numpy arrays where N = ngroups
    arrays_by_group = actx.to_numpy(dofarray)

    nvertices = len(inverted_conn)
    sum_per_vertex = np.zeros(nvertices)
    count_per_vertex = np.zeros(nvertices, dtype=int)

    for global_vertex, locations in inverted_conn.items():
        for (grp_id, el_id, local_id) in locations:
            sum_per_vertex[global_vertex] += arrays_by_group[grp_id][el_id, local_id]
            count_per_vertex[global_vertex] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.zeros_like(sum_per_vertex)
        mask = count_per_vertex > 0
        avg[mask] = sum_per_vertex[mask] / count_per_vertex[mask]

    return avg


def scatter_vertex_values_to_dofarray(actx, template_dofarray,
                                      inverted_conn, vertex_vals):
    """
    Scatter global vertex values into a new DOFArray.

    Parameters
    ----------
        actx: ArrayContext
            The array context for building new DOFArrays.
        template_dofarray: DOFArray
            Used to get the shape and group structure for the new DOFArray.
        inverted_conn: dict[int, list[tuple[int, int, int]]]
            Maps global vertex IDs to (group_id, element_id, local_vertex_id).
        vertex_vals: np.ndarray
            Values at each global vertex, indexed by global vertex ID.

    Returns
    -------
        DOFArray:
            New DOFArray with each el node filled with its vertex value.
    """
    arrays_by_group = actx.to_numpy(template_dofarray)
    new_arrays = []

    # Allocate empty arrays matching shape of each group
    for group_array in arrays_by_group:
        new_arrays.append(0.*group_array)

    # Scatter the vertex average to every occurrence of that vertex
    for global_vertex_id, locations in inverted_conn.items():
        value = vertex_vals[global_vertex_id]
        for (grp_id, el_id, local_id) in locations:
            new_arrays[grp_id][el_id, local_id] = value

    # Convert numpy arrays to dofarrays
    group_arrays = tuple(new_arrays)
    dary = actx.from_numpy(DOFArray(actx, group_arrays))

    return dary


def quad_shape_functions(xi, eta):
    """
    Compute bilinear shape functions for a 4-node quad at (xi, eta).
    Reference coordinates are in [-1, 1]².
    """
    n_quad = np.zeros(4)
    n_quad[0] = 0.25 * (1 - xi) * (1 - eta)
    n_quad[1] = 0.25 * (1 + xi) * (1 - eta)
    n_quad[2] = 0.25 * (1 + xi) * (1 + eta)
    n_quad[3] = 0.25 * (1 - xi) * (1 + eta)
    return n_quad


def quad_shape_deriv(xi, eta):
    """
    Compute bilinear shape functions derivs
    for a 4-node quad at (xi, eta).
    Reference coordinates are in [-1, 1]^2.
    """
    # Shape function derivatives
    dn_dxi = np.array([
        [-0.25 * (1 - eta), -0.25 * (1 - xi)],
        [0.25 * (1 - eta), -0.25 * (1 + xi)],
        [0.25 * (1 + eta),  0.25 * (1 + xi)],
        [-0.25 * (1 + eta),  0.25 * (1 - xi)],
    ])
    return dn_dxi


def hex_shape_functions(xi, eta, zeta):
    """
    Trilinear shape functions for hexahedra
    Reference coords: xi, eta, zeta in [-1, 1]^3
    """
    node_signs = [
        (-1, -1, -1),
        (1, -1, -1),
        (1,  1, -1),
        (-1,  1, -1),
        (-1, -1,  1),
        (1, -1,  1),
        (1,  1,  1),
        (-1,  1,  1)
    ]
    n_hex = np.array([
        0.125 * (1 + sx * xi) * (1 + sy * eta) * (1 + sz * zeta)
        for sx, sy, sz in node_signs
    ])
    return n_hex


def hex_shape_deriv(xi, eta, zeta):
    """
    Trilinear shape derivs (8-node hexahedron).
    Reference coords: xi, eta, zeta in [-1, 1]^3
    """
    node_signs = [
        (-1, -1, -1),
        (1, -1, -1),
        (1,  1, -1),
        (-1,  1, -1),
        (-1, -1,  1),
        (1, -1,  1),
        (1,  1,  1),
        (-1,  1,  1)
    ]
    dn_hex = np.zeros((8, 3))
    for i, (sx, sy, sz) in enumerate(node_signs):
        dn_hex[i, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
        dn_hex[i, 1] = 0.125 * (1 + sx * xi) * sy * (1 + sz * zeta)
        dn_hex[i, 2] = 0.125 * (1 + sx * xi) * (1 + sy * eta) * sz
    return dn_hex


def el_shape_functions(el_coords):
    """Wrap shape functions."""
    dim = len(el_coords)
    if dim == 3:
        return hex_shape_functions(el_coords[0], el_coords[1], el_coords[2])
    elif dim == 2:
        return quad_shape_functions(el_coords[0], el_coords[1])
    else:
        raise ValueError("Invalid dimension.")
    return None


def el_shape_deriv(el_coords):

    dim = len(el_coords)
    if dim == 3:
        return hex_shape_deriv(el_coords[0], el_coords[1], el_coords[2])
    elif dim == 2:
        return quad_shape_deriv(el_coords[0], el_coords[1])
    else:
        raise ValueError("Invalid dimension.")
    return None


def inverse_map_quad_gn(el_nodes, target_point, tol=1e-10,
                        max_iter=15, retries=10):
    """
    Perform Gauss-Newton minimization to map a physical point
    to reference (xi, eta) coordinates inside a quad element.

    Parameters
    ----------
        el_nodes: (4, 2) array of the physical quad vertices
        target_point: (2,) array, physical location
        tol: convergence threshold on residual norm
        max_iter: max number of Gauss-Newton iterations
        retries: number of random initial guesses

    Returns
    -------
        (xi, eta) if successful, or None if all retries failed
    """
    # Permuting the node ordering to the canonical element
    el_nodes = el_nodes[[0, 1, 3, 2]]
    for attempt in range(retries):  # noqa
        # Start at random-ish location inside reference element
        xi, eta = np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3)
        # print(f"initial guess{attempt=}: xi={xi}, eta={eta}")

        for _ in range(max_iter):
            # Shape functions
            n_el = el_shape_functions([xi, eta])
            # Shape function derivatives
            dn_dxi = el_shape_deriv([xi, eta])

            # Mapped physical point at (xi, eta)
            x = np.dot(n_el, el_nodes)  # shape (2,)
            f_resid = x - target_point  # resid shape (2,)

            # build
            jac = np.zeros((2, 2))
            for i in range(4):
                jac[0, 0] += dn_dxi[i, 0] * el_nodes[i, 0]
                jac[1, 0] += dn_dxi[i, 0] * el_nodes[i, 1]
                jac[0, 1] += dn_dxi[i, 1] * el_nodes[i, 0]
                jac[1, 1] += dn_dxi[i, 1] * el_nodes[i, 1]

            j_t_j = jac.T @ jac
            j_t_f = jac.T @ f_resid

            try:
                delta = np.linalg.solve(j_t_j, -j_t_f)
            except np.linalg.LinAlgError:
                # unlucky: singular matrix, try another guess
                break

            xi += delta[0]
            eta += delta[1]

            if np.linalg.norm(f_resid) < tol:
                return xi, eta

    return None


def inverse_map_hex_gn(el_nodes, target_point, tol=1e-10,
                       max_iter=20, retries=10):
    """
    Perform Gauss-Newton minimization to map a physical point to reference
    coordinates (xi, eta, zeta) inside a hex element.

    Parameters
    ----------
        el_nodes: (8, 3) array of the physical HEX8 node coordinates
        target_point: (3,) array of the physical location
        tol: convergence threshold on residual norm
        max_iter: max number of Gauss-Newton iterations
        retries: number of random initial guesses

    Returns
    -------
        (xi, eta, zeta) if successful, or None if all retries failed
    """
    # Permuting the node ordering to the canonical element
    el_nodes = el_nodes[[0, 1, 3, 2, 4, 5, 7, 6]]

    for attempt in range(retries):  # noqa
        xi, eta, zeta = np.random.uniform(-0.3, 0.3, size=3)

        for _ in range(max_iter):
            # Shape functions N_i(xi, eta, zeta)
            n_el = el_shape_functions([xi, eta, zeta])

            # Residual: x(xi, eta, zeta) - target
            x = np.dot(n_el, el_nodes)
            f_resid = x - target_point

            # Shape function derivatives
            dn_dref = el_shape_deriv([xi, eta, zeta])

            # Build 3x3 Jacobian
            jac = np.zeros((3, 3))
            for i in range(8):
                for j in range(3):
                    for k in range(3):
                        jac[k, j] += dn_dref[i, j] * el_nodes[i, k]

            j_t_j = jac.T @ jac
            j_t_f = jac.T @ f_resid

            try:
                delta = np.linalg.solve(j_t_j, -j_t_f)
            except np.linalg.LinAlgError:
                # unlucky: try another initial guess
                break

            xi += delta[0]
            eta += delta[1]
            zeta += delta[2]

            if np.linalg.norm(f_resid) < tol:
                return xi, eta, zeta

    return None


def inverse_map_quad(el_nodes, target_point, tol=1e-13, max_iter=15, retries=10):
    """
    Newton iteration to map physical point to reference coordinates (xi, eta).
    Tries multiple initial guesses if needed.
    """
    # Permuting the node ordering to the canonical element
    el_nodes = el_nodes[[0, 1, 3, 2]]
    for attempts in range(retries):  # noqa
        xi = np.random.uniform(-0.3, 0.3)
        eta = np.random.uniform(-0.3, 0.3)

        for _ in range(max_iter):
            n_el = el_shape_functions([xi, eta])
            dn_dxi = el_shape_deriv([xi, eta])

            x = np.dot(n_el, el_nodes)
            f_resid = x - target_point

            jac = np.zeros((2, 2))
            for i in range(4):
                jac[0, 0] += dn_dxi[i, 0] * el_nodes[i, 0]
                jac[1, 0] += dn_dxi[i, 0] * el_nodes[i, 1]
                jac[0, 1] += dn_dxi[i, 1] * el_nodes[i, 0]
                jac[1, 1] += dn_dxi[i, 1] * el_nodes[i, 1]

            try:
                delta = np.linalg.solve(jac, -f_resid)
            except np.linalg.LinAlgError:
                break  # try next random initial guess

            xi += delta[0]
            eta += delta[1]

            if np.linalg.norm(delta) < tol:
                return xi, eta

    return None  # all attempts failed


def inverse_map_hex(el_nodes, target_point, tol=1e-13, max_iter=15, retries=10):
    """
    Newton iteration to map physical point to reference coordinates (xi, eta, zeta).
    Tries multiple random starting points if needed.

    Parameters
    ----------
        el_nodes: ndarray (8, 3)
            Physical coordinates of the 8 hexahedron nodes.
        target_point: ndarray (3,)
            Physical coordinates of the point to map.

    Returns
    -------
        (xi, eta, zeta) tuple if successful, otherwise None.
    """
    # Permuting the node ordering to the canonical element
    el_nodes = el_nodes[[0, 1, 3, 2, 4, 5, 7, 6]]

    for attempt in range(retries):  # noqa
        xi, eta, zeta = np.random.uniform(-0.3, 0.3, size=3)

        for _ in range(max_iter):
            # Shape functions
            n_el = el_shape_functions([xi, eta, zeta])

            # Physical point at current (xi, eta, zeta)
            x = np.dot(n_el, el_nodes)
            f_resid = x - target_point

            # Derivatives: dN_i/dxi, deta, dzeta
            dn_dref = el_shape_deriv([xi, eta, zeta])

            # Build 3x3 Jacobian
            jac = np.zeros((3, 3))
            for i in range(8):
                for j in range(3):      # ref-space direction (xi, eta, zeta)
                    for k in range(3):  # physical direction (x, y, z)
                        jac[k, j] += dn_dref[i, j] * el_nodes[i, k]

            try:
                delta = np.linalg.solve(jac, -f_resid)
            except np.linalg.LinAlgError:
                break  # try another random initial guess

            xi += delta[0]
            eta += delta[1]
            zeta += delta[2]

            if np.linalg.norm(delta) < tol:
                return xi, eta, zeta

    return None


def point_in_quad_reference(xi, eta, tol=1e-8):
    """
    Check if reference coords (xi, eta) are inside the reference square [-1, 1]².
    """
    return -1 - tol <= xi <= 1 + tol and -1 - tol <= eta <= 1 + tol


def point_in_reference_coords(ref_coords, element_type, tol=1e-8):
    """
    Check if the given reference coordinates are inside the canonical reference el.

    Parameters
    ----------
        ref_coords: ndarray-like
            Reference coordinates (e.g., [xi, eta] or [xi, eta, zeta])
        element_type: str
            One of: "quad", "hex", "tri", "tet"
        tol: float
            Tolerance for bounds checking.

    Returns
    -------
        bool
    """
    ref_coords = np.asarray(ref_coords)

    if element_type == "quad" or element_type == "hex":
        return np.all((-1 - tol <= ref_coords) & (ref_coords <= 1 + tol))
    if element_type == "tpe":
        return np.all((-1 - tol <= ref_coords) & (ref_coords <= 1 + tol))

    elif element_type == "tri":
        r, s = ref_coords
        return (
            -tol <= r <= 1 + tol
            and -tol <= s <= 1 + tol
            and r + s <= 1 + tol
        )

    elif element_type == "tet":
        r, s, t = ref_coords
        return (
            -tol <= r <= 1 + tol
            and -tol <= s <= 1 + tol
            and -tol <= t <= 1 + tol
            and r + s + t <= 1 + tol
        )

    else:
        raise ValueError(f"Unknown element type '{element_type}'")


@dataclass
class ElementalInterpolationInfo:
    src_element_ids: np.ndarray  # shape: (ntgt_nodes,)
    ref_coords: np.ndarray  # shape: (ntgt_nodes, dim)
    fallback_mask: np.ndarray  # shape: (ntgt_nodes,), bool
    fallback_indices: list[int]  # indices of points that failed


def build_element_centroid_kdtree(src_vertices, src_elements):
    el_centroids = np.array([
        np.mean(src_vertices[conn], axis=0) for conn in src_elements
    ])
    return cKDTree(el_centroids), el_centroids


def recover_interp_fallbacks(interp_info, mesh1, mesh2,
                             target_point_map=None, inverse_map_fn=None,
                             meter_level=100.):
    """
    Re-attempt point location for fallback points using
    Gauss-Newton instead of Newton.

    Parameters:
        interp_info: ElementalInterpolationInfo
        mesh1: source mesh (meshmode mesh)
        mesh2: target mesh (meshmode mesh)
        inverse_map_gauss_newton_fn: optional function to override solver

    Returns:
        Updated ElementalInterpolationInfo with fallbacks resolved where possible.
    """
    src_vertices = mesh1.vertices.T
    tgt_vertices = mesh2.vertices.T
    src_elements = mesh1.groups[0].vertex_indices
    nsrc_els = len(src_elements)
    dim = src_vertices.shape[1]
    fallback_indices = interp_info.fallback_indices
    n_tgt_nodes = len(fallback_indices)
    updated_src_element_ids = interp_info.src_element_ids.copy()
    updated_ref_coords = interp_info.ref_coords.copy()
    updated_fallback_mask = interp_info.fallback_mask.copy()
    eltype = "quad" if dim == 2 else "hex"
    # Pick solver based on dimension
    if inverse_map_fn is None:
        inverse_map_fn = (
            inverse_map_quad_gn if dim == 2 else inverse_map_hex_gn
        )

    tree, centroids = build_element_centroid_kdtree(src_vertices, src_elements)
    if meter_level < 100:
        print(f"GN: Finding {n_tgt_nodes} target points "
              f"in {nsrc_els} source elements.")

    # from tqdm import tqdm
    # for i in tqdm(fallback_indices, desc="Recovering fallbacks with Gauss-Newton"):
    for cnt, i in enumerate(fallback_indices):
        pct = 100.*float(cnt / n_tgt_nodes)
        cur_meter = meter_level
        if pct >= cur_meter:
            print(f"{pct}% target points searched.")
            cur_meter = pct + meter_level
        x_tgt = tgt_vertices[i]
        if target_point_map is not None:
            x_tgt = target_point_map(x_tgt)

        # Query k nearest neighbors
        k = 30
        _, candidate_ids = tree.query(x_tgt, k=k)

        # for el_id, conn in enumerate(src_elements):
        for el_id in candidate_ids:
            conn = src_elements[el_id]
            el_nodes = src_vertices[conn]
            ref = inverse_map_fn(el_nodes, x_tgt)
            if ref is not None and point_in_reference_coords(ref, eltype):
                updated_src_element_ids[i] = el_id
                updated_ref_coords[i] = ref
                updated_fallback_mask[i] = False
                break  # stop at first match

    new_fallbacks = [i for i in fallback_indices if updated_fallback_mask[i]]

    return ElementalInterpolationInfo(
        src_element_ids=updated_src_element_ids,
        ref_coords=updated_ref_coords,
        fallback_mask=updated_fallback_mask,
        fallback_indices=new_fallbacks
    )


def build_elemental_interpolation_info(mesh1, mesh2, target_point_map=None,
                                       meter_level=100.0):
    """
    Construct interpolation metadata mapping the vertices of mesh2 to
    reference coordinates in elements of mesh1.

    This routine locates each vertex of mesh2 within an element of mesh1,
    computing the element ID and corresponding element-local reference
    coordinates for use in later interpolation.

    Parameters
    ----------
    mesh1: meshmode.mesh.Mesh
        The source mesh from which field data will be interpolated.
    mesh2: meshmode.mesh.Mesh
        The target mesh whose vertex locations are used for interpolation.

    Returns
    -------
    interp_info: ElementInterpolationInfo
        A structure containing element IDs, reference coordinates,
        and fallback indicators for each vertex of the target mesh.
    """
    src_vertices = mesh1.vertices.T  # (nverts, dim)
    tgt_vertices = mesh2.vertices.T  # (nverts, dim)
    # assuming single group for now
    src_elements = mesh1.groups[0].vertex_indices  # (nelements, nvertices_per_el)
    nsrc_els = len(src_elements)

    dim = src_vertices.shape[1]
    n_tgt_nodes = tgt_vertices.shape[0]

    src_element_ids = np.full(n_tgt_nodes, -1, dtype=np.int32)
    ref_coords = np.full((n_tgt_nodes, dim), np.nan, dtype=np.float64)
    fallback_mask = np.zeros(n_tgt_nodes, dtype=bool)
    fallback_indices = []
    tree, centroids = build_element_centroid_kdtree(src_vertices, src_elements)

    if meter_level < 100:
        print(f"N: Finding {n_tgt_nodes} target points in "
              f"{nsrc_els} source elements.")

    # from tqdm import tqdm
    # for i in tqdm(range(n_tgt_nodes), desc="Locating target verts in source mesh"):
    for i in range(n_tgt_nodes):
        pct = 100.*float(i / n_tgt_nodes)
        cur_meter = meter_level
        if pct >= cur_meter:
            print(f"{pct}% target points searched.")
            cur_meter = pct + meter_level

        x_tgt = tgt_vertices[i]
        if target_point_map is not None:
            x_tgt = target_point_map(x_tgt)

        # Query k nearest neighbors
        k = 20
        _, candidate_ids = tree.query(x_tgt, k=k)

        found = False
        # for el_id, conn in enumerate(src_elements):
        for el_id in candidate_ids:
            # el_nodes = src_vertices[conn]  # shape: (nvert_el, dim)
            conn = src_elements[el_id]
            # print(f"{conn=}")
            el_nodes = src_vertices[conn]  # shape: (nvert_el, dim)

            if dim == 2:
                ref = inverse_map_quad(el_nodes, x_tgt)
                if ref is None or not point_in_reference_coords(ref, "quad"):
                    continue
            else:
                ref = inverse_map_hex(el_nodes, x_tgt)
                if ref is None or not point_in_reference_coords(ref, "hex"):
                    continue

            # Point found
            src_element_ids[i] = el_id
            ref_coords[i] = ref
            found = True
            break

        if not found:
            fallback_mask[i] = True
            fallback_indices.append(i)

    interp_info = ElementalInterpolationInfo(
        src_element_ids=src_element_ids,
        ref_coords=ref_coords,
        fallback_mask=fallback_mask,
        fallback_indices=fallback_indices
    )

    return interp_info


def apply_elemental_interpolation(src_field, interp_info):
    """
    Evaluate the source field at target points using interpolation info.

    Parameters:
        src_field: DOFArray from mesh1
        interp_info: ElementalInterpolationInfo
        shape_function: function (xi, eta[, zeta]) -> shape values

    Returns:
        numpy array of shape (n_target_vertices,)
    """
    from warnings import warn
    result = np.empty(len(interp_info.src_element_ids))

    dim = interp_info.ref_coords.shape[1]
    for i, (el_id, ref) in enumerate(zip(interp_info.src_element_ids,
                                         interp_info.ref_coords)):

        if el_id < 0:
            warn("Setting unfound point solution to 0.")
            result[i] = 0.0
            continue

        if dim == 2:
            # going to permute the dofs to canonical order
            shape_vals = el_shape_functions(ref)
            perm = [0, 1, 3, 2]
        elif dim == 3:
            shape_vals = el_shape_functions(ref)
            perm = [0, 1, 3, 2, 4, 5, 7, 6]
        else:
            raise ValueError(f"Unsupported dim={dim}")

        el_dofs = src_field[el_id][perm]
        result[i] = np.dot(shape_vals, el_dofs)

    return result


def remap_dofarrays_in_structure(actx, struct, source_mesh, target_mesh,
                                 interp_info=None, target_point_map=None,
                                 volume_id=None, meter_level=100.):
    """
    Recursively remap all DOFArrays in a nested data structure from mesh1 to mesh2.

    Traverses the given data structure and applies mesh-to-mesh interpolation
    to any DOFArray it contains. Handles scalar DOFArrays, object arrays of
    DOFArrays, and dataclasses containing DOFArrays. Non-field data (e.g., ints,
    floats, strings) are passed through unchanged.

    Parameters
    ----------
    actx: arraycontext.ArrayContext
        The array context used for all array operations.
    struct: Any
        The input structure containing DOFArrays to be remapped. May be a nested
        combination of dicts, lists, tuples, dataclasses, or numpy object arrays.
    source_mesh: meshmode.mesh.Mesh
        The source mesh from which the DOFArrays were defined.
    target_mesh: meshmode.mesh.Mesh
        The target mesh to which the DOFArrays will be remapped.
    interp_info: ElementInterpolationInfo, optional
        Precomputed interpolation metadata. If not provided, it will be constructed
        internally using `build_elemental_interpolation_info`.

    Returns
    -------
    struct_mapped: Any
        A new structure with the same layout as the input, but with all
        DOFArrays remapped to mesh2.
    """

    from grudge import discretization as grudge_discr
    # from mirgecom.simutil import inverse_element_connectivity
    from mirgecom.discretization import create_discretization_collection
    from grudge.dof_desc import (
        DOFDesc,
        VolumeDomainTag,
        DISCR_TAG_BASE,
    )

    if isinstance(source_mesh, dict):
        if volume_id is None:
            raise ValueError("Data transfer for multivolume must specify volume_id.")
        source_mesh = source_mesh[volume_id]

    # Set up target discretization and template
    if isinstance(target_mesh, dict):
        if volume_id is None:
            raise ValueError("Data transfer for multivolume must specify volume_id.")
        multivol_dcoll = \
            create_discretization_collection(
                actx, volume_meshes=target_mesh, order=1)
        dd = DOFDesc(VolumeDomainTag(volume_id), DISCR_TAG_BASE)
        # grabs the x-component of the nodes for an example dof_array
        template_dofs = actx.thaw(multivol_dcoll.nodes(dd)[0])
        target_mesh = target_mesh[volume_id]
    else:
        # dcoll_2 = grudge_discr.DiscretizationCollection(actx, target_mesh, order=1)
        dcoll_2 = grudge_discr.DiscretizationCollection(actx, target_mesh, order=1)
        template_dofs = actx.thaw(dcoll_2.nodes()[0])
    iconn_2 = inverse_element_connectivity(target_mesh)

    # Precompute interpolation info once
    if interp_info is None:
        interp_info = build_elemental_interpolation_info(
            source_mesh, target_mesh, target_point_map=target_point_map,
            meter_level=meter_level)
        if np.any(interp_info.fallback_mask):
            interp_info = recover_interp_fallbacks(
                interp_info, source_mesh, target_mesh,
                target_point_map=target_point_map,
                meter_level=meter_level)
        n_fallback = np.count_nonzero(interp_info.fallback_mask)
        # if np.any(interp_info.fallback_mask):
        if n_fallback > 0:
            # raise RuntimeError
            # ("Could not find some mesh2 vertices in mesh1 elems.")
            from warnings import warn
            warn(f"Could not find {n_fallback} target vertices in source mesh.")

    def _map(obj):
        if isinstance(obj, DOFArray):
            src_np = actx.to_numpy(obj)[0]
            mapped_vals = apply_elemental_interpolation(src_np, interp_info)
            return scatter_vertex_values_to_dofarray(actx, template_dofs,
                                                     iconn_2, mapped_vals)

        elif (isinstance(obj, np.ndarray) and obj.dtype == object
              and all(isinstance(x, DOFArray) for x in obj.flat)):
            # Object array of DOFArrays → use make_obj_array
            return make_obj_array([_map(x) for x in obj])

        elif isinstance(obj, (list, tuple)):
            return type(obj)(_map(o) for o in obj)

        elif isinstance(obj, dict):
            return {k: _map(v) for k, v in obj.items()}

        elif is_dataclass(obj):
            return type(obj)(**{
                f.name: _map(getattr(obj, f.name)) for f in fields(obj)
            })

        else:
            return obj  # Pass through anything else unchanged

    return _map(struct)


def get_box_mesh(dim, a, b, n, t=None, periodic=None,
                 tensor_product_elements=False, **kwargs):
    """
    Create a rectangular "box" like mesh with tagged boundary faces.

    The resulting mesh has boundary tags
    `"-i"` and `"+i"` for `i=1,...,dim`
    corresponding to lower and upper faces normal to coordinate dimension `i`.

    Parameters
    ----------
    dim: int
        The mesh topological dimension
    a: float or tuple
        The coordinates of the lower corner of the box. If scalar-valued, gets
        promoted to a uniform tuple.
    b: float or tuple
        The coordinates of the upper corner of the box. If scalar-valued, gets
        promoted to a uniform tuple.
    n: int or tuple
        The number of elements along a given dimension. If scalar-valued, gets
        promoted to a uniform tuple.
    t: str or None
        The mesh type. See
        :func:`meshmode.mesh.generation.generate_box_mesh` for details.
    periodic: bool or tuple or None
        Indicates whether the mesh is periodic in a given dimension. If
        scalar-valued, gets promoted to a uniform tuple.

    Returns
    -------
    :class:`meshmode.mesh.Mesh`
        The generated box mesh.
    """
    if np.isscalar(a):
        a = (a,)*dim
    if np.isscalar(b):
        b = (b,)*dim
    if np.isscalar(n):
        n = (n,)*dim
    if periodic is None:
        periodic = (False,)*dim
    elif np.isscalar(periodic):
        periodic = (periodic,)*dim
    if tensor_product_elements is None:
        tensor_product_elements = False

    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]

    from meshmode.mesh import TensorProductElementGroup
    group_cls = TensorProductElementGroup if tensor_product_elements else None
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, nelements_per_axis=n,
               boundary_tag_to_face=bttf,
               mesh_type=t, periodic=periodic, group_cls=group_cls,
               **kwargs)


def check_step(step, interval):
    """
    Check step number against a user-specified interval.

    Utility is used typically for visualization.

    - Negative numbers mean 'never visualize'.
    - Zero means 'always visualize'.

    Useful for checking whether the current step is an output step,
    or anything else that occurs on fixed intervals.
    """
    if interval == 0:
        return True
    elif interval < 0:
        return False
    elif step % interval == 0:
        return True
    return False


def get_sim_timestep(
        dcoll, state, t, dt, cfl, t_final=0.0, constant_cfl=False,
        local_dt=False, fluid_dd=DD_VOLUME_ALL):
    r"""Return the maximum stable timestep for a typical fluid simulation.

    This routine returns a constraint-limited timestep size for a fluid
    simulation.  The returned timestep will be constrained by the specified
    Courant-Friedrichs-Lewy number, *cfl*, and the simulation max simulated time
    limit, *t_final*, and subject to the user's optional settings.

    The local fluid timestep, $\delta{t}_l$, is computed by
    :func:`~mirgecom.viscous.get_viscous_timestep`.  Users are referred to that
    routine for the details of the local timestep.

    With the remaining simulation time $\Delta{t}_r =
    \left(\mathit{t\_final}-\mathit{t}\right)$, three modes are supported
    for the returned timestep, $\delta{t}$:

    - "Constant DT" mode (default): $\delta{t} = \mathbf{\text{min}}
      \left(\textit{dt},~\Delta{t}_r\right)$
    - "Constant CFL" mode (constant_cfl=True): $\delta{t} =
      \mathbf{\text{min}}\left(\mathbf{\text{global\_min}}\left(\delta{t}\_l\right)
      ,~\Delta{t}_r\right)$
    - "Local DT" mode (local_dt=True): $\delta{t} = \mathbf{\text{cell\_local\_min}}
      \left(\delta{t}_l\right)$

    Note that for "Local DT" mode, *t_final* is ignored, and a
    :class:`~meshmode.dof_array.DOFArray` containing the local *cfl*-limited
    timestep, where $\mathbf{\text{cell\_local\_min}}\left(\delta{t}\_l\right)$ is
    defined as the minimum over the cell collocation points. This mode is useful for
    stepping to convergence of steady-state solutions.

    .. important::
        For "Constant CFL" mode, this routine calls the collective
        :func:`~grudge.op.nodal_min` on the inside which involves MPI collective
        functions.  Thus all MPI ranks on the
        :class:`~grudge.discretization.DiscretizationCollection` must call this
        routine collectively when using "Constant CFL" mode.

    Parameters
    ----------
    dcoll: :class:`~grudge.discretization.DiscretizationCollection`
        The DG discretization collection to use
    state: :class:`~mirgecom.gas_model.FluidState`
        The full fluid conserved and thermal state
    t: float
        Current time
    t_final: float
        Final time
    dt: float
        The current timestep
    cfl: float
        The current CFL number
    constant_cfl: bool
        True if running constant CFL mode
    local_dt: bool
        True if running local DT mode. False by default.
    fluid_dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *state* lives. Must be a
        volume on the base discretization.

    Returns
    -------
    float or :class:`~meshmode.dof_array.DOFArray`
        The global maximum stable DT based on a viscous fluid.
    """
    actx = state.array_context

    if local_dt:
        ones = actx.np.zeros_like(state.cv.mass) + 1.0
        vdt = get_viscous_timestep(dcoll, state, dd=fluid_dd)
        emin = op.elementwise_min(dcoll, fluid_dd, vdt)
        return cfl * ones * emin

    my_dt = dt
    t_remaining = max(0, t_final - t)
    if constant_cfl:
        my_dt = state.array_context.to_numpy(
            cfl * op.nodal_min(
                dcoll, fluid_dd,
                get_viscous_timestep(dcoll=dcoll, state=state, dd=fluid_dd)))[()]

    return min(t_remaining, my_dt)


def write_visfile(dcoll, io_fields, visualizer, vizname,
                  step=0, t=0, overwrite=False, vis_timer=None,
                  comm=None):
    """Write parallel VTK output for the fields specified in *io_fields*.

    This routine writes a parallel-compatible unstructured VTK visualization
    file set in (vtu/pvtu) format. One file per MPI rank is written with the
    following naming convention: *vizname*_*step*_<mpi-rank>.vtu, and a single
    file manifest with naming convention: *vizname*_*step*.pvtu.  Users are
    advised to visualize the data using _Paraview_, _VisIt_, or other
    VTU-compatible visualization software by opening the PVTU files.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    visualizer:
        A :class:`meshmode.discretization.visualization.Visualizer`
        VTK output object.
    io_fields:
        List of tuples indicating the (name, data) for each field to write.
    vizname: str
        Root part of the visualization file name to write
    step: int
        The step number to use in the file names
    t: float
        The simulation time to write into the visualization files
    overwrite: bool
        Option whether to overwrite existing files (True) or fail if files
        exist (False=default).
    comm:
        An MPI Communicator is required for parallel writes. If no
        mpi_communicator is provided, then the write is assumed to be serial.
        (deprecated behavior: pull an MPI communicator from the discretization
        collection.  This will stop working in Fall 2022.)
    """
    from contextlib import nullcontext

    from mirgecom.io import make_par_fname, make_rank_fname

    if comm is None:  # None is OK for serial writes!
        comm = dcoll.mpi_communicator
        if comm is not None:  # It's *not* OK to get comm from dcoll
            from warnings import warn
            warn("Using `write_visfile` in parallel without an MPI communicator is "
                 "deprecated and will stop working in Fall 2022. For parallel "
                 "writes, specify an MPI communicator with the `mpi_communicator` "
                 "argument.")
    rank = 0

    if comm is not None:
        rank = comm.Get_rank()

    rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)

    if rank == 0:
        import os
        viz_dir = os.path.dirname(rank_fn)
        if viz_dir and not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

    if comm is not None:
        comm.barrier()

    if vis_timer:
        ctm = vis_timer.start_sub_timer()
    else:
        ctm = nullcontext()

    with ctm:
        visualizer.write_parallel_vtk_file(
            comm, rank_fn, io_fields,
            overwrite=overwrite,
            par_manifest_filename=make_par_fname(
                basename=vizname, step=step, t=t
            )
        )


def global_reduce(local_values, op, *, comm=None):
    """Perform a global reduction (allreduce if MPI comm is provided).

    This routine is a convenience wrapper for the MPI AllReduce operation
    that also works outside of an MPI context.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    local_values:
        The (:mod:`mpi4py`-compatible) value or array of values on which the
        reduction operation is to be performed.

    op: str
        Reduction operation to be performed. Must be one of "min", "max", "sum",
        "prod", "lor", or "land".

    comm:
        Optional parameter specifying the MPI communicator on which the
        reduction operation (if any) is to be performed

    Returns
    -------
    Any ( like *local_values* )
        Returns the result of the reduction operation on *local_values*
    """
    if comm is not None:
        from mpi4py import MPI
        op_to_mpi_op = {
            "min": MPI.MIN,
            "max": MPI.MAX,
            "sum": MPI.SUM,
            "prod": MPI.PROD,
            "lor": MPI.LOR,
            "land": MPI.LAND,
        }
        return comm.allreduce(local_values, op=op_to_mpi_op[op])
    else:
        if np.ndim(local_values) == 0:
            return local_values
        else:
            op_to_numpy_func = {
                "min": np.minimum,
                "max": np.maximum,
                "sum": np.add,
                "prod": np.multiply,
                "lor": np.logical_or,
                "land": np.logical_and,
            }
            from functools import reduce
            return reduce(op_to_numpy_func[op], local_values)


def allsync(local_values, comm=None, op=None):
    """
    Perform allreduce if MPI comm is provided.

    Deprecated. Do not use in new code.
    """
    from warnings import warn
    warn("allsync is deprecated and will disappear in Q1 2022. "
         "Use global_reduce instead.", DeprecationWarning, stacklevel=2)

    if comm is None:
        return local_values

    from mpi4py import MPI

    if op is None:
        op = MPI.MAX

    if op == MPI.MIN:
        op_string = "min"
    elif op == MPI.MAX:
        op_string = "max"
    elif op == MPI.SUM:
        op_string = "sum"
    elif op == MPI.PROD:
        op_string = "prod"
    elif op == MPI.LOR:
        op_string = "lor"
    elif op == MPI.LAND:
        op_string = "land"
    else:
        raise ValueError(f"Unrecognized MPI reduce op {op}.")

    return global_reduce(local_values, op_string, comm=comm)


def check_range_local(dcoll: DiscretizationCollection, dd: str, field: DOFArray,
                      min_value: float, max_value: float) -> List[float]:
    """Return the values that are outside the range [min_value, max_value]."""
    actx = field.array_context
    local_min = actx.to_numpy(  # type: ignore[union-attr]
        op.nodal_min_loc(dcoll, dd, field)).item()
    local_max = actx.to_numpy(  # type: ignore[union-attr]
        op.nodal_max_loc(dcoll, dd, field)).item()

    failing_values = []

    if local_min < min_value:
        failing_values.append(local_min)
    if local_max > max_value:
        failing_values.append(local_max)

    return failing_values


def check_naninf_local(dcoll: DiscretizationCollection, dd: str,
                       field: DOFArray) -> bool:
    """Return True if there are any NaNs or Infs in the field."""
    actx = field.array_context
    s = actx.to_numpy(op.nodal_sum_loc(dcoll, dd, field))
    return not np.isfinite(s)


def compare_fluid_solutions(dcoll, red_state, blue_state, *, dd=DD_VOLUME_ALL):
    """Return inf norm of (*red_state* - *blue_state*) for each component.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    # added tag_axes calls to eliminate fallback warnings at compile time
    actx = red_state.array_context
    resid = tag_axes(actx,
                     {
                         0: DiscretizationElementAxisTag(),
                         1: DiscretizationDOFAxisTag()
                     }, red_state - blue_state)
    resid_errs = actx.to_numpy(
        tag_axes(actx,
                 {
                     0: DiscretizationElementAxisTag()
                 },
                 flatten(
                     componentwise_norms(dcoll, resid, order=np.inf, dd=dd), actx)
                 )
    )

    return resid_errs.tolist()


def componentwise_norms(dcoll, fields, order=np.inf, *, dd=DD_VOLUME_ALL):
    """Return the *order*-norm for each component of *fields*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    if not isinstance(fields, DOFArray):
        return map_array_container(
            partial(componentwise_norms, dcoll, order=order, dd=dd), fields)
    if len(fields) > 0:
        return op.norm(dcoll, fields, order, dd=dd)
    else:
        # FIXME: This work-around for #575 can go away after #569
        return 0


def max_component_norm(dcoll, fields, order=np.inf, *, dd=DD_VOLUME_ALL):
    """Return the max *order*-norm over the components of *fields*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.
    """
    actx = fields.array_context
    return max(actx.to_numpy(flatten(
        componentwise_norms(dcoll, fields, order, dd=dd), actx)))


class PartitioningError(Exception):
    """Error tossed to indicate an error with domain decomposition."""

    pass


def assign_elements_to_volumes(mesh, volumes, debug=False):
    """
    Assigns elements in a mesh to user-defined volumes in precedence order,
    with a final fallback to the bounding box of the entire mesh (Bm).

    Parameters
    ----------
    mesh: meshmode.mesh.Mesh
        The unstructured mesh.
    volumes: dict
        A dictionary mapping volume IDs to bounding boxes of the form:
        {vol_id: (xmin, xmax, ymin, ymax, zmin, zmax), ...}
        The order in which they appear dictates precedence.
    debug: bool
        If True, prints debugging information.

    Returns
    -------
    volume_to_elements: dict
        A mapping from volume ID to a NumPy array of element indices.
    """
    # Extract element centroids
    mesh_verts = mesh.vertices
    dim = mesh.dim

    all_elem_group_centroids = []
    for group in mesh.groups:
        # (dim, nelements, nnodes_per_element)
        elem_group_coords = mesh_verts[:, group.vertex_indices]
        # (dim, nelements)
        elem_group_centroids = np.mean(elem_group_coords, axis=2)
        # (nelements, dim)
        all_elem_group_centroids.append(elem_group_centroids.T)

    # (total_nelements, dim)
    elem_centroids = np.concatenate(all_elem_group_centroids)
    total_nelements = len(elem_centroids)

    # Track assigned elements
    assigned_mask = np.zeros(total_nelements, dtype=bool)
    volume_to_elements = {vol_id: [] for vol_id in volumes}

    # Assign elements to volumes in order of precedence
    for vol_id, (vxn, vxx, vyn, vyx, vzn, vzx) in volumes.items():
        print(f"Finding elements in {vol_id}: {volumes[vol_id]=}")
        for e_idx, centroid in enumerate(elem_centroids):
            if assigned_mask[e_idx]:
                continue  # Skip already assigned elements

            x, y, z = centroid if dim == 3 else (*centroid, 0)  # Handle 2D case
            if ((vxn <= x <= vxx) and (vyn <= y <= vyx) and (vzn <= z <= vzx)):
                volume_to_elements[vol_id].append(e_idx)
                assigned_mask[e_idx] = True

    # Assign remaining elements to bb
    for e_idx in range(total_nelements):
        if not assigned_mask[e_idx]:
            print("Yikes! Unassigned element!")
            volume_to_elements["_Vol_0"].append(e_idx)
            assigned_mask[e_idx] = True

    # Convert lists to NumPy arrays
    for vol_id in volume_to_elements:
        volume_to_elements[vol_id] = np.array(volume_to_elements[vol_id], dtype=int)

    # Validation: Ensure all elements are assigned exactly once
    total_assigned = sum(len(elems) for elems in volume_to_elements.values())
    if total_assigned != total_nelements:
        raise ValueError("Mismatch in element count: Not all elements "
                         "were assigned!")

    # Debugging output
    if debug:
        print("Volume Assignment Order:")
        for vol_id in volumes:
            print(f"  {vol_id}")
        print("\nVolume Stats:")
        for vol_id, elems in volume_to_elements.items():
            bounds = volumes[vol_id]
            print(f"- Volume {vol_id}: {len(elems)} elements in {bounds}")

    return volume_to_elements


def compute_volume_partitions(volume_to_elements, npart, imbalance_tolerance,
                              debug=False):
    """
    Computes the number of partitions assigned to each volume while ensuring
    imbalance tolerance constraints and proper partition distribution.

    Parameters
    ----------
    volume_to_elements: dict
        A dictionary mapping volume IDs to lists of element indices.
    npart: int
        The total number of partitions available.
    imbalance_tolerance: float
        The maximum allowed imbalance beyond the ideal partition size.
    debug: bool
        Enables debugging output.

    Returns
    -------
    volume_partition_counts: dict
        A mapping of volume IDs to the number of partitions assigned to them.
    """
    total_nelements = sum(len(elems) for elems in volume_to_elements.values())
    pbar = total_nelements / npart  # Initial average partition size
    remaining_elements = total_nelements
    remaining_partitions = npart
    volume_partition_counts = {}

    if debug:
        print("Initial total elements:", total_nelements)
        print("Initial pbar (target elements per partition):", pbar)
        print("\nProcessing Volumes:\n")
    # Let's process them smallest to largest, to make sure everyone gets a part
    sorted_vols = sorted(volume_to_elements.items(), key=lambda v: len(v[1]))
    for vol_id, elements in sorted_vols:
        nelements = len(elements)

        if nelements <= (1 + imbalance_tolerance) * pbar:
            npart_vol = 1  # Assign a single partition if within imbalance tolerance
        else:
            npart_vol = min(remaining_partitions,
                            max(1, int(np.ceil(nelements / pbar))))
        nepp = int(nelements / npart_vol)
        if debug:
            print(f"Volume {vol_id}: {nelements} elements, {npart_vol} "
                  f"partitions, ~{nepp}/part")

        volume_partition_counts[vol_id] = npart_vol
        remaining_elements -= nelements
        remaining_partitions -= npart_vol

        if remaining_partitions <= 0 and remaining_elements > 0:
            raise ValueError("Ran out of partitions before processing all vols!")

        if remaining_partitions > 0:
            if remaining_elements <= 0:
                raise ValueError("Ran out of elements to partition!")
            pbar = remaining_elements / remaining_partitions
            if debug:
                print(f"  New remaining elements: {remaining_elements}")
                print(f"  New remaining partitions: {remaining_partitions}")
                print(f"  Updated pbar: {pbar}\n")

    if debug:
        print("\nFinal Partitioning Summary:")
        for vol_id, nparts in volume_partition_counts.items():
            nels = len(volume_to_elements[vol_id])
            pbar = nels / nparts
            print(f"  {vol_id}: {nparts} partitions {pbar=}")

    return volume_partition_counts


def get_longest_axis(vol_bounds):
    axes = {"X": (0, 1), "Y": (2, 3), "Z": (4, 5)}
    # return "X"  # hardcode to X
    return max(axes, key=lambda a: vol_bounds[axes[a][1]]
               - vol_bounds[axes[a][0]])


def geometric_mesh_partitioner(mesh, num_ranks=None, *, nranks_per_axis=None,
                               auto_balance=False, imbalance_tolerance=.01,
                               volumes=None, debug=False, part_axis=None):
    """Partition a mesh uniformly along the X coordinate axis.

    The intent is to partition the mesh uniformly along user-specified
    directions. In this current interation, the capability is demonstrated
    by splitting along the X axis.

    Parameters
    ----------
    mesh: :class:`meshmode.mesh.Mesh`
        The serial mesh to partition
    num_ranks: int
        The number of partitions to make (deprecated)
    nranks_per_axis: numpy.ndarray
        How many partitions per specified axis.
    auto_balance: bool
        Indicates whether to perform automatic balancing.  If true, the
        partitioner will try to balance the number of elements over
        the partitions.
    imbalance_tolerance: float
        If *auto_balance* is True, this parameter indicates the acceptable
        relative difference to the average number of elements per partition.
        It defaults to balance within 1%.
    debug: bool
        En/disable debugging/diagnostic print reporting.

    Returns
    -------
    elem_to_rank: numpy.ndarray
        Array indicating the MPI rank for each element
    """
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Partitioning mesh: {now}")

    mesh_dimension = mesh.dim
    if nranks_per_axis is None or num_ranks is not None:
        from warnings import warn
        warn("num_ranks is deprecated, use nranks_per_axis instead.")
        num_ranks = num_ranks or 1
        nranks_per_axis = np.ones(mesh_dimension, dtype=np.int32)
        nranks_per_axis[0] = num_ranks
    if len(nranks_per_axis) != mesh_dimension:
        raise ValueError("nranks_per_axis must match mesh dimension.")
    num_ranks = np.prod(nranks_per_axis)
    if np.prod(nranks_per_axis[1:]) != 1:
        raise NotImplementedError("geometric_mesh_partitioner currently only "
                                "supports partitioning in the X-dimension."
                                "(only nranks_per_axis[0] should be > 1).")

    mesh_verts = mesh.vertices
    x0, y0 = np.min(mesh_verts[:2], axis=1)
    x1, y1 = np.max(mesh_verts[:2], axis=1)
    z0 = 0.
    z1 = 0.
    if mesh_dimension == 3:
        z0, z1 = np.min(mesh_verts[2]), np.max(mesh_verts[2])

    vol_0_bounds = (x0, x1, y0, y1, z0, z1)
    if volumes is not None:
        volumes = OrderedDict(volumes)
        volumes["_Vol_0"] = vol_0_bounds
    else:
        volumes = OrderedDict({"_Vol_0": vol_0_bounds})

    part_vols = assign_elements_to_volumes(mesh=mesh, volumes=volumes,
                                           debug=debug)

    nparts_vol = compute_volume_partitions(part_vols, num_ranks,
                                           imbalance_tolerance, debug)

    all_elem_group_centroids = []
    for group in mesh.groups:
        elem_group_coords = mesh_verts[:, group.vertex_indices]
        elem_group_centroids = np.mean(elem_group_coords, axis=2)
        all_elem_group_centroids.append(elem_group_centroids.T)

    elem_centroids = np.concatenate(all_elem_group_centroids)
    global_nelements = len(elem_centroids)

    total_volume_els = 0
    for elements in part_vols.values():
        nelements_vol = len(elements)
        total_volume_els += nelements_vol

    assert total_volume_els == global_nelements
    # nparts_used = 0
    vparts_to_elements = {}
    for vol_id, elements in part_vols.items():
        nelements_vol = len(elements)
        bounds_vol = volumes[vol_id]
        npart_vol = nparts_vol[vol_id]
        if part_axis is None:
            part_axis = get_longest_axis(bounds_vol)
        target_part = nelements_vol / npart_vol
        vpax = {"X": 0, "Y": 1, "Z": 2}[part_axis]

        if debug:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now} Partitioning volume: {vol_id}")
            print(f"Volume bounding box: {bounds_vol}")
            print(f"Partitioning along axis: {vpax}")
            print(f"Number of elements in volume: {nelements_vol}")
            print(f"Number of partitions: {npart_vol}")
            print(f"Target partition size: {target_part}")

        # x_vol = elem_centroids[elements, vpax]
        elem_to_centroid = {e: elem_centroids[e, vpax] for e in elements}
        x_min = bounds_vol[2*vpax]
        x_max = bounds_vol[2*vpax+1]
        x_interval = x_max - x_min
        part_loc = np.linspace(x_min, x_max, npart_vol+1)

        part_interval = x_interval / npart_vol
        aver_part_nelem = target_part

        if debug:
            print(f"Initial part locs along volume axis: {part_loc=}")

        # Create geometrically even partitions
        # elem_to_vrank = ((x_vol-x_min) / part_interval).astype(int)
        # Initialize with invalid rank
        elem_to_vrank = np.full(global_nelements, -1, dtype=int)
        for e in elements:
            rank = int((elem_to_centroid[e] - x_min) / part_interval)
            elem_to_vrank[e] = rank

        # elem_to_vrank = {e: int((elem_to_centroid[e] - x_min) / part_interval)
        #                 for e in elements}
        # Check the initial partitioning for sanity
        invalid_elems = set()
        for e in elements:
            if elem_to_vrank[e] < 0 or elem_to_vrank[e] >= npart_vol:
                invalid_elems.add(e)
        if len(invalid_elems) > 0:
            raise PartitioningError(
                f"Initial Partitioning Error: Some elements in {vol_id} "
                "received invalid ranks!\n"
                f"Expected range: [0, {npart_vol-1}], "
                "but got out-of-bounds values.\n"
                f"Problematic elements: {invalid_elems}"
            )

        if debug:
            print(f"Validated ranks for {vol_id}: All elements assigned correctly.")

        # map partition id to list of elements in that partition
        vpart_to_elements = {r: set(np.where(elem_to_vrank == r)[0])
                             for r in range(npart_vol)}

        # make an array of the geometrically even partition sizes
        # avoids calling "len" over and over on the element sets
        nelem_vpart = [len(vpart_to_elements[r])
                      for r in range(npart_vol)]

        # Check if any elements were not assigned a partition
        assigned_elements = np.concatenate([list(vpart_to_elements[r])
                                            for r in range(npart_vol)])

        orphaned_elements = set(elements) - set(assigned_elements)

        if orphaned_elements:
            raise PartitioningError(
                f"Orphaned Elements Detected in {vol_id}: "
                f"{len(orphaned_elements)} elements "
                f"were not assigned to any partition!\n"
                f"Orphaned element indices: {list(orphaned_elements)[:10]} "
                "(showing first 10)"
            )

        if debug:
            print(f"Initial partitioning for {vol_id}: {nelem_vpart=}")

        # Automatic load-balancing
        if auto_balance:

            for r in range(npart_vol-1):
                num_elem_needed = aver_part_nelem - nelem_vpart[r]
                part_imbalance = np.abs(num_elem_needed) / float(aver_part_nelem)

                if debug:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{now}: Processing {vol_id} part({r})")
                    print(f"{part_loc[r]=}")
                    print(f"{num_elem_needed=}, {part_imbalance=}")
                    print(f"{nelem_vpart=}")

                niter = 0
                total_change = 0
                moved_elements = set()
                adv_part = r + 1
                while part_imbalance > imbalance_tolerance:
                    # This partition needs to keep changing in size until it meets
                    # specified imbalance tolerance, or gives up trying

                    # seek out the element reservoir
                    if num_elem_needed > 0:
                        while nelem_vpart[adv_part] == 0:
                            adv_part = adv_part + 1
                            if adv_part >= npart_vol:
                                raise PartitioningError("Ran out of elems to "
                                                        "partition!")

                    if debug:
                        print(f"-{nelem_vpart[r]=}, adv_part({adv_part}),"
                              f" {nelem_vpart[adv_part]=}")
                        print(f"-{part_loc[r+1]=},{part_loc[adv_part+1]=}")
                        print(f"-{num_elem_needed=},{part_imbalance=}")

                    if niter > 100:
                        raise PartitioningError("Detected too many iterations in"
                                                " partitioning.")

                    # The purpose of the next block is to populate "moved_elements"
                    # data structure. Then those elements will be moved between the
                    # current partition being processed and the "reservoir,"
                    # *and* to adjust the position of the "right" side of the current
                    # partition boundary.
                    moved_elements = set()
                    num_elements_added = 0

                    if num_elem_needed > 0:

                        # Partition is SMALLER than it should be, grab elements from
                        # the reservoir
                        if debug:
                            print(f"-Grabn elements from vol reservoir({adv_part})"
                                  f", {nelem_vpart[adv_part]=}")

                        portion_needed = (float(abs(num_elem_needed))
                                          / float(nelem_vpart[adv_part]))
                        portion_needed = min(portion_needed, 1.0)

                        if debug:
                            print(f"--Chomping {portion_needed*100}% of"
                                  f" reservoir({adv_part}) [by nelem].")

                        if portion_needed == 1.0:  # Chomp
                            new_loc = part_loc[adv_part+1]
                            moved_elements.update(vpart_to_elements[adv_part])
                        else:  # Bite
                            # This is the spatial size of the reservoir
                            reserv_interval = part_loc[adv_part+1] - part_loc[r+1]

                            # Find what portion of the reservoir to grab spatially
                            # This part is needed because the elements are not
                            # distributed uniformly in space.
                            fine_tuned = False
                            trial_portion_needed = portion_needed
                            while not fine_tuned:
                                pos_update = trial_portion_needed*reserv_interval
                                new_loc = part_loc[r+1] + pos_update

                                moved_elements = set()
                                num_elem_mv = 0
                                for e in vpart_to_elements[adv_part]:
                                    if elem_to_centroid[e] <= new_loc:
                                        moved_elements.add(e)
                                        num_elem_mv = num_elem_mv + 1
                                if num_elem_mv < num_elem_needed:
                                    fine_tuned = True
                                else:
                                    ovrsht = (num_elem_mv - num_elem_needed)
                                    rel_ovrsht = ovrsht/float(num_elem_needed)
                                    if rel_ovrsht > 0.8:
                                        # bisect the space grabbed and try again
                                        trial_portion_needed = \
                                            trial_portion_needed/2.0
                                    else:
                                        fine_tuned = True

                            portion_needed = trial_portion_needed
                            new_loc = part_loc[r+1] + pos_update
                            if debug:
                                print(f"--Tuned: {portion_needed=} [spatial]")
                                print(f"--Advancing part({r}) by +{pos_update}")

                        num_elements_added = len(moved_elements)
                        if debug:
                            print(f"--Adding {num_elements_added} to part({r}).")

                    else:

                        # Partition is LARGER than it should be
                        # Grab the spatial size of the current partition
                        # to estimate the portion we need to shave off
                        # assuming uniform element density
                        part_interval = part_loc[r+1] - part_loc[r]
                        num_to_move = -num_elem_needed
                        portion_needed = num_to_move/float(nelem_vpart[r])

                        if debug:
                            print(f"--Shaving off {portion_needed*100}% of"
                                  f" partition({r}) [by nelem].")

                        # Tune the shaved portion to account for
                        # non-uniform element density
                        fine_tuned = False
                        while not fine_tuned:
                            pos_update = portion_needed*part_interval
                            new_pos = part_loc[r+1] - pos_update
                            moved_elements = set()
                            num_elem_mv = 0
                            for e in vpart_to_elements[r]:
                                if elem_to_centroid[e] > new_pos:
                                    moved_elements.add(e)
                                    num_elem_mv = num_elem_mv + 1
                            if num_elem_mv < num_to_move:
                                fine_tuned = True
                            else:
                                ovrsht = (num_elem_mv - num_to_move)
                                rel_ovrsht = ovrsht/float(num_to_move)
                                if rel_ovrsht > 0.8:
                                    # bisect and try again
                                    portion_needed = portion_needed/2.0
                                else:
                                    fine_tuned = True

                        # new "right" wall location of shranken part
                        # and negative num_elements_added for removal
                        new_loc = new_pos
                        num_elements_added = -len(moved_elements)
                        if debug:
                            print(f"--Reducing part size by {portion_needed*100}%"
                                  " [by nelem].")
                            print(f"--Remv {-num_elements_added} from part({r}).")

                    # Now "moved_elements", "num_elements_added", and "new_loc"
                    # are computed.  Update the partition, and reservoir.
                    if debug:
                        print(f"--Number of elements to ADD: {num_elements_added}.")

                    if num_elements_added > 0:
                        vpart_to_elements[r].update(moved_elements)
                        vpart_to_elements[adv_part].difference_update(
                            moved_elements)
                        for e in moved_elements:
                            elem_to_vrank[e] = r
                    else:
                        vpart_to_elements[r].difference_update(moved_elements)
                        vpart_to_elements[adv_part].update(moved_elements)
                        for e in moved_elements:
                            elem_to_vrank[e] = adv_part

                    total_change = total_change + num_elements_added
                    part_loc[r+1] = new_loc
                    if debug:
                        print(f"--Before: {nelem_vpart=}")
                    nelem_vpart[r] = nelem_vpart[r] + num_elements_added
                    nelem_vpart[adv_part] = \
                        nelem_vpart[adv_part] - num_elements_added
                    if debug:
                        print(f"--After: {nelem_vpart=}")

                    # Compute new nelem_needed and part_imbalance
                    num_elem_needed = num_elem_needed - num_elements_added
                    part_imbalance = \
                        np.abs(num_elem_needed) / float(aver_part_nelem)
                    niter = niter + 1

                # Summarize the total change and state of the partition
                # and reservoir
                if debug:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{now} ---------")
                    print(f"-Vol Part({r}): {total_change=}")
                    print(f"-Vol Part({r}): {nelem_vpart[r]=}, {part_imbalance=}")
                    print(f"-Vol Part({adv_part})[Resv]: {nelem_vpart[adv_part]=}")
                    print(f"-Vol Part({r}) Box: ({part_loc[r]},{part_loc[r+1]})")
                    print(f"-Vol Part({adv_part})[Resv] Box: ({part_loc[r+1]},"
                          f"{part_loc[adv_part]})")

                # loop over volume ranks scope
            # autobalance
            # Validation: Ensure no elements are lost after processing this volume
            total_partitioned_elements = sum(len(vpart_to_elements[r])
                                             for r in range(npart_vol))

            if total_partitioned_elements != nelements_vol:
                raise PartitioningError(f"Auto-Balance Error: Element count mismatch"
                                        f" after partitioning {vol_id}. "
                                        f"Expected {nelements_vol}, "
                                        f"got {total_partitioned_elements}")

        vparts_to_elements[vol_id] = vpart_to_elements
        # Loop over volumes

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} Making global element-to-rank array.")
    # Initialize the global element-to-rank array
    elem_to_rank = np.full(global_nelements, -1, dtype=int)  # -1 means unassigned
    nelem_part = np.full(num_ranks, 0, dtype=int)
    part_to_elements = {}
    current_global_rank = 0
    # Ensure _Vol_0 is processed first
    sorted_vol_ids = ["_Vol_0"] + [v for v in vparts_to_elements if v != "_Vol_0"]
    for vol_id in sorted_vol_ids:
        vpart_map = vparts_to_elements[vol_id]
        if debug:
            print(f"Adding Volume({vol_id}) to global partitioning.")
        for vrank, elements in vpart_map.items():
            global_rank = current_global_rank + vrank
            elem_to_rank[np.array(list(elements))] = global_rank
            part_to_elements[global_rank] = elements
            nelem_part[global_rank] = len(elements)
            if debug:
                print(f"{vol_id}({vrank}) with {nelem_part[global_rank]} "
                      f"elements to global rank ({global_rank}).")
        current_global_rank += len(vpart_map)

    # Validate the partitioning before returning
    total_partitioned_elements = sum([len(part_to_elements[r])
                                      for r in range(num_ranks)])
    total_nelem_part = sum([nelem_part[r] for r in range(num_ranks)])

    if debug:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{now}: Validating mesh parts.")

    if total_partitioned_elements != total_nelem_part:
        raise PartitioningError("Validator: parted element counts dont match")
    if len(elem_to_rank) != global_nelements:
        raise PartitioningError("Validator: elem-to-rank wrong size.")
    if np.any(nelem_part) <= 0:
        raise PartitioningError("Validator: empty partitions.")
    part_counts = np.zeros(global_nelements)
    for part_elements in part_to_elements.values():
        for element in part_elements:
            part_counts[element] = part_counts[element] + 1
    if np.any(part_counts < 1):
        raise PartitioningError("Validator: orphaned elements")
    if np.any(part_counts > 1):
        raise PartitioningError("Validator: degenerate elements")
    for e in range(global_nelements):
        part = elem_to_rank[e]
        if e not in part_to_elements[part]:
            raise PartitioningError("Validator: part/element/part map mismatch.")

    if total_partitioned_elements != global_nelements:
        raise PartitioningError("Validator: global element counts dont match."
                                f"{total_partitioned_elements=},{global_nelements=}")

    if debug:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now}: Final partitioning: {nelem_part}")

    return elem_to_rank


def generate_and_distribute_mesh(comm, generate_mesh, **kwargs):
    """Generate a mesh and distribute it among all ranks in *comm*.

    Generate the mesh with the user-supplied mesh generation function
    *generate_mesh*, partition the mesh, and distribute it to every
    rank in the provided MPI communicator *comm*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the mesh
    generate_mesh:
        Callable of zero arguments returning a :class:`meshmode.mesh.Mesh`.
        Will only be called on one (undetermined) rank.

    Returns
    -------
    local_mesh : :class:`meshmode.mesh.Mesh`
        The local partition of the the mesh returned by *generate_mesh*.
    global_nelements : :class:`int`
        The number of elements in the serial mesh
    """
    from warnings import warn
    warn(
        "generate_and_distribute_mesh is deprecated and will go away Q4 2022. "
        "Use distribute_mesh instead.", DeprecationWarning, stacklevel=2)
    return distribute_mesh(comm, generate_mesh, **kwargs)


def invert_decomp(decomp_map):
    """Return a list of global elements for each partition."""
    from collections import defaultdict
    global_elements_per_part = defaultdict(list)

    for elemid, part in enumerate(decomp_map):
        global_elements_per_part[part].append(elemid)

    return global_elements_per_part


def _partition_single_volume_mesh(
        mesh, num_ranks, rank_per_element, *, return_ranks=None):
    rank_to_elements = {
        rank: np.where(rank_per_element == rank)[0]
        for rank in range(num_ranks)}

    from meshmode.mesh.processing import partition_mesh
    return partition_mesh(
        mesh, rank_to_elements, return_parts=return_ranks)


def _get_multi_volume_partitions(mesh, num_ranks, rank_per_element,
                                 tag_to_elements, volume_to_tags):

    volumes = list(volume_to_tags.keys())

    tag_to_volume = {
        tag: vol
        for vol, tags in volume_to_tags.items()
        for tag in tags}

    volume_index_per_element = np.full(mesh.nelements, -1, dtype=int)
    for tag, elements in tag_to_elements.items():
        volume_index_per_element[elements] = volumes.index(
            tag_to_volume[tag])

    if np.any(volume_index_per_element < 0):
        raise ValueError("Missing volume specification for some elements.")

    part_id_to_elements = {
        PartID(volumes[vol_idx], rank):
            np.where(
                (volume_index_per_element == vol_idx)
                & (rank_per_element == rank))[0]
        for vol_idx in range(len(volumes))
        for rank in range(num_ranks)}

    return part_id_to_elements


def _partition_multi_volume_mesh(
        mesh, num_ranks, rank_per_element,
        part_id_to_elements, tag_to_elements, volume_to_tags, *,
        return_ranks=None):
    if return_ranks is None:
        return_ranks = list(range(num_ranks))
    volumes = list(volume_to_tags.keys())

    # TODO: Add a public meshmode function to accomplish this? So we're
    # not depending on meshmode internals
    # PartID is (rank, vol) pair
    # part_index is range(len(PartID))
    # part_id_to_part_index is {PartID: part_index}
    # global_elem_to_part_elem is ary[global_elem_index] = \
    #    [part_index, local_element_index]
    part_id_to_part_index = {
        part_id: part_index
        for part_index, part_id in enumerate(part_id_to_elements.keys())}
    from meshmode.mesh.processing import _compute_global_elem_to_part_elem
    global_elem_to_part_elem = _compute_global_elem_to_part_elem(
        mesh.nelements, part_id_to_elements, part_id_to_part_index,
        mesh.element_id_dtype)

    # tag_to_global_to_part = \
    #    {tag: ary[global_elem_index][part_index, local_element_index]}
    tag_to_global_to_part = {
        tag: global_elem_to_part_elem[elements, :]
        for tag, elements in tag_to_elements.items()}

    part_id_to_tag_to_elements = {}
    for part_id in part_id_to_elements.keys():
        part_idx = part_id_to_part_index[part_id]
        part_tag_to_elements = {}
        for tag, global_to_part in tag_to_global_to_part.items():
            part_tag_to_elements[tag] = global_to_part[
                global_to_part[:, 0] == part_idx, 1]
        part_id_to_tag_to_elements[part_id] = part_tag_to_elements

    return_parts = {
        PartID(vol, rank)
        for vol in volumes
        for rank in return_ranks}

    from meshmode.mesh.processing import partition_mesh
    part_id_to_mesh = partition_mesh(
        mesh, part_id_to_elements, return_parts=return_parts)

    return {
        rank: {
            vol: (
                part_id_to_mesh[PartID(vol, rank)],
                part_id_to_tag_to_elements[PartID(vol, rank)])
            for vol in volumes}
        for rank in return_ranks}


@contextmanager
def _manage_mpi_comm(comm):
    try:
        yield comm
    finally:
        comm.Free()


def distribute_mesh(comm, get_mesh_data, partition_generator_func=None, logmgr=None,
                    num_per_batch=None):
    r"""Distribute a mesh among all ranks in *comm*.

    Retrieve the global mesh data with the user-supplied function *get_mesh_data*,
    partition the mesh, and distribute it to every rank in the provided MPI
    communicator *comm*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the mesh
    get_mesh_data:
        Callable of zero arguments returning *mesh* or
        *(mesh, tag_to_elements, volume_to_tags)*, where *mesh* is a
        :class:`meshmode.mesh.Mesh`, *tag_to_elements* is a
        :class:`dict` mapping mesh volume tags to :class:`numpy.ndarray`\ s of
        element numbers, and *volume_to_tags* is a :class:`dict` that maps volumes
        in the resulting distributed mesh to volume tags in *tag_to_elements*.
    partition_generator_func:
        Optional callable that takes *mesh*, *tag_to_elements*, and *comm*'s size,
        and returns a :class:`numpy.ndarray` indicating to which rank each element
        belongs.

    Returns
    -------
    local_mesh_data: :class:`meshmode.mesh.Mesh` or :class:`dict`
        If the result of calling *get_mesh_data* specifies a single volume,
        *local_mesh_data* is the local mesh.  If it specifies multiple volumes,
        *local_mesh_data* will be a :class:`dict` mapping volume tags to
        tuples of the form *(local_mesh, local_tag_to_elements)*.
    global_nelements: :class:`int`
        The number of elements in the global mesh
    """
    from mpi4py import MPI
    from mpi4py.util import pkl5
    from socket import gethostname

    num_ranks = comm.Get_size()
    my_global_rank = comm.Get_rank()
    hostname = gethostname()

    t_mesh_dist = IntervalTimer("t_mesh_dist", "Time spent distributing mesh data.")
    t_mesh_data = IntervalTimer("t_mesh_data", "Time spent getting mesh data.")
    t_mesh_part = IntervalTimer("t_mesh_part", "Time spent partitioning the mesh.")
    t_mesh_split = IntervalTimer("t_mesh_split", "Time spent splitting mesh parts.")

    if partition_generator_func is None:
        def partition_generator_func(mesh, tag_to_elements, num_ranks):
            from meshmode.distributed import get_partition_by_pymetis
            return get_partition_by_pymetis(mesh, num_ranks)

    with _manage_mpi_comm(
            pkl5.Intracomm(comm.Split_type(MPI.COMM_TYPE_SHARED,
                                           comm.Get_rank(), MPI.INFO_NULL))
    ) as node_comm:

        node_ranks = node_comm.gather(comm.Get_rank(), root=0)
        my_node_rank = node_comm.Get_rank()
        reader_color = 0 if my_node_rank == 0 else 1
        reader_comm = comm.Split(reader_color, my_global_rank)
        my_reader_rank = reader_comm.Get_rank()
        num_node_ranks = node_comm.Get_size()

        if my_node_rank == 0:
            num_reading_ranks = reader_comm.Get_size()
            num_per_batch = num_per_batch or num_reading_ranks
            num_reading_batches = max(int(num_reading_ranks / num_per_batch), 1)
            read_batch = int(my_reader_rank / num_per_batch)

            print(f"Read(rank, batch): Dist({my_reader_rank}, "
                  f"{read_batch}) on {hostname}.")

            global_data = None
            if logmgr:
                logmgr.add_quantity(t_mesh_data)
                with t_mesh_data.get_sub_timer():
                    for reading_batch in range(num_reading_batches):
                        if read_batch == reading_batch:
                            print(f"Reading mesh on {hostname}.")
                            global_data = get_mesh_data()
                            reader_comm.Barrier()
                        else:
                            reader_comm.Barrier()
            else:
                for reading_batch in range(num_reading_batches):
                    if read_batch == reading_batch:
                        print(f"Reading mesh on {hostname}.")
                        global_data = get_mesh_data()
                        reader_comm.Barrier()
                    else:
                        reader_comm.Barrier()

            reader_comm.Barrier()
            if my_reader_rank == 0:
                print("Mesh reading done on all nodes.")

            from meshmode.mesh import Mesh
            if isinstance(global_data, Mesh):
                mesh = global_data
                tag_to_elements = None
                volume_to_tags = None
            elif isinstance(global_data, tuple) and len(global_data) == 3:
                mesh, tag_to_elements, volume_to_tags = global_data
            else:
                raise TypeError("Unexpected result from get_mesh_data")

            reader_comm.Barrier()
            if my_reader_rank == 0:
                print("Making partition table on all nodes.")

            if logmgr:
                logmgr.add_quantity(t_mesh_part)
                with t_mesh_part.get_sub_timer():
                    rank_per_element = \
                        partition_generator_func(mesh, tag_to_elements,
                                                 num_ranks)
            else:
                rank_per_element = \
                    partition_generator_func(mesh, tag_to_elements,
                                             num_ranks)

            def get_rank_to_mesh_data_dict():
                if tag_to_elements is None:
                    rank_to_mesh_data = _partition_single_volume_mesh(
                        mesh, num_ranks, rank_per_element,
                        return_ranks=node_ranks)
                else:
                    part_id_to_elements = _get_multi_volume_partitions(
                        mesh, num_ranks, rank_per_element, tag_to_elements,
                        volume_to_tags)
                    rank_to_mesh_data = _partition_multi_volume_mesh(
                        mesh, num_ranks, rank_per_element, part_id_to_elements,
                        tag_to_elements, volume_to_tags, return_ranks=node_ranks)

                rank_to_node_rank = {
                    rank: node_rank
                    for node_rank, rank in enumerate(node_ranks)}

                node_rank_to_mesh_data_dict = {
                    rank_to_node_rank[rank]: mesh_data
                    for rank, mesh_data in rank_to_mesh_data.items()}

                return node_rank_to_mesh_data_dict

            reader_comm.Barrier()
            if my_reader_rank == 0:
                print("Partitioning mesh on all nodes.")

            if logmgr:
                logmgr.add_quantity(t_mesh_split)
                with t_mesh_split.get_sub_timer():
                    node_rank_to_mesh_data_dict = get_rank_to_mesh_data_dict()
            else:
                node_rank_to_mesh_data_dict = get_rank_to_mesh_data_dict()

            node_rank_to_mesh_data = [
                node_rank_to_mesh_data_dict[rank]
                for rank in range(num_node_ranks)]

            reader_comm.Barrier()
            if my_reader_rank == 0:
                print("Partitioning done, distributing to node-local ranks.")

            global_nelements = node_comm.bcast(mesh.nelements, root=0)

            if logmgr:
                logmgr.add_quantity(t_mesh_dist)
                with t_mesh_dist.get_sub_timer():
                    local_mesh_data = \
                        node_comm.scatter(node_rank_to_mesh_data, root=0)
            else:
                local_mesh_data = \
                    node_comm.scatter(node_rank_to_mesh_data, root=0)

        else:  # my_node_rank > 0, get mesh part from MPI
            global_nelements = node_comm.bcast(None, root=0)

            if logmgr:
                logmgr.add_quantity(t_mesh_dist)
                with t_mesh_dist.get_sub_timer():
                    local_mesh_data = node_comm.scatter(None, root=0)
            else:
                local_mesh_data = node_comm.scatter(None, root=0)

    return local_mesh_data, global_nelements


def distribute_mesh_pkl(comm, get_mesh_data, filename="mesh",
                        num_target_ranks=0, num_reader_ranks=0,
                        partition_generator_func=None, logmgr=None):
    r"""Distribute a mesh among all ranks in *comm*.

    Retrieve the global mesh data with the user-supplied function *get_mesh_data*,
    partition the mesh, and distribute it to every rank in the provided MPI
    communicator *comm*.

    .. note::
        This is a collective routine and must be called by all MPI ranks.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the mesh
    get_mesh_data:
        Callable of zero arguments returning *mesh* or
        *(mesh, tag_to_elements, volume_to_tags)*, where *mesh* is a
        :class:`meshmode.mesh.Mesh`, *tag_to_elements* is a
        :class:`dict` mapping mesh volume tags to :class:`numpy.ndarray`\ s of
        element numbers, and *volume_to_tags* is a :class:`dict` that maps volumes
        in the resulting distributed mesh to volume tags in *tag_to_elements*.
    partition_generator_func:
        Optional callable that takes *mesh*, *tag_to_elements*, and *comm*'s size,
        and returns a :class:`numpy.ndarray` indicating to which rank each element
        belongs.

    Returns
    -------
    local_mesh_data: :class:`meshmode.mesh.Mesh` or :class:`dict`
        If the result of calling *get_mesh_data* specifies a single volume,
        *local_mesh_data* is the local mesh.  If it specifies multiple volumes,
        *local_mesh_data* will be a :class:`dict` mapping volume tags to
        tuples of the form *(local_mesh, local_tag_to_elements)*.
    global_nelements: :class:`int`
        The number of elements in the global mesh
    """
    from mpi4py.util import pkl5
    from datetime import datetime
    comm_wrapper = pkl5.Intracomm(comm)

    num_ranks = comm_wrapper.Get_size()
    my_rank = comm_wrapper.Get_rank()

    if num_target_ranks <= 0:
        num_target_ranks = num_ranks
    if num_reader_ranks <= 0:
        num_reader_ranks = num_ranks

    reader_color = 1 if my_rank < num_reader_ranks else 0
    reader_comm = comm_wrapper.Split(reader_color, my_rank)
    reader_comm_wrapper = pkl5.Intracomm(reader_comm)
    reader_rank = reader_comm_wrapper.Get_rank()
    num_ranks_per_reader = int(num_target_ranks / num_reader_ranks)
    num_leftover = num_target_ranks - (num_ranks_per_reader * num_reader_ranks)
    num_ranks_this_reader = num_ranks_per_reader + (1 if reader_rank
                                                    < num_leftover else 0)

    t_mesh_dist = IntervalTimer("t_mesh_dist", "Time spent distributing mesh data.")
    t_mesh_data = IntervalTimer("t_mesh_data", "Time spent getting mesh data.")
    t_mesh_part = IntervalTimer("t_mesh_part", "Time spent partitioning the mesh.")
    t_mesh_split = IntervalTimer("t_mesh_split", "Time spent splitting mesh parts.")

    if reader_color and num_ranks_this_reader > 0:
        my_starting_rank = num_ranks_per_reader * reader_rank
        my_starting_rank = my_starting_rank + (reader_rank if reader_rank
                                               < num_leftover else num_leftover)
        my_ending_rank = my_starting_rank + num_ranks_this_reader - 1
        ranks_to_write = list(range(my_starting_rank, my_ending_rank+1))

        if reader_rank == 0:
            print("Reading(world_rank,reader_rank): "
                  "Writing[starting_rank,ending_rank]")
            print("----------------------------------")
        reader_comm.Barrier()
        print(f"R({my_rank},{reader_rank}): "
              f"W[{my_starting_rank},{my_ending_rank}]")

        if partition_generator_func is None:
            def partition_generator_func(mesh, tag_to_elements, num_target_ranks):
                from meshmode.distributed import get_partition_by_pymetis
                return get_partition_by_pymetis(mesh, num_target_ranks)

        if reader_rank == 0:
            if logmgr:
                logmgr.add_quantity(t_mesh_data)
                with t_mesh_data.get_sub_timer():
                    global_data = get_mesh_data()
            else:
                global_data = get_mesh_data()
            print(f"{datetime.now()}: Done reading source mesh from file. "
                  "Broadcasting...")
            global_data = reader_comm_wrapper.bcast(global_data, root=0)
        else:
            global_data = reader_comm_wrapper.bcast(None, root=0)

        reader_comm.Barrier()
        if reader_rank == 0:
            print(f"{datetime.now()}: Done distrbuting source mesh data."
                  " Partitioning...")

        from meshmode.mesh import Mesh
        if isinstance(global_data, Mesh):
            mesh = global_data
            tag_to_elements = None
            volume_to_tags = None
        elif isinstance(global_data, tuple) and len(global_data) == 3:
            mesh, tag_to_elements, volume_to_tags = global_data
        else:
            raise TypeError("Unexpected result from get_mesh_data")

        if logmgr:
            logmgr.add_quantity(t_mesh_part)
            with t_mesh_part.get_sub_timer():
                rank_per_element = partition_generator_func(mesh, tag_to_elements,
                                                            num_target_ranks)
        else:
            rank_per_element = partition_generator_func(mesh, tag_to_elements,
                                                        num_target_ranks)

        # Save this little puppy for later (m-to-n restart support)
        if reader_rank == 0:
            part_table_fname = filename + f"_decomp_np{num_target_ranks}.pkl"
            if os.path.exists(part_table_fname):
                os.remove(part_table_fname)
            with open(part_table_fname, "wb") as pkl_file:
                pickle.dump(rank_per_element, pkl_file)
            rank_to_elems = invert_decomp(rank_per_element)
            part_table_fname = filename + f"_idecomp_np{num_target_ranks}.pkl"
            with open(part_table_fname, "wb") as pkl_file:
                pickle.dump(rank_to_elems, pkl_file)

        reader_comm.Barrier()
        if reader_rank == 0:
            print(f"{datetime.now()}: Done with global partitioning. Splitting...")

        if tag_to_elements is None:
            part_id_to_elements = None
        else:
            part_id_to_elements = _get_multi_volume_partitions(
                mesh, num_target_ranks, rank_per_element, tag_to_elements,
                volume_to_tags)

            # Save this little puppy for later (m-to-n restart support)
            if reader_rank == 0:
                mv_part_table_fname = \
                    filename + f"_multivol_idecomp_np{num_target_ranks}.pkl"
                if os.path.exists(mv_part_table_fname):
                    os.remove(mv_part_table_fname)
                with open(mv_part_table_fname, "wb") as pkl_file:
                    pickle.dump(part_id_to_elements, pkl_file)

        reader_comm.Barrier()
        if reader_rank == 0:
            print(f"{datetime.now()}: - Got PartID-to-elements. "
                  "Making mesh data structures...")

        def get_rank_to_mesh_data():
            if tag_to_elements is None:
                rank_to_mesh_data = _partition_single_volume_mesh(
                    mesh, num_target_ranks, rank_per_element,
                    return_ranks=ranks_to_write)
            else:
                rank_to_mesh_data = _partition_multi_volume_mesh(
                    mesh, num_target_ranks, rank_per_element, part_id_to_elements,
                    tag_to_elements, volume_to_tags, return_ranks=ranks_to_write)
            return rank_to_mesh_data

        reader_comm.Barrier()
        if logmgr:
            logmgr.add_quantity(t_mesh_split)
            with t_mesh_split.get_sub_timer():
                rank_to_mesh_data = get_rank_to_mesh_data()
        else:
            rank_to_mesh_data = get_rank_to_mesh_data()

        reader_comm.Barrier()
        if reader_rank == 0:
            print(f"{datetime.now()}: Done splitting mesh. Writing...")

        if logmgr:
            logmgr.add_quantity(t_mesh_dist)
            with t_mesh_dist.get_sub_timer():
                for part_rank, part_mesh in rank_to_mesh_data.items():
                    pkl_filename = (filename
                                    + f"_np{num_target_ranks}_rank{part_rank}.pkl")
                    mesh_data_to_pickle = (mesh.nelements, part_mesh)
                    if os.path.exists(pkl_filename):
                        os.remove(pkl_filename)
                    with open(pkl_filename, "wb") as pkl_file:
                        pickle.dump(mesh_data_to_pickle, pkl_file)
        else:
            for part_rank, part_mesh in rank_to_mesh_data.items():
                pkl_filename = filename + f"_rank{part_rank}.pkl"
                mesh_data_to_pickle = (mesh.nelements, part_mesh)
                if os.path.exists(pkl_filename):
                    os.remove(pkl_filename)
                with open(pkl_filename, "wb") as pkl_file:
                    pickle.dump(mesh_data_to_pickle, pkl_file)

        reader_comm.Barrier()
        if reader_rank == 0:
            print(f"{datetime.now()}: Done writing partitioned mesh.")


def extract_volumes(mesh, tag_to_elements, selected_tags, boundary_tag):
    r"""
    Create a mesh containing a subset of another mesh's volumes.

    Parameters
    ----------
    mesh: :class:`meshmode.mesh.Mesh`
        The original mesh.
    tag_to_elements:
        A :class:`dict` mapping mesh volume tags to :class:`numpy.ndarray`\ s
        of element numbers in *mesh*.
    selected_tags:
        A sequence of tags in *tag_to_elements* representing the subset of volumes
        to be included.
    boundary_tag:
        Tag to assign to the boundary that was previously the interface between
        included/excluded volumes.

    Returns
    -------
    in_mesh: :class:`meshmode.mesh.Mesh`
        The resulting mesh.
    tag_to_in_elements:
        A :class:`dict` mapping the tags from *selected_tags* to
        :class:`numpy.ndarray`\ s of element numbers in *in_mesh*.
    """
    is_in_element = np.full(mesh.nelements, False)
    for tag, elements in tag_to_elements.items():
        if tag in selected_tags:
            is_in_element[elements] = True

    from meshmode.mesh.processing import partition_mesh
    in_mesh = partition_mesh(mesh, {
        "_in": np.where(is_in_element)[0],
        "_out": np.where(~is_in_element)[0]})["_in"]

    # partition_mesh creates a partition boundary for "_out"; replace with a
    # normal boundary
    new_facial_adjacency_groups = []
    from meshmode.mesh import BoundaryAdjacencyGroup, InterPartAdjacencyGroup
    for grp_list in in_mesh.facial_adjacency_groups:
        new_grp_list = []
        for fagrp in grp_list:
            if (
                    isinstance(fagrp, InterPartAdjacencyGroup)
                    and fagrp.part_id == "_out"):
                new_fagrp = BoundaryAdjacencyGroup(
                    igroup=fagrp.igroup,
                    boundary_tag=boundary_tag,
                    elements=fagrp.elements,
                    element_faces=fagrp.element_faces)
            else:
                new_fagrp = fagrp
            new_grp_list.append(new_fagrp)
        new_facial_adjacency_groups.append(new_grp_list)
    in_mesh = in_mesh.copy(facial_adjacency_groups=new_facial_adjacency_groups)

    element_to_in_element = np.where(
        is_in_element,
        np.cumsum(is_in_element) - 1,
        np.full(mesh.nelements, -1))

    tag_to_in_elements = {
        tag: element_to_in_element[tag_to_elements[tag]]
        for tag in selected_tags}

    return in_mesh, tag_to_in_elements


def copy_mapped_dof_array_data(trg_dof_array, src_dof_array, index_map):
    """Copy data between DOFArrays from disparate meshes."""
    # Assume ONE group (tetrahedra ONLY)
    actx = trg_dof_array.array_context
    trg_dof_array_np = actx.to_numpy(trg_dof_array)
    src_dof_array_np = actx.to_numpy(src_dof_array)

    trg_array = trg_dof_array_np[0]
    src_array = src_dof_array_np[0]
    src_nel, src_nnodes = src_array.shape
    trg_nel, trg_nnodes = trg_array.shape

    if trg_nel == 0 or src_nel == 0:
        return trg_dof_array

    if src_nnodes != trg_nnodes:
        raise ValueError("DOFArray mapped copy must be of same order.")

    # Actual data copy
    for trg_el, src_el in index_map.items():
        trg_array[trg_el] = src_array[src_el]

    return actx.from_numpy(trg_dof_array_np)


def interdecomposition_imapping(target_idecomp, source_idecomp):
    """
    Return a mapping of which partitions to source for the target decomp.

    Expects input format: {rank: [elements]}

    Parameters
    ----------
    target_idecomp: dict
        Target decomposition in the format {rank: [elements]}
    source_idecomp: dict
        Source decomposition in the format {rank: [elements]}

    Returns
    -------
    dict
        Dictionary like {trg_rank: [src_ranks]}
    """
    from collections import defaultdict

    # Convert {rank: [elements]} format into {element: rank} for faster look-up
    source_elem_to_rank = {}
    for rank, elements in source_idecomp.items():
        for elem in elements:
            source_elem_to_rank[elem] = rank

    interdecomp_map = defaultdict(set)

    for trg_rank, trg_elements in target_idecomp.items():
        for elem in trg_elements:
            src_rank = source_elem_to_rank.get(elem)
            if src_rank is not None:
                interdecomp_map[trg_rank].add(src_rank)

    # Convert sets to lists for the final output
    for rank in interdecomp_map:
        interdecomp_map[rank] = list(interdecomp_map[rank])

    return interdecomp_map


def interdecomposition_mapping(target_decomp, source_decomp):
    """Return a mapping of which partitions to source for the target decomp."""
    from collections import defaultdict

    interdecomp_map = defaultdict(set)

    for elemid, part in enumerate(target_decomp):
        interdecomp_map[part].add(source_decomp[elemid])

    for part in interdecomp_map:
        interdecomp_map[part] = list(interdecomp_map[part])

    return interdecomp_map


def summarize_decomposition(decomp_map, multivol_decomp_map):
    """Summarize decomp."""
    # Inputs are the decomp_map {rank: [elements]}
    # and multivol_decomp_map {PartID: array([elements])}
    nranks = len(decomp_map)

    # Initialize counters and containers
    total_num_elem = 0
    volume_element_counts = defaultdict(int)
    rank_element_counts = {}
    rank_volume_element_counts = defaultdict(lambda: defaultdict(int))
    unique_volumes = set()

    # Process data from decomp_map
    for rank, elements in decomp_map.items():
        rank_element_counts[rank] = len(elements)
        total_num_elem += len(elements)

    # Process data from multivol_decomp_map
    for partid, elements in multivol_decomp_map.items():
        vol, rank = partid.volume_tag, partid.rank
        unique_volumes.add(vol)
        nvol_els = len(elements)
        volume_element_counts[vol] += nvol_els
        rank_volume_element_counts[rank][vol] = nvol_els

    # Print summary
    print(f"Number of elements: {total_num_elem}")
    print(f"Volumes({len(unique_volumes)}): {unique_volumes}")
    for vol, count in volume_element_counts.items():
        print(f" - Volume({vol}): {count} elements.")
    print(f"Number of ranks: {nranks}")
    for rank in range(nranks):
        print(f" - Rank({rank}): {rank_element_counts[rank]} elements.")
        for vol, size in rank_volume_element_counts[rank].items():
            print(f" -- Vol({vol}): {size}")


# Need a function to determine which of my local elements overlap
# with a disparate decomp part. Optionally restrict attention to
# selected parts.
# For each (or selected) new part
#   Make a dict/map to hold element mapping (key: old_part, val: dict[elid]->[elid])
#   For each local element in the new part
#     1. find the global el id
#     2. grab the old part for that global el
#     3. find the old part-local id (i.e. index) for that global el
#     4. Append local and remote el id lists for old-part-specific dict entry
def interdecomposition_overlap(target_decomp_map, source_decomp_map,
                               return_parts=None):
    """Map element indices for overlapping, disparate decompositions.

    For each (or optionally selected) target parts, this routine
    returns a dictionary keyed by overlapping remote partitions from
    the source decomposition, the value of which is a map from the
    target-part-specific local indexes to the source-part-specific local
    index of for the corresponding element.

    Example dictionary structure:

    .. code-block:: python

       {
         targ_part_1 : {
           src_part_1 : { local_el_index : remote_el_index, ... },
           src_part_2 : { local_el_index : remote_el_index, ... },
           ...
         },
         targ_part_2 : { ... },
         ...
       }

    This data structure is useful for mapping the solution data from
    the old decomp pkl restart files to the new decomp solution arrays.
    """
    src_part_to_els = invert_decomp(source_decomp_map)
    trg_part_to_els = invert_decomp(target_decomp_map)
    ipmap = interdecomposition_mapping(target_decomp_map, source_decomp_map)

    ntrg_parts = len(trg_part_to_els)
    if return_parts is None:
        return_parts = list(range(ntrg_parts))
    overlap_maps = {}
    for trg_part in return_parts:
        overlap_maps[trg_part] = {}
        for src_part in ipmap[trg_part]:
            overlap_maps[trg_part][src_part] = {}
    for trg_part in return_parts:
        trg_els = trg_part_to_els[trg_part]
        for glb_el in trg_els:
            src_part = source_decomp_map[glb_el]
            src_el_index = src_part_to_els[src_part].index(glb_el)
            loc_el_index = trg_els.index(glb_el)
            overlap_maps[trg_part][src_part][loc_el_index] = src_el_index
    return overlap_maps


# Interdecomposition overlap utility for multi-volume datasets
def multivolume_interdecomposition_overlap(src_decomp_map, trg_decomp_map,
                              src_multivol_decomp_map, trg_multivol_decomp_map,
                              return_ranks=None):
    """
    Construct local-to-local index mapping for overlapping decompositions.

    Parameters
    ----------
    src_decomp_map: dict
        Source decomposition map {rank: [elements]}.

    trg_decomp_map: dict
        Target decomposition map {rank: [elements]}.

    src_multivol_decomp_map: dict
        Source multivolume decomposition map {PartID: np.array(elements)}.

    trg_multivol_decomp_map: dict
        Target multivolume decomposition map {PartID: np.array(elements)}.

    Returns
    -------
    dict
        A dictionary with the following structure

        .. code-block:: python

            {
                trg_partid: {
                    src_partid: {
                        trg_local_el_index: src_local_el_index
                    }
                }
            }
    """
    # If no specific ranks are provided, consider all ranks in the target decomp
    if return_ranks is None:
        return_ranks = list(trg_decomp_map.keys())

    mapping = {}

    # First, identify overlapping ranks using the regular decomp maps
    overlapping_ranks = interdecomposition_imapping(trg_decomp_map, src_decomp_map)
    # print(f"{overlapping_ranks=}")

    # Now, for each overlapping rank, determine the overlapping elements using the
    # multivol decomp maps
    for trg_partid, trg_elems in trg_multivol_decomp_map.items():
        trg_nelem_part = len(trg_elems)
        # print(f"dbg Finding overlaps for: {trg_partid=}, {trg_nelem_part}")
        if trg_nelem_part == 0:
            # print("dbg - Skipping empty target part.")
            continue
        trg_rank = trg_partid.rank
        if trg_rank not in return_ranks:
            # print("dbg - Skipping unrelated trg rank.")
            continue
        target_elements_set = set(trg_elems)
        # print(f"dbg {target_elements_set=}")
        # print(f"dbg type of trg_elems={type(trg_elems)}, "
        #       f"trg_elems content={trg_elems}")
        noverlap = 0
        for src_rank in overlapping_ranks[trg_rank]:
            # print(f"dbg - Searching for src partids with {src_rank=}")
            for src_partid, src_elems in src_multivol_decomp_map.items():
                # print(f"dbg -- Considering {src_partid=}, {len(src_elems)=}")
                if src_partid.rank != src_rank:
                    # print("dbg --- Skipping unrelated src rank.")
                    continue
                if src_partid.volume_tag != trg_partid.volume_tag:
                    # print("dbg --- Skipping unrelated src volume.")
                    continue
                # print("dbg --- Determining overlap")
                # Determine element overlaps, set is used for performance
                source_elements_set = set(src_elems)
                # print(f"dbg {source_elements_set=}")
                common_elements_set = \
                    target_elements_set.intersection(source_elements_set)
                # print(f"dbg {common_elements_set=}")
                if common_elements_set:
                    if trg_partid not in mapping:
                        mapping[trg_partid] = {}

                local_mapping = {}
                for trg_el in common_elements_set:
                    # print(f"dbg {trg_el=}")
                    trg_local_idx = np.where(trg_elems == trg_el)[0][0]
                    src_local_idx = np.where(src_elems == trg_el)[0][0]
                    local_mapping[trg_local_idx] = src_local_idx

                # Store the local mapping if there are any overlapping elements
                if local_mapping:
                    # init empty dict if needed
                    # if trg_partid not in mapping:
                    #    mapping[trg_partid] = {}
                    num_local_overlap = len(local_mapping)
                    noverlap = noverlap + num_local_overlap
                    # print(f"dbg ---- found overlap: {trg_partid=}, {src_partid=}")
                    # print(f"dbg ---- num_olap: {num_local_overlap}")
                    mapping[trg_partid][src_partid] = local_mapping
                # else:
                #    print("dbg ---- No overlap found.")
        # if noverlap == trg_nelem_part:
        # print("dbg - Full overlaps found.")
        if noverlap != trg_nelem_part:
            # print("dbg - Overlaps did not cover target part!")
            raise AssertionError("Source overlaps did not cover target part."
                                 f" {trg_partid=}")

    return mapping


def boundary_report(dcoll, boundaries, outfile_name, *, dd=DD_VOLUME_ALL,
                    mesh=None):
    """Generate a report of the mesh boundaries."""
    boundaries = normalize_boundaries(boundaries)

    comm = dcoll.mpi_communicator
    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.Get_size()
        rank = comm.Get_rank()

    if mesh is not None:
        nelem = 0
        for grp in mesh.groups:
            nelem = nelem + grp.nelements
        local_header = f"nproc: {nproc}\nrank: {rank}\nnelem: {nelem}\n"
    else:
        local_header = f"nproc: {nproc}\nrank: {rank}\n"

    from io import StringIO
    local_report = StringIO(local_header)
    local_report.seek(0, 2)

    for bdtag in boundaries:
        boundary_discr = dcoll.discr_from_dd(bdtag)
        nnodes = sum([grp.ndofs for grp in boundary_discr.groups])
        local_report.write(f"{bdtag}: {nnodes}\n")

    from meshmode.distributed import get_connected_parts
    from meshmode.mesh import BTAG_PARTITION
    connected_part_ids = get_connected_parts(dcoll.discr_from_dd(dd).mesh)
    local_report.write(f"num_nbr_parts: {len(connected_part_ids)}\n")
    local_report.write(f"connected_part_ids: {connected_part_ids}\n")
    part_nodes = []
    for connected_part_id in connected_part_ids:
        boundary_discr = dcoll.discr_from_dd(
            dd.trace(BTAG_PARTITION(connected_part_id)))
        nnodes = sum([grp.ndofs for grp in boundary_discr.groups])
        part_nodes.append(nnodes)
    if part_nodes:
        local_report.write(f"nnodes_pb: {part_nodes}\n")

    local_report.write("-----\n")
    local_report.seek(0)

    for irank in range(nproc):
        if irank == rank:
            f = open(outfile_name, "a+")
            f.write(local_report.read())
            f.close()
        if comm is not None:
            comm.barrier()


def force_evaluation(actx, expn):
    """Wrap freeze/thaw forcing evaluation of expressions.

    Deprecated; use :func:`mirgecom.utils.force_evaluation` instead.
    """
    from warnings import warn
    warn("simutil.force_evaluation is deprecated and will disappear in Q3 2023. "
         "Use utils.force_evaluation instead.", DeprecationWarning, stacklevel=2)
    return actx.thaw(actx.freeze(expn))


def get_reasonable_memory_pool(ctx: cl.Context, queue: cl.CommandQueue,
                               force_buffer: bool = False,
                               force_non_pool: bool = False):
    """Return an SVM or buffer memory pool based on what the device supports.

    By default, it prefers SVM allocations over CL buffers, and memory
    pools over direct allocations.
    """
    import pyopencl.tools as cl_tools
    from pyopencl.characterize import has_coarse_grain_buffer_svm

    if force_buffer and force_non_pool:
        logger.info(f"Using non-pooled CL buffer allocations on {queue.device}.")
        return cl_tools.DeferredAllocator(ctx)

    if force_buffer:
        logger.info(f"Using pooled CL buffer allocations on {queue.device}.")
        return cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))

    if force_non_pool and has_coarse_grain_buffer_svm(queue.device):
        logger.info(f"Using non-pooled SVM allocations on {queue.device}.")
        return cl_tools.SVMAllocator(  # pylint: disable=no-member
            ctx, alignment=0, queue=queue)

    if has_coarse_grain_buffer_svm(queue.device) and hasattr(cl_tools, "SVMPool"):
        logger.info(f"Using SVM-based memory pool on {queue.device}.")
        return cl_tools.SVMPool(cl_tools.SVMAllocator(  # pylint: disable=no-member
            ctx, alignment=0, queue=queue))
    else:
        from warnings import warn
        if not has_coarse_grain_buffer_svm(queue.device):
            warn(f"No SVM support on {queue.device}, returning a CL buffer-based "
                  "memory pool. If you are running with PoCL-cuda, please update "
                  "your PoCL installation.")
        else:
            warn("No SVM memory pool support with your version of PyOpenCL, "
                 f"returning a CL buffer-based memory pool on {queue.device}. "
                 "Please update your PyOpenCL version.")
        return cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))


def configurate(config_key, config_object=None, default_value=None):
    """Return a configured item from a configuration object."""
    if config_object is not None:
        d = config_object if isinstance(config_object, dict) else\
            config_object.__dict__
        if config_key in d:
            value = d[config_key]
            if default_value is not None:
                return type(default_value)(value)
            return value
    return default_value


def compare_files_vtu(
        first_file: str,
        second_file: str,
        file_type: str,
        tolerance: float = 1e-12,
        field_tolerance: Optional[Dict[str, float]] = None
        ) -> None:
    """Compare files of vtu type.

    Parameters
    ----------
    first_file:
        First file to compare
    second_file:
        Second file to compare
    file_type:
        Vtu files
    tolerance:
        Max acceptable absolute difference
    field_tolerance:
        Dictionary of individual field tolerances

    Returns
    -------
    True:
        If it passes the files contain data within the given tolerance.
    False:
        If it fails the files contain data outside the given tolerance.
    """
    import xml.etree.ElementTree as Et

    import vtk

    # read files:
    if file_type == "vtu":
        reader1 = vtk.vtkXMLUnstructuredGridReader()  # pylint: disable=no-member
        reader2 = vtk.vtkXMLUnstructuredGridReader()  # pylint: disable=no-member
    else:
        reader1 = vtk.vtkXMLPUnstructuredGridReader()  # pylint: disable=no-member
        reader2 = vtk.vtkXMLPUnstructuredGridReader()  # pylint: disable=no-member

    reader1.SetFileName(first_file)
    reader1.Update()
    output1 = reader1.GetOutput()

    # Check rank number
    def numranks(filename: str) -> int:
        tree = Et.parse(filename)
        root = tree.getroot()
        return len(root.findall(".//Piece"))

    if file_type == "pvtu":
        rank1 = numranks(first_file)
        rank2 = numranks(second_file)
        if rank1 != rank2:
            raise ValueError(f"File '{first_file}' has {rank1} ranks, "
                f"but file '{second_file}' has {rank2} ranks.")

    reader2.SetFileName(second_file)
    reader2.Update()
    output2 = reader2.GetOutput()

    # check fidelity
    point_data1 = output1.GetPointData()
    point_data2 = output2.GetPointData()

    # verify same number of PointData arrays in both files
    if point_data1.GetNumberOfArrays() != point_data2.GetNumberOfArrays():
        print("File 1:", point_data1.GetNumberOfArrays(), "\n",
              "File 2:", point_data2.GetNumberOfArrays())
        raise ValueError("Fidelity test failed: Mismatched data array count")

    nfields = point_data1.GetNumberOfArrays()
    max_field_errors = [0 for _ in range(nfields)]

    if field_tolerance is None:
        field_tolerance = {}
    field_specific_tols = [configurate(point_data1.GetArrayName(i),
        field_tolerance, tolerance) for i in range(nfields)]
    field_names = []

    max_file_diff = 0.
    max_field_name = ""

    for i in range(nfields):
        arr1 = point_data1.GetArray(i)
        arr2 = point_data2.GetArray(i)
        field_tol = field_specific_tols[i]
        num_points = arr2.GetNumberOfTuples()
        num_components = arr1.GetNumberOfComponents()

        # verify both files contain same arrays
        if point_data1.GetArrayName(i) != point_data2.GetArrayName(i):
            print("File 1:", point_data1.GetArrayName(i), "\n",
                  "File 2:", point_data2.GetArrayName(i))
            raise ValueError("Fidelity test failed: Mismatched data array names")

        # verify arrays are same sizes in both files
        if arr1.GetSize() != arr2.GetSize():
            print("File 1, DataArray", i, ":", arr1.GetSize(), "\n",
                  "File 2, DataArray", i, ":", arr2.GetSize())
            raise ValueError("Fidelity test failed: Mismatched data array sizes")

        # verify individual values w/in given tolerance
        fieldname = point_data1.GetArrayName(i)
        field_names.append(fieldname)

        # Ignore any fields that are named here
        # FIXME: rhs, grad are here because they fail thermally-coupled example
        ignored_fields = ["resid", "tagged", "grad", "rhs"]
        ignore_field = False
        for ifld in ignored_fields:
            if ifld in fieldname:
                ignore_field = True
        if ignore_field:
            continue

        max_true_component = max([max(abs(arr2.GetComponent(j, icomp))
                                      for j in range(num_points))
                                  for icomp in range(num_components)])
        max_other_component = max([max(abs(arr1.GetComponent(j, icomp))
                                      for j in range(num_points))
                                  for icomp in range(num_components)])

        # FIXME
        # Choose an arbitrary "level" to define what is a "significant" field
        # Don't compare dinky/insignificant fields
        significant_field = 1000.*field_tol
        if ((max_true_component
             < significant_field) and (max_other_component < significant_field)):
            continue

        max_true_value = max_true_component
        use_relative_diff = max_true_value > field_tol
        err_scale = 1./max_true_value if use_relative_diff else 1.

        print(f"{fieldname}: Max true value: {max_true_value}")
        print(f"{fieldname}: Error scale: {err_scale}")

        # Spew out some stats on each field component
        print(f"Field({fieldname}) Component (min, max, mean)s:")
        for icomp in range(num_components):
            min_true_value = min(arr2.GetComponent(j, icomp)
                                 for j in range(num_points))
            min_other_value = min(arr1.GetComponent(j, icomp)
                                  for j in range(num_points))
            max_true_value = max(arr2.GetComponent(j, icomp)
                                 for j in range(num_points))
            max_other_value = max(arr1.GetComponent(j, icomp)
                                  for j in range(num_points))
            true_mean = sum(arr2.GetComponent(j, icomp)
                            for j in range(num_points)) / num_points
            other_mean = sum(arr1.GetComponent(j, icomp)
                             for j in range(num_points)) / num_points
            print(f"{fieldname}({icomp}): ({min_other_value},"
                  f" {max_other_value}, {other_mean})")
            print(f"{fieldname}({icomp}): ({min_true_value},"
                  f" {max_true_value}, {true_mean})")

        print(f"Field({fieldname}) comparison:")

        for icomp in range(num_components):
            if num_components > 1:
                print(f"Comparing component {icomp}")
            for j in range(num_points):
                test_err = abs(arr1.GetComponent(j, icomp)
                               - arr2.GetComponent(j, icomp))*err_scale
                if test_err > max_field_errors[i]:
                    max_field_errors[i] = test_err

        print(f"Max Error: {max_field_errors[i]}", end=" ")
        print(f"Tolerance: {field_specific_tols[i]}")

    violated_tols = []
    violating_values = []
    violating_fields = []

    for i in range(nfields):
        if max_field_errors[i] > max_file_diff:
            max_file_diff = max_field_errors[i]
            max_field_name = field_names[i]

        if max_field_errors[i] > field_specific_tols[i]:
            violated_tols.append(field_specific_tols[i])
            violating_values.append(max_field_errors[i])
            violating_fields.append(field_names[i])

    print(f"Max File Difference: {max_field_name} : {max_file_diff}")

    if violated_tols:
        raise ValueError("Data comparison failed:\n"
                         f"Fields: {violating_fields=}\n"
                         f"Max differences: {violating_values=}\n"
                         f"Tolerances: {violated_tols=}\n")

    print("VTU Fidelity test completed successfully")


class _Hdf5Reader:
    def __init__(self, filename):
        import h5py

        self.file_obj = h5py.File(filename, "r")

    def read_specific_data(self, datapath):
        return self.file_obj[datapath]


class _XdmfReader:
    # CURRENTLY DOES NOT SUPPORT MULTIPLE Grids

    def __init__(self, filename):
        from xml.etree import ElementTree

        tree = ElementTree.parse(filename)
        root = tree.getroot()

        domains = tuple(root)
        self.domain = domains[0]
        self.grids = tuple(self.domain)
        self.uniform_grid = self.grids[0]

    def get_topology(self):
        connectivity = None

        for a in self.uniform_grid:
            if a.tag == "Topology":
                connectivity = a

        if connectivity is None:
            raise ValueError("File is missing mesh connectivity data")

        return connectivity

    def get_geometry(self):
        geometry = None

        for a in self.uniform_grid:
            if a.tag == "Geometry":
                geometry = a

        if geometry is None:
            raise ValueError("File is missing mesh node location data")

        return geometry

    def read_data_item(self, data_item):
        # CURRENTLY DOES NOT SUPPORT 'DataItem' THAT STORES VALUES DIRECTLY

        # check that data stored as separate hdf5 file
        if data_item.get("Format") != "HDF":
            raise TypeError("Data stored in unrecognized format")

        # get corresponding hdf5 file
        source_info = data_item.text
        split_source_info = source_info.partition(":")

        h5_filename = split_source_info[0]
        # TODO: change file name to match actual mirgecom output directory later
        h5_filename = "examples/" + h5_filename
        h5_datapath = split_source_info[2]

        # read data from corresponding hdf5 file
        h5_reader = _Hdf5Reader(h5_filename)
        return h5_reader.read_specific_data(h5_datapath)


def compare_files_xdmf(first_file: str, second_file: str, tolerance: float = 1e-12):
    """Compare files of xdmf type.

    Parameters
    ----------
    first_file:
        First file to compare
    second_file:
        Second file to compare
    file_type:
        Xdmf files
    tolerance:
        Max acceptable absolute difference

    Returns
    -------
    True:
        If it passes the file type test or contains same data.
    False:
        If it fails the file type test or contains different data.
    """
    # read files
    file_reader1 = _XdmfReader(first_file)
    file_reader2 = _XdmfReader(second_file)

    # check same number of grids
    if len(file_reader1.grids) != len(file_reader2.grids):
        print("File 1:", len(file_reader1.grids), "\n",
              "File 2:", len(file_reader2.grids))
        raise ValueError("Fidelity test failed: Mismatched grid count")

    # check same number of cells in gridTrue:
    if len(file_reader1.uniform_grid) != len(file_reader2.uniform_grid):
        print("File 1:", len(file_reader1.uniform_grid), "\n",
              "File 2:", len(file_reader2.uniform_grid))
        raise ValueError("Fidelity test failed: Mismatched cell count in "
                         "uniform grid")

    # compare Topology:
    top1 = file_reader1.get_topology()
    top2 = file_reader2.get_topology()

    # check TopologyType
    if top1.get("TopologyType") != top2.get("TopologyType"):
        print("File 1:", top1.get("TopologyType"), "\n",
              "File 2:", top2.get("TopologyType"))
        raise ValueError("Fidelity test failed: Mismatched topology type")

    # check number of connectivity values
    connectivities1 = file_reader1.read_data_item(tuple(top1)[0])
    connectivities2 = file_reader2.read_data_item(tuple(top2)[0])

    connectivities1 = np.array(connectivities1)
    connectivities2 = np.array(connectivities2)

    if connectivities1.shape != connectivities2.shape:
        print("File 1:", connectivities1.shape, "\n",
              "File 2:", connectivities2.shape)
        raise ValueError("Fidelity test failed: Mismatched connectivities count")

    if not np.allclose(connectivities1, connectivities2, atol=tolerance):
        print("Tolerance:", tolerance)
        raise ValueError("Fidelity test failed: Mismatched connectivity values "
                         "with given tolerance")

    # compare Geometry:
    geo1 = file_reader1.get_geometry()
    geo2 = file_reader2.get_geometry()

    # check GeometryType
    if geo1.get("GeometryType") != geo2.get("GeometryType"):
        print("File 1:", geo1.get("GeometryType"), "\n",
              "File 2:", geo2.get("GeometryType"))
        raise ValueError("Fidelity test failed: Mismatched geometry type")

    # check number of node values
    nodes1 = file_reader1.read_data_item(tuple(geo1)[0])
    nodes2 = file_reader2.read_data_item(tuple(geo2)[0])

    nodes1 = np.array(nodes1)
    nodes2 = np.array(nodes2)

    if nodes1.shape != nodes2.shape:
        print("File 1:", nodes1.shape, "\n", "File 2:", nodes2.shape)
        raise ValueError("Fidelity test failed: Mismatched nodes count")

    if not np.allclose(nodes1, nodes2, atol=tolerance):
        print("Tolerance:", tolerance)
        raise ValueError("Fidelity test failed: Mismatched node values with "
                         "given tolerance")

    # compare other Attributes:
    maxerrorvalue = 0
    for i in range(len(file_reader1.uniform_grid)):
        curr_cell1 = file_reader1.uniform_grid[i]
        curr_cell2 = file_reader2.uniform_grid[i]

        # skip already checked cells
        if curr_cell1.tag == "Geometry" or curr_cell1.tag == "Topology":
            continue

        # check AttributeType
        if curr_cell1.get("AttributeType") != curr_cell2.get("AttributeType"):
            print("File 1:", curr_cell1.get("AttributeType"), "\n",
                  "File 2:", curr_cell2.get("AttributeType"))
            raise ValueError("Fidelity test failed: Mismatched cell type")

        # check Attribtue name
        if curr_cell1.get("Name") != curr_cell2.get("Name"):
            print("File 1:", curr_cell1.get("Name"), "\n",
                  "File 2:", curr_cell2.get("Name"))
            raise ValueError("Fidelity test failed: Mismatched cell name")

        # check number of Attribute values
        values1 = file_reader1.read_data_item(tuple(curr_cell1)[0])
        values2 = file_reader2.read_data_item(tuple(curr_cell2)[0])

        if len(values1) != len(values2):
            print("File 1,", curr_cell1.get("Name"), ":", len(values1), "\n",
                  "File 2,", curr_cell2.get("Name"), ":", len(values2))
            raise ValueError("Fidelity test failed: Mismatched data values count")

        # check values w/in tolerance
        for i in range(len(values1)):
            if abs(values1[i] - values2[i]) > tolerance:
                print("Tolerance:", tolerance, "\n", "Cell:", curr_cell1.get("Name"))
                if maxerrorvalue < abs(values1[i] - values2[i]):
                    maxerrorvalue = abs(values1[i] - values2[i])

    if not maxerrorvalue == 0:
        raise ValueError("Fidelity test failed: Mismatched data array "
                                 "values with given tolerance. "
                                 "Max Error Value:", maxerrorvalue)

    print("XDMF Fidelity test completed successfully with tolerance", tolerance)


def compare_files_hdf5(first_file: str, second_file: str, tolerance: float = 1e-12):
    """Compare files of hdf5 type.

    Parameters
    ----------
    first_file:
        First file to compare
    second_file:
        Second file to compare
    file_type:
        Hdf5 files
    tolerance:
        Max acceptable absolute difference

    Returns
    -------
    True:
        If it passes the file type test or contains same data.
    False:
        If it fails the file type test or contains different data.
    """
    file_reader1 = _Hdf5Reader(first_file)
    file_reader2 = _Hdf5Reader(second_file)
    f1 = file_reader1.file_obj
    f2 = file_reader2.file_obj

    objects1 = list(f1.keys())
    objects2 = list(f2.keys())

    # check number of Grids
    if len(objects1) != len(objects2):
        print("File 1:", len(objects1), "\n", "File 2:", len(objects2))
        raise ValueError("Fidelity test failed: Mismatched grid count")

    # loop through Grids
    maxvalueerror = 0
    for i in range(len(objects1)):
        obj_name1 = objects1[i]
        obj_name2 = objects2[i]

        if obj_name1 != obj_name2:
            print("File 1:", obj_name1, "\n", "File 2:", obj_name2)
            raise ValueError("Fidelity test failed: Mismatched grid names")

        curr_o1 = list(f1[obj_name1])
        curr_o2 = list(f2[obj_name2])

        if len(curr_o1) != len(curr_o2):
            print("File 1,", obj_name1, ":", len(curr_o1), "\n",
                  "File 2,", obj_name2, ":", len(curr_o2))
            raise ValueError("Fidelity test failed: Mismatched group count")

        # loop through Groups
        for j in range(len(curr_o1)):
            subobj_name1 = curr_o1[j]
            subobj_name2 = curr_o2[j]

            if subobj_name1 != subobj_name2:
                print("File 1:", subobj_name1, "\n", "File 2:", subobj_name2)
                raise ValueError("Fidelity test failed: Mismatched group names")

            subpath1 = obj_name1 + "/" + subobj_name1
            subpath2 = obj_name2 + "/" + subobj_name2

            data_arrays_list1 = list(f1[subpath1])
            data_arrays_list2 = list(f2[subpath2])

            if len(data_arrays_list1) != len(data_arrays_list2):
                print("File 1,", subobj_name1, ":", len(data_arrays_list1), "\n",
                      "File 2,", subobj_name2, ":", len(data_arrays_list2))
                raise ValueError("Fidelity test failed: Mismatched data list count")

            # loop through data arrays
            for k in range(len(data_arrays_list1)):
                curr_listname1 = data_arrays_list1[k]
                curr_listname2 = data_arrays_list2[k]

                if curr_listname1 != curr_listname2:
                    print("File 1:", curr_listname1, "\n", "File 2:", curr_listname2)
                    raise ValueError("Fidelity test failed: Mismatched data "
                                     "list names")

                curr_listname1 = subpath1 + "/" + curr_listname1
                curr_listname2 = subpath2 + "/" + curr_listname2

                curr_datalist1 = np.array(list(f1[curr_listname1]))
                curr_datalist2 = np.array(list(f2[curr_listname2]))

                if curr_datalist1.shape != curr_datalist2.shape:
                    print("File 1,", curr_listname1, ":", curr_datalist1.shape, "\n",
                          "File 2,", curr_listname2, ":", curr_datalist2.shape)
                    raise ValueError("Fidelity test failed: Mismatched data "
                                     "list size")

                if not np.allclose(curr_datalist1, curr_datalist2, atol=tolerance):
                    print("Tolerance:", tolerance, "\n",
                          "Data List:", curr_listname1)
                    if maxvalueerror < abs(curr_datalist1 - curr_datalist2):
                        maxvalueerror = abs(curr_datalist1 - curr_datalist2)

    if not maxvalueerror == 0:
        raise ValueError("Fidelity test failed: Mismatched data "
                             "values with given tolerance. "
                             "Max Value Error: ", maxvalueerror)

    print("HDF5 Fidelity test completed successfully with tolerance", tolerance)
