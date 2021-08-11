__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
import numpy.linalg as la
import modepy as mp

from pytools.obj_array import make_obj_array

from meshmode.dof_array import DOFArray, thaw

def _get_query_map(query_point, src_nodes, src_grp, tol):
    ################################ MAIN MAPPING CODE ###################################
    dim = src_grp.dim
    _, ntest_elements, _ = src_nodes.shape

    initial_guess = np.array([0.5, 0.5, 0.5])
    src_unit_query_points = np.empty((dim, ntest_elements))
    src_unit_query_points[:] = initial_guess.reshape(-1, 1)

    src_grp_basis_fcts = src_grp.basis_obj().functions
    vdm = mp.vandermonde(src_grp_basis_fcts, src_grp.unit_query_points)
    inv_t_vdm = la.inv(vdm.T)
    nsrc_funcs = len(src_grp_basis_fcts)

    def apply_map(unit_query_points):
        # unit_query_points: (dim, ntest_elements)

        # basis_at_unit_query_points
        basis_at_unit_query_points = np.empty((nsrc_funcs, ntest_elements))

        for i, f in enumerate(src_grp_basis_fcts):
            basis_at_unit_query_points[i] = (
                    f(unit_query_points).reshape(ntest_elements))

        intp_coeffs = np.einsum("fj,je->fe", inv_t_vdm, basis_at_unit_query_points)

        # If we're interpolating 1, we had better get 1 back.
        one_deviation = np.abs(np.sum(intp_coeffs, axis=0) - 1)
        assert (one_deviation < tol).all(), np.max(one_deviation)

        mapped = np.einsum("fe,aef->ae", intp_coeffs, src_nodes)
        return mapped

    def get_map_jacobian(unit_query_points):
        # unit_query_points: (dim, ntest_elements)

        # basis_at_unit_query_points
        dbasis_at_unit_query_points = np.empty(
                (dim, nsrc_funcs, ntest_elements))

        for i, df in enumerate(src_grp.basis_obj().gradients):
            df_result = df(unit_query_points.reshape(dim, -1))

            for rst_axis, df_r in enumerate(df_result):
                dbasis_at_unit_query_points[rst_axis, i] = (
                        df_r.reshape(ntest_elements))

        dintp_coeffs = np.einsum(
                "fj,rje->rfe", inv_t_vdm, dbasis_at_unit_query_points)

        return np.einsum("rfe,aef->rae", dintp_coeffs, src_nodes)

    niter = 0
    while True:
        resid = apply_map(src_unit_query_points) - query_point[:, np.newaxis]

        df = get_map_jacobian(src_unit_query_points)
        df_inv_resid = np.empty_like(src_unit_query_points)

        # For the 1D/2D accelerated versions, we'll use the normal
        # equations and Cramer's rule. If you're looking for high-end
        # numerics, look no further than meshmode.

        if dim == 1:
            # A is df.T
            ata = np.einsum("ike,jke->ije", df, df)
            atb = np.einsum("ike,ke->ie", df, resid)

            df_inv_resid = atb / ata[0, 0]

        elif dim == 2:
            # A is df.T
            ata = np.einsum("ike,jke->ije", df, df)
            atb = np.einsum("ike,ke->ie", df, resid)

            det = ata[0, 0]*ata[1, 1] - ata[0, 1]*ata[1, 0]

            df_inv_resid = np.empty_like(src_unit_query_points)
            df_inv_resid[0] = 1/det * (ata[1, 1] * atb[0] - ata[1, 0]*atb[1])
            df_inv_resid[1] = 1/det * (-ata[0, 1] * atb[0] + ata[0, 0]*atb[1])

        else:
            # The boundary of a 3D mesh is 2D, so that's the
            # highest-dimensional case we genuinely care about.
            #
            # This stinks, performance-wise, because it's not vectorized.
            # But we'll only hit it for boundaries of 4+D meshes, in which
            # case... good luck. :)
            for e in range(ntest_elements):
                df_inv_resid[:, e], _, _, _ = \
                        la.lstsq(df[:, :, e].T, resid[:, e])

        src_unit_query_points = src_unit_query_points - df_inv_resid

        max_resid = np.max(np.abs(resid))

        if max_resid < tol:
            return src_unit_query_points

        niter += 1
        if niter > 10:
            raise RuntimeError("Gauss-Newton (for finding opposite-face reference "
                    "coordinates) did not converge (residual: %g)" % max_resid)

    raise AssertionError()

    ################################ MAPPING CODE ENDS ###################################

def query_eval(query_point, actx, discr, dim, tol):
    nodes = thaw(actx, discr.nodes())

    vol_discr = discr.discr_from_dd("vol")

    query_mapped = None

    for igrp, src_grp in enumerate(vol_discr.groups):
        grp_nodes = np.stack([actx.to_numpy(nodes[i][igrp]) for i in range(dim)])
        box_ls = np.stack([coords.min(axis=1) for coords in grp_nodes])
        box_us = np.stack([coords.max(axis=1) for coords in grp_nodes])
        overlaps_in_dim = (
            (query_point[:, np.newaxis] >= box_ls)
            & (query_point[:, np.newaxis] <= box_us))
        overlaps = overlaps_in_dim[0]
        for i in range(1, dim):
            overlaps = overlaps & overlaps_in_dim[i]
        matched_elems, = np.where(overlaps)
        src_nodes = np.stack([grp_nodes[i][matched_elems, :] for i in range(dim)])
        query_mapped_cand = _get_query_map(query_point, src_nodes, src_grp, tol)
        # TODO: Figure out which candidate element actually contains the query point
        query_mapped = query_mapped_cand[:, 0]

    return query_mapped
