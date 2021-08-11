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
    _, ntest_elements, nnodes = src_nodes.shape
    nnodes = 1
    ntest_elements = 1 #<-- Keep for now, comment out when batch testing

    initial_guess = np.array([0.5, 0.5, 0.5])
    src_unit_nodes = np.empty((dim, ntest_elements, nnodes))
    src_unit_nodes[:] = initial_guess.reshape(-1, 1, 1)

    src_grp_basis_fcts = src_grp.basis_obj().functions
    vdm = mp.vandermonde(src_grp_basis_fcts, src_grp.unit_nodes)
    inv_t_vdm = la.inv(vdm.T)
    nsrc_funcs = len(src_grp_basis_fcts)

    def apply_map(unit_nodes):
        # unit_nodes: (dim, ntest_elements, nnodes)

        # basis_at_unit_nodes
        basis_at_unit_nodes = np.empty((nsrc_funcs, ntest_elements, nnodes))

        for i, f in enumerate(src_grp_basis_fcts):
            basis_at_unit_nodes[i] = (
                    f(unit_nodes).reshape(ntest_elements, nnodes))

        intp_coeffs = np.einsum("fj,jet->fet", inv_t_vdm, basis_at_unit_nodes)

        # If we're interpolating 1, we had better get 1 back.
        one_deviation = np.abs(np.sum(intp_coeffs, axis=0) - 1)
        assert (one_deviation < tol).all(), np.max(one_deviation)

        mapped = np.einsum("fet,aef->aet", intp_coeffs, src_nodes)
        assert query_nodes.shape == mapped.shape
        return mapped

    def get_map_jacobian(unit_nodes):
        # unit_nodes: (dim, ntest_elements, nnodes)

        # basis_at_unit_nodes
        dbasis_at_unit_nodes = np.empty(
                (dim, nsrc_funcs, ntest_elements, nnodes))

        for i, df in enumerate(src_grp.basis_obj().gradients):
            df_result = df(unit_nodes.reshape(dim, -1))

            for rst_axis, df_r in enumerate(df_result):
                dbasis_at_unit_nodes[rst_axis, i] = (
                        df_r.reshape(ntest_elements, nnodes))

        dintp_coeffs = np.einsum(
                "fj,rjet->rfet", inv_t_vdm, dbasis_at_unit_nodes)

        return np.einsum("rfet,aef->raet", dintp_coeffs, src_nodes)

    niter = 0
    while True:
        resid = apply_map(src_nodes) - query_point

        df = get_map_jacobian(src_nodes)
        df_inv_resid = np.empty_like(src_unit_nodes)

        # For the 1D/2D accelerated versions, we'll use the normal
        # equations and Cramer's rule. If you're looking for high-end
        # numerics, look no further than meshmode.

        if dim == 1:
            # A is df.T
            ata = np.einsum("iket,jket->ijet", df, df)
            atb = np.einsum("iket,ket->iet", df, resid)

            df_inv_resid = atb / ata[0, 0]

        elif dim == 2:
            # A is df.T
            ata = np.einsum("iket,jket->ijet", df, df)
            atb = np.einsum("iket,ket->iet", df, resid)

            det = ata[0, 0]*ata[1, 1] - ata[0, 1]*ata[1, 0]

            df_inv_resid = np.empty_like(src_nodes)
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
                for t in range(nnodes):
                    df_inv_resid[:, e, t], _, _, _ = \
                            la.lstsq(df[:, :, e, t].T, resid[:, e, t])

        src_nodes = src_nodes - df_inv_resid

        max_resid = np.max(np.abs(resid))

        if max_resid < tol:
            logger.debug("_find_src_unit_nodes_via_gauss_newton: done, "
                    "final residual: %g", max_resid)
            return src_nodes

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
