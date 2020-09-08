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
import loopy as lp
from pytools.obj_array import obj_array_vectorized_n_args


def exponential_mode_response_function(mode, alpha, cutoff, nfilt, filter_order):
    return np.exp(-1.0 * alpha * ((mode - cutoff) / nfilt)
                  ** (2*filter_order))


def make_spectral_filter(element_group, cutoff, mode_response_function):
    r"""
    Create a spectral filter with the provided *mode_response_function*.

    Discussion of the spectral filter design can be found in:
    JSH/TW Nodal DG Methods (DOI: 10.1007/978-0-387-72067-8), Section 5.3
    This routine returns a filter operator in the modal basis designed to
    apply the user-provided *mode_response_function* to the spectral modes
    beyond the user-provided *cutoff*.

    Parameters
    ----------
    element_group: :class:`meshmode.mesh.MeshElementGroup`
        A :class:`meshmode.mesh.MeshElementGroup` from which the mode ids,
        element order, and dimension may be retrieved.
    cutoff: integer
        Mode cutoff beyond which the filter will be applied, and below which
        the filter will preserve.
    mode_response_function:
        A function that returns a filter weight for for each mode id.

    Returns
    -------
    filter: np.ndarray
        filter operator in the modal basis
    """
    mode_ids = element_group.mode_ids()
    order = element_group.order
    dim = element_group.dim

    nmodes = len(mode_ids)
    filter = np.identity(nmodes)
    nfilt = order - cutoff
    if nfilt <= 0:
        return filter

    for mode_index, mode_id in enumerate(mode_ids):
        mode = mode_id
        if dim > 1:
            mode = sum(mode_id)
        if mode >= cutoff:
            filter[mode_index, mode_index] = mode_response_function(mode,
                                                                    cutoff=cutoff,
                                                                    nfilt=nfilt)
    return filter


def linear_operator_kernel():
    from meshmode.array_context import make_loopy_program
    knl = make_loopy_program(
        """{[iel,idof,j]:
        0<=iel<nelements and
        0<=idof<ndiscr_nodes_out and
        0<=j<ndiscr_nodes_in}""",
        "result[iel,idof] = sum(j, mat[idof, j] * vec[iel, j])",
        name="spectral_filter")
    knl = lp.tag_array_axes(knl, "mat", "stride:auto,stride:auto")
    return knl


class SpectralFilter:
    r"""
    Encapsulates the simulation-static filter operators and
    provides the methods to apply filtering to input fields.
    """
    def __init__(self, discr, filter_mat):
        self._filter_operators = []
        from modepy import vandermonde
        for group in discr.groups:
            vander = vandermonde(group.basis(), group.unit_nodes)
            vanderm1 = np.linalg.inv(vander)
            filter_operator = vander @ filter_mat @ vanderm1
            self._filter_operators.append(filter_operator)
        self._knl = linear_operator_kernel()

    @obj_array_vectorized_n_args
    def __call__(self, discr, fields):
        actx = fields.array_context
        result = discr.empty(fields.array_context, dtype=fields.entry_dtype)
        for group_index, group in enumerate(discr.groups):
            # FIXME Do the transfer once, not repeatedly
            filter_operator = actx.from_numpy(self._filter_operators[group_index])
            actx.call_loopy(self._knl,
                    mat=filter_operator,
                    result=result[group_index],
                    vec=fields[group_index])
        return result


@obj_array_vectorized_n_args
def apply_linear_operator(discr, operator, fields):
    actx = fields.array_context
    result = discr.empty(actx, dtype=fields.entry_dtype)
    for group in discr.groups:
        actx.call_loopy(
            linear_operator_kernel(),
            mat=actx.from_numpy(operator),
            result=result[group.index],
            vec=fields[group.index])
    return result
