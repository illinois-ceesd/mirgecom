""":mod:`mirgecom.filter` is for filters and filter-related constructs.

Discussion of the spectral filter design can be found in:
JSH/TW Nodal DG Methods (DOI: 10.1007/978-0-387-72067-8), Section 5.3

.. automethod: exponential_mode_response_function
.. automethod: make_spectral_filter
.. automethod: apply_filter_matrix
.. automethod: filter_modally
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

import numpy as np
import grudge.dof_desc as dof_desc

from meshmode.dof_array import DOFArray
from pytools import keyed_memoize_in
from pytools.obj_array import obj_array_vectorized_n_args


def exponential_mode_response_function(mode, alpha, cutoff, nfilt, filter_order):
    """Return an exponential filter coefficient for the user-provided *mode*."""
    return np.exp(-1.0 * alpha * ((mode - cutoff) / nfilt)
                  ** (2*filter_order))


def make_spectral_filter(actx, group, cutoff, mode_response_function):
    r"""
    Create a spectral filter with the provided *mode_response_function*.

    This routine returns a filter operator in the modal basis designed to
    apply the user-provided *mode_response_function* to the spectral modes
    beyond the user-provided *cutoff*.

    Parameters
    ----------
    actx: :class:`meshmode.array_context.ArrayContext`
        A :class:`meshmode.array_context.ArrayContext` associated with
        an array of degrees of freedom
    group: :class:`meshmode.mesh.MeshElementGroup`
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

    @keyed_memoize_in(
        actx, (make_spectral_filter,
               mode_response_function,
               "_spectral_filter_matrix"),
        lambda grp: grp.discretization_key()
    )
    def _spectral_filter_matrix(group):
        mode_ids = group.mode_ids()
        order = group.order

        nmodes = len(mode_ids)
        filter_mat = np.identity(nmodes)
        nfilt = order - cutoff
        if nfilt <= 0:
            return filter

        for mode_index, mode_id in enumerate(mode_ids):
            mode = sum(mode_id)
            if mode >= cutoff:
                filter_mat[mode_index, mode_index] = \
                    mode_response_function(mode,
                                           cutoff=cutoff,
                                           nfilt=nfilt)

        return actx.freeze(actx.from_numpy(filter_mat))

    return _spectral_filter_matrix(group)


def apply_filter_matrix(actx, modal_field, discr, cutoff,
                        mode_response_function):
    r"""
    Applies the filter matrix, defined by the *mode_response_function*.

    This routine returns filtered data in the modal basis, which has
    been applied using a user-provided *mode_response_function*
    to dampen modes beyond the user-provided *cutoff*.

    Parameters
    ----------
    actx: :class:`meshmode.array_context.ArrayContext`
        A :class:`meshmode.array_context.ArrayContext` associated with
        an array of degrees of freedom
    modal_field: numpy.ndarray
        DOFArray or object array of DOFArrays denoting the modal data
    discr: :class:`meshmode.discretization.Discretization`
        A :class:`meshmode.discretization.Discretization` describing
        the volume discretization the *modal_field* comes from.
    cutoff: integer
        Mode cutoff beyond which the filter will be applied, and below which
        the filter will preserve.
    mode_response_function:
        A function that returns a filter weight for for each mode id.

    Returns
    -------
    modal_field: DOFArray
        DOFArray or object array of DOFArrays

    """
    from meshmode.array_context import FirstAxisIsElementsTag
    return DOFArray(
        actx,
        tuple(actx.einsum("ij,ej->ei",
                          make_spectral_filter(
                              actx,
                              group=grp,
                              cutoff=cutoff,
                              mode_response_function=mode_response_function
                          ),
                          vec_i,
                          arg_names=("filter_mat", "vec"),
                          tagged=(FirstAxisIsElementsTag(),))
              for grp, vec_i in zip(discr.groups, modal_field))
    )


@obj_array_vectorized_n_args
def filter_modally(dcoll, dd, cutoff, mode_resp_func, field):
    """Stand-alone procedural interface to spectral filtering.

    For each element group in the discretization, and restriction,
    This routine generates:
    * a filter operator:
        - *cutoff* filters only modes above this mode id
        - *mode_resp_func* function returns a filter coefficient
        for a given mode
        - memoized into the discretization
    * a kernel to apply the operator
        - memoized into the array context
    * a filtered solution wherein the filter is applied to *field*.

    Parameters
    ----------
    dcoll: :class:`grudge.DiscretizationCollection`
        Grudge discretization with boundaries object
    dd: :class:`grudge.dof_desc.DOFDesc` or as accepted by
        :func:`grudge.dof_desc.as_dof_desc`
        Describe the type of DOF vector on which to operate.
    cutoff: integer
        Mode below which *field* will not be filtered
    mode_resp_func:
        Modal response function returns a filter coefficient for input mode id
    field: numpy.ndarray
        DOFArray or object array of DOFArrays

    Returns
    -------
    result: numpy.ndarray
        Filtered version of *field*.
    """
    dd = dof_desc.as_dofdesc(dd)
    dd_modal = dof_desc.DD_VOLUME_MODAL
    discr = dcoll.discr_from_dd(dd)

    assert isinstance(field, DOFArray)
    actx = field.array_context

    modal_map = dcoll.connection_from_dds(dd, dd_modal)
    nodal_map = dcoll.connection_from_dds(dd_modal, dd)
    field = modal_map(field)
    field = apply_filter_matrix(actx, field, discr, cutoff,
                                mode_resp_func)
    return nodal_map(field)
