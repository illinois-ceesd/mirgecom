""":mod:`mirgecom.filter` is for filters and filter-related constructs.

Discussion of the spectral filter design can be found in:
[Hesthaven_2008]_, Section 5.3

Mode Response Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: exponential_mode_response_function

Helper Functions
^^^^^^^^^^^^^^^^

.. autofunction:: make_spectral_filter
.. autofunction:: apply_spectral_filter

Applying Filters
^^^^^^^^^^^^^^^^

.. autofunction:: filter_modally

Spectral Analysis Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_element_spectrum_from_modal_representation
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

from functools import partial

import numpy as np
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    DISCR_TAG_MODAL,
)
from meshmode.dof_array import DOFArray

from pytools import keyed_memoize_in

from arraycontext import map_array_container


# TODO: Revisit for multi-group meshes
def get_element_spectrum_from_modal_representation(actx, vol_discr, modal_fields,
                                                   element_order):
    """Get the Legendre Polynomial expansion for specified fields on elements.

    Parameters
    ----------
    actx: :class:`arraycontext.ArrayContext`
        A :class:`arraycontext.ArrayContext` associated with
        an array of degrees of freedom
    vol_discr: :class:`grudge.discretization.DiscretizationCollection`
        Grudge discretization for volume elements only
    modal_fields: numpy.ndarray
        Array of DOFArrays with modal respresentations for each field
    element_order: int
        Polynomial order for the elements in the discretization
    Returns
    -------
    numpy.ndarray
        Array with the element modes accumulated into the corresponding
        "modes" for the polynomial basis functions for each field.
    """
    modal_spectra = np.stack([
        actx.to_numpy(ary)[0]
        for ary in modal_fields])

    numfields, numelem, nummodes = modal_spectra.shape

    emodes_to_pmodes = np.array([0 for _ in range(nummodes)], dtype=np.uint32)

    for group in vol_discr.groups:
        mode_ids = group.mode_ids()
        for modi, mode in enumerate(mode_ids):
            emodes_to_pmodes[modi] = sum(mode)

    accumulated_spectra = np.zeros(
        (numfields, numelem, element_order+1), dtype=np.float64)

    for i in range(nummodes):
        accumulated_spectra[:, :, emodes_to_pmodes[i]] += np.abs(
            modal_spectra[:, :, i])

    return accumulated_spectra


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
    actx: :class:`arraycontext.ArrayContext`
        A :class:`arraycontext.ArrayContext` associated with
        an array of degrees of freedom
    group: :class:`meshmode.mesh.MeshElementGroup`
        A :class:`meshmode.mesh.MeshElementGroup` from which the mode ids,
        element order, and dimension may be retrieved.
    cutoff: int
        Mode cutoff beyond which the filter will be applied, and below which
        the filter will preserve.
    mode_response_function:
        A function that returns a filter weight for for each mode id.

    Returns
    -------
    filter: :class:`numpy.ndarray`
        filter operator in the modal basis
    """

    @keyed_memoize_in(
        actx, (make_spectral_filter,
               mode_response_function,
               "_spectral_filter_scaling"),
        lambda grp: grp.discretization_key()
    )
    def _spectral_filter_scaling(group):
        mode_ids = group.mode_ids()
        order = group.order

        nmodes = len(mode_ids)
        filter_scaling = np.ones(nmodes)
        nfilt = order - cutoff
        if nfilt <= 0:
            return filter

        for mode_index, mode_id in enumerate(mode_ids):
            mode = sum(mode_id)
            if mode >= cutoff:
                filter_scaling[mode_index] = \
                    mode_response_function(mode,
                                           cutoff=cutoff,
                                           nfilt=nfilt)

        return actx.freeze(actx.from_numpy(filter_scaling))

    return _spectral_filter_scaling(group)


def apply_spectral_filter(actx, modal_field, discr, cutoff,
                          mode_response_function):
    r"""Apply the spectral filter, defined by the *mode_response_function*.

    This routine returns filtered data in the modal basis, which has
    been applied using a user-provided *mode_response_function*
    to dampen modes beyond the user-provided *cutoff*.

    Parameters
    ----------
    actx: :class:`arraycontext.ArrayContext`
        A :class:`arraycontext.ArrayContext` associated with
        an array of degrees of freedom
    modal_field: numpy.ndarray
        DOFArray or object array of DOFArrays denoting the modal data
    discr: :class:`meshmode.discretization.Discretization`
        A :class:`meshmode.discretization.Discretization` describing
        the volume discretization the *modal_field* comes from.
    cutoff: int
        Mode cutoff beyond which the filter will be applied, and below which
        the filter will preserve.
    mode_response_function:
        A function that returns a filter weight for for each mode id.

    Returns
    -------
    modal_field: :class:`meshmode.dof_array.DOFArray`
        DOFArray or object array of DOFArrays

    """
    from meshmode.transform_metadata import FirstAxisIsElementsTag
    return DOFArray(
        actx,
        tuple(actx.einsum("j,ej->ej",
                          make_spectral_filter(
                              actx,
                              group=grp,
                              cutoff=cutoff,
                              mode_response_function=mode_response_function
                          ),
                          vec_i,
                          arg_names=("filter", "vec"),
                          tagged=(FirstAxisIsElementsTag(),))
              for grp, vec_i in zip(discr.groups, modal_field))
    )


def filter_modally(dcoll, cutoff, mode_resp_func, field, *, dd=DD_VOLUME_ALL):
    """Stand-alone procedural interface to spectral filtering.

    For each element group in the discretization, and restriction,
    This routine generates:

    * a filter operator:
        - *cutoff* filters only modes above this mode id
        - *mode_resp_func* function returns a filter coefficient
            for a given mode
        - memoized into the array context

    * a filtered solution wherein the filter is applied to *field*.

    Parameters
    ----------
    dcoll: :class:`grudge.discretization.DiscretizationCollection`
        Grudge discretization with boundaries object
    cutoff: int
        Mode below which *field* will not be filtered
    mode_resp_func:
        Modal response function returns a filter coefficient for input mode id
    field: :class:`mirgecom.fluid.ConservedVars`
        An array container containing the relevant field(s) to filter.
    dd: grudge.dof_desc.DOFDesc
        Describe the type of DOF vector on which to operate. Must be on the base
        discretization.

    Returns
    -------
    result: :class:`mirgecom.fluid.ConservedVars`
        An array container containing the filtered field(s).
    """
    if not isinstance(field, DOFArray):
        return map_array_container(
            partial(filter_modally, dcoll, cutoff, mode_resp_func, dd=dd), field
        )

    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    dd_nodal = dd
    dd_modal = dd_nodal.with_discr_tag(DISCR_TAG_MODAL)

    discr = dcoll.discr_from_dd(dd_nodal)

    actx = field.array_context

    modal_map = dcoll.connection_from_dds(dd_nodal, dd_modal)
    nodal_map = dcoll.connection_from_dds(dd_modal, dd_nodal)
    field = modal_map(field)
    field = apply_spectral_filter(actx, field, discr, cutoff,
                                  mode_resp_func)
    return nodal_map(field)
