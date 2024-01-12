"""
:mod:`mirgecom.limiter` is for limiters and limiter-related constructs.

Field limiter functions
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: bound_preserving_limiter

"""

__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
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

from grudge.discretization import DiscretizationCollection
import grudge.op as op

from grudge.dof_desc import DD_VOLUME_ALL, DISCR_TAG_MODAL

import numpy as np
from meshmode.transform_metadata import FirstAxisIsElementsTag
from meshmode.dof_array import DOFArray

from functools import partial

def bound_preserving_limiter(dcoll: DiscretizationCollection, field,
                             mmin=0.0, mmax=None, modify_average=False,
                             dd=DD_VOLUME_ALL):
    r"""Implement a slope limiter for bound-preserving properties.

    The implementation is summarized in [Zhang_2011]_, Sec. 2.3, Eq. 2.9,
    which uses a linear scaling factor

    .. math::

        \theta = \min\left( \left| \frac{M - \bar{u}_j}{M_j - \bar{u}_j} \right|,
                            \left| \frac{m - \bar{u}_j}{m_j - \bar{u}_j} \right|,
                           1 \right)

    to limit the high-order polynomials

    .. math::

        \tilde{p}_j = \theta (p_j - \bar{u}_j) + \bar{u}_j

    The lower and upper bounds are given by $m$ and $M$, respectively, and can be
    specified by the user. By default, no limiting is performed to the upper bound.

    The scheme is conservative since the cell average $\bar{u}_j$ is not
    modified in this operation. However, a boolean argument can be invoked to
    modify the cell average. Negative values may appear when changing polynomial
    order during execution or any extra interpolation (i.e., changing grids).
    If these negative values remain during the temporal integration, the current
    slope limiter will fail to ensure positive values.

    Parameters
    ----------
    dcoll: :class:`grudge.discretization.DiscretizationCollection`
        Grudge discretization with boundaries object
    field: meshmode.dof_array.DOFArray or numpy.ndarray
        A field to limit
    mmin: float
        Optional float with the target lower bound. Default to 0.0.
    mmax: float
        Optional float with the target upper bound. Default to None.
    modify_average: bool
        Flag to avoid modification the cell average. Defaults to False.
    dd: grudge.dof_desc.DOFDesc
        The DOF descriptor corresponding to *field*.

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        An array container containing the limited field(s).
    """
    actx = field.array_context

    # Compute cell averages of the state
    def cancel_polynomials(grp):
        return actx.from_numpy(np.asarray([1 if sum(mode_id) == 0
                                           else 0 for mode_id in grp.mode_ids()]))

    # map from nodal to modal
    if dd is None:
        dd = DD_VOLUME_ALL

    dd_nodal = dd
    dd_modal = dd_nodal.with_discr_tag(DISCR_TAG_MODAL)

    modal_map = dcoll.connection_from_dds(dd_nodal, dd_modal)
    nodal_map = dcoll.connection_from_dds(dd_modal, dd_nodal)

    modal_discr = dcoll.discr_from_dd(dd_modal)
    modal_field = modal_map(field)

    # cancel the ``high-order'' polynomials p > 0, and only the average remains
    filtered_modal_field = DOFArray(
        actx,
        tuple(actx.einsum("ej,j->ej",
                          vec_i,
                          cancel_polynomials(grp),
                          arg_names=("vec", "filter"),
                          tagged=(FirstAxisIsElementsTag(),))
              for grp, vec_i in zip(modal_discr.groups, modal_field))
    )

    # convert back to nodal to have the average at all points
    cell_avgs = nodal_map(filtered_modal_field)

    # Bound cell average in case it doesn't respect the realizability
    if modify_average:
        cell_avgs = actx.np.where(actx.np.greater(cell_avgs, mmin), cell_avgs, mmin)

    # Compute elementwise max/mins of the field
    mmin_i = op.elementwise_min(dcoll, dd, field)

    # Linear scaling of polynomial coefficients
    _theta = actx.np.minimum(
        1.0, actx.np.where(actx.np.less(mmin_i, mmin),
                           abs((mmin-cell_avgs)/(mmin_i-cell_avgs+1e-13)),
                           1.0)
        )

    if mmax is not None:
        if modify_average:
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, mmax),
                                      mmax, cell_avgs)

        mmax_i = op.elementwise_max(dcoll, dd, field)

        _theta = actx.np.minimum(
            _theta, actx.np.where(actx.np.greater(mmax_i, mmax),
                                  abs((mmax-cell_avgs)/(mmax_i-cell_avgs+1e-13)),
                                  1.0)
        )

    return _theta*(field - cell_avgs) + cell_avgs

from arraycontext import map_array_container
from arraycontext import thaw, freeze

def neighbor_list(dim, mesh):

    centroids = np.empty(
            (mesh.ambient_dim, mesh.nelements),
            dtype=mesh.vertices.dtype)

    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups):
        centroids[:, base_element_nr:base_element_nr + grp.nelements] = (
                np.sum(mesh.vertices[:, grp.vertex_indices], axis=-1)
                / grp.vertex_indices.shape[-1])

    adj = mesh.facial_adjacency_groups[0]
    nconnections = (adj[0].elements).shape[0]
    connections = np.empty((nconnections, 2), dtype=np.int32)
    connections[:,0] = np.sort(adj[0].elements)
    connections[:,1] = adj[0].neighbors[np.argsort(adj[0].elements)]

    #print(nconnections)

    neighbors = np.zeros((mesh.nelements,dim+2),dtype=np.int32)
    ii = 0
    for kk in range(0,mesh.nelements):
        neighbors[kk,:] = kk
        idx = 0
        while connections[ii,0] == kk:
            idx = idx + 1
            neighbors[kk,idx] = connections[ii,1]
            #print(ii, kk, idx, connections[ii,:])
            ii = ii + 1

            if ii == nconnections:
                break

    return neighbors

def limiter_liu_osher(dcoll: DiscretizationCollection, neig, field, vizdata=False):
    """.

    Parameters
    ----------
    dcoll: :class:`grudge.discretization.DiscretizationCollection`
        Grudge discretization with boundaries object
    field: meshmode.dof_array.DOFArray or numpy.ndarray
        A field or collection of scalar fields to limit
    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        An array container containing the limited field(s).
    """

    actx = field.array_context

    volume = op.elementwise_integral(dcoll, field*0.0 + 1.0)

    # Compute cell averages of the state
    cell_avgs = 1.0/volume*op.elementwise_integral(dcoll, field)
    avgs = actx.to_numpy(cell_avgs[0])[:,0]

    # Compute nodal and elementwise max/mins of the field
    mmax_i = actx.to_numpy(op.elementwise_max(dcoll, field)[0])[:,0]
    mmin_i = actx.to_numpy(op.elementwise_min(dcoll, field)[0])[:,0]
    
    # Compute minmod factor (Eq. 2.9)
    nneighbors = neig.shape[1]

    mmax = avgs[neig[:,1]]
    for i in range(2,nneighbors):
        mmax = np.maximum( mmax, avgs[neig[:,i]] )

    mmin = avgs[neig[:,1]]
    for i in range(2,nneighbors):
        mmin = np.minimum( mmin, avgs[neig[:,i]] )

#    mmax = np.maximum( avgs[neig[:,0]], avgs[neig[:,1]] )
#    for i in range(2,nneighbors):
#        mmax = np.maximum( mmax, avgs[neig[:,i]] )

#    mmin = np.minimum( avgs[neig[:,0]], avgs[neig[:,1]] )
#    for i in range(2,nneighbors):
#        mmin = np.minimum( mmin, avgs[neig[:,i]] )

#    #mmax = np.maximum( mmax_i[neig[:,0]], mmax_i[neig[:,1]] )
#    mmax = mmax_i[neig[:,1]]
#    for i in range(2,nneighbors):
#        mmax = np.maximum( mmax, mmax_i[neig[:,i]] )
#
#    #mmin = np.minimum( mmin_i[neig[:,0]], mmin_i[neig[:,1]] )
#    mmin = mmin_i[neig[:,1]]
#    for i in range(2,nneighbors):
#        mmin = np.minimum( mmin, mmin_i[neig[:,i]] )

    denom_max = np.where(np.abs(mmax_i-avgs) < 1e-14, 1.0, mmax_i-avgs)
    denom_min = np.where(np.abs(mmin_i-avgs) < 1e-14, 1.0, mmin_i-avgs)

    _theta = np.minimum(
                1., np.minimum(
                abs( (mmax-avgs)/(denom_max) ),
                abs( (mmin-avgs)/(denom_min) ) )
             )

    # Transform back to array context
    #FIXME apparently there is a broadcast operation
    dummy = np.zeros(cell_avgs[0].shape)       
    for i in range(0,cell_avgs[0].shape[-1]):
      dummy[:,i] = _theta[:]

    theta = DOFArray(actx, data=(actx.from_numpy(np.array(dummy)), ))
  
    minRatio = (mmin-avgs)/denom_min
    dummy = np.zeros(cell_avgs[0].shape)       
    for i in range(0,cell_avgs[0].shape[-1]):
      dummy[:,i] = minRatio[:]
    minRatio = DOFArray(actx, data=(actx.from_numpy(np.array(dummy)), ))

    maxRatio = (mmax-avgs)/denom_max
    dummy = np.zeros(cell_avgs[0].shape)       
    for i in range(0,cell_avgs[0].shape[-1]):
      dummy[:,i] = maxRatio[:]
    maxRatio = DOFArray(actx, data=(actx.from_numpy(np.array(dummy)), ))

    if vizdata:
        return theta*(field - cell_avgs) + cell_avgs, theta, cell_avgs, minRatio, maxRatio
    return theta*(field - cell_avgs) + cell_avgs
