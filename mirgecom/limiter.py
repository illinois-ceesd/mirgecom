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

from grudge.dof_desc import DD_VOLUME_ALL


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
    if dd is None:
        dd = DD_VOLUME_ALL

    actx = field.array_context
    cell_vols = abs(op.elementwise_integral(dcoll, dd,
                                            actx.np.zeros_like(field) + 1.0))
    cell_avgs = op.elementwise_integral(dcoll, dd, field)/cell_vols

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
