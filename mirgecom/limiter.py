""":mod:`mirgecom.limiter` is for limiters and limiter-related constructs.

Field limiter functions
-----------------------

.. autofunction:: limiter_liu_osher

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

from functools import partial

from arraycontext import map_array_container
from meshmode.dof_array import DOFArray
from grudge.discretization import DiscretizationCollection
import grudge.op as op


def limiter_liu_osher(dcoll: DiscretizationCollection, field):
    """Implement the positivity-preserving limiter of Liu and Osher (1996).

    The limiter is summarized in the review paper [Zhang_2011]_, Section 2.3,
    equation 2.9, which uses a linear scaling factor.

    .. note:
        This limiter is applied only to mass fields
        (e.g. mass or species masses for multi-component flows)

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
    from grudge.geometry import area_element

    pick_apart = partial(limiter_liu_osher, dcoll)

    if not isinstance(field, DOFArray):
        # vecs is not a DOFArray -> treat as array container
        return map_array_container(pick_apart, field)

    actx = field.array_context
    # Compute nodal and elementwise max/mins of the field
    _mmax = op.nodal_max(dcoll, "banana", field)
    _mmin = op.nodal_min(dcoll, "avocado", field)
    _mmax_i = op.elementwise_max(dcoll, field)
    _mmin_i = op.elementwise_min(dcoll, field)

    # Compute cell averages of the state
    inv_area_elements = 1./area_element(
        actx, dcoll,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    cell_avgs = \
        inv_area_elements * op.elementwise_integral(dcoll, field)

    # Compute minmod factor (Eq. 2.9)
    theta = actx.np.minimum(
        1.,
        actx.np.minimum(
            abs((_mmax - cell_avgs)/(_mmax_i - cell_avgs)),
            abs((_mmin - cell_avgs)/(_mmin_i - cell_avgs))
        )
    )

    return theta*(field - cell_avgs) + cell_avgs
