""":mod:`mirgecom.limiter` is for limiters and limiter-related constructs.

Field limiter functions
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: positivity_preserving_limiter


Helper Functions
^^^^^^^^^^^^^^^^

.. autofunction:: cell_volume
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


def cell_volume(actx, dcoll: DiscretizationCollection):
    """Evaluate cell area or volume."""
    zeros = actx.thaw(actx.freeze(dcoll.nodes()))[0]
    return op.elementwise_integral(dcoll, zeros + 1.0)


def positivity_preserving_limiter(dcoll: DiscretizationCollection, volume, field,
                                  mmin=0.0, mmax=1.0):
    """Implement the positivity-preserving limiter of Liu and Osher (1996)."""
    actx = field.array_context

    # Compute cell averages of the state
    cell_avgs = 1.0/volume*op.elementwise_integral(dcoll, field)

    # Without also enforcing the averaged to be bounded, the limiter may fail
    # since we work with a posteriori correction of the values. This operation
    # is not described in the paper but greatly increased the robustness after
    # some numerical exercises with this function.
    # This will not make the limiter conservative but it is better than having
    # negative species. This should only be necessary for coarse grids or
    # underresolved regions... If it is knowingly underresolved, then I think
    # we can abstain to ensure "exact" conservation.
    cell_avgs = actx.np.where(actx.np.greater(cell_avgs, mmin), cell_avgs, mmin)
    cell_avgs = actx.np.where(actx.np.greater(cell_avgs, mmax), mmax, cell_avgs)

    # Compute nodal and elementwise max/mins of the field
    mmin_i = op.elementwise_min(dcoll, field)
    mmax_i = op.elementwise_max(dcoll, field)

    _theta = actx.np.minimum(
        1.0, actx.np.minimum(
            actx.np.where(actx.np.less(mmin_i, mmin),
                     abs((mmin-cell_avgs)/(mmin_i-cell_avgs+1e-13)), 1.0),
            actx.np.where(actx.np.greater(mmax_i, mmax),
                     abs((mmax-cell_avgs)/(mmax_i-cell_avgs+1e-13)), 1.0)
            )
        )

    return _theta*(field - cell_avgs) + cell_avgs
