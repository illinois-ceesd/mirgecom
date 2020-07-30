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
import math
import numpy as np


def get_spectral_filter(dim, order, cutoff, filter_order):
    r"""
    Exponential spectral filter from JSH/TW Nodal DG Methods, pp. 130, 186
    Returns a filter operator in the modal basis designed to apply an
    exponentially decreasing weight to spectral modes beyond the cutoff
    mode.
    """
    npol = 1
    for d in range(1, dim+1):
        npol *= (order + d)
    npol /= math.factorial(int(dim))
    npol = int(npol)
    filter = np.identity(npol)
    alpha = -1.0*np.log(np.finfo(float).eps)
    nfilt = npol - cutoff
    if nfilt <= 0:
        return filter
    nstart = cutoff - 1
    for m in range(nstart, npol):
        filter[m, m] = np.exp(-1.0 * alpha
                              * ((m - nstart) / nfilt) ** filter_order)
    return filter
