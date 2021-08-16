""":mod:`mirgecom.flux` provides inter-facial flux routines.

Numerical Flux Routines
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: central_scalar_flux
.. autofunction:: lfr_flux
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
import numpy as np  # noqa
from meshmode.dof_array import DOFArray
from mirgecom.fluid import (
    ConservedVars,
    make_conserved
)


def central_scalar_flux(trace_pair, normal):
    r"""Compute a central scalar flux.

    The central scalar flux, $h$, is calculated as:

    .. math::

        h(\mathbf{u}^-, \mathbf{u}^+; \mathbf{n}) = \frac{1}{2}
        \left(\mathbf{u}^{+}+\mathbf{u}^{-}\right)\hat{n}

    where $\mathbf{u}^-, \mathbf{u}^+$, are the vector of independent scalar
    components and scalar solution components on the interior and exterior of the
    face on which the central flux is to be calculated, and $\hat{n}$ is the normal
    vector.

    Parameters
    ----------
    trace_pair: `grudge.trace_pair.TracePair`
        Trace pair for the face upon which flux calculation is to be performed
    normal: numpy.ndarray
        object array of :class:`meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray
        object array of `meshmode.dof_array.DOFArray` with the central scalar flux
        for each scalar component.
    """
    tp_avg = trace_pair.avg
    tp_join = tp_avg
    if isinstance(tp_avg, DOFArray):
        return tp_avg*normal
    elif isinstance(tp_avg, ConservedVars):
        tp_join = tp_avg.join()

    ncomp = len(tp_join)
    if ncomp > 1:
        result = np.empty((ncomp, len(normal)), dtype=object)
        for i in range(ncomp):
            result[i] = tp_join[i] * normal
    else:
        result = tp_join*normal
    if isinstance(tp_avg, ConservedVars):
        return make_conserved(tp_avg.dim, q=result)
    return result


def lfr_flux(cv_tpair, f_tpair, normal, lam):
    r"""Compute Lax-Friedrichs/Rusanov flux after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-) + \mathbf{F}(q^+)) \cdot
        \hat{n} + \frac{\lambda}{2}(q^{-} - q^{+}),

    where $q^-, q^+$ are the scalar solution components on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the vector flux function, $\hat{n}$ is the face normal, and $\lambda$ is the
    user-supplied jump term coefficient.

    Parameters
    ----------
    cv_tpair: :class:`grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`meshmode.dof_array.DOFArray` with outward-pointing
        normals

    lam: :class:`meshmode.dof_array.DOFArray`

        lambda parameter for Lax-Friedrichs/Rusanov flux

    Returns
    -------
    numpy.ndarray

        object array of :class:`meshmode.dof_array.DOFArray` with the
        Lax-Friedrichs/Rusanov flux.
    """
    return f_tpair.avg@normal - lam*cv_tpair.diff/2
