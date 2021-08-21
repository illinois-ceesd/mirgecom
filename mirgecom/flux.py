""":mod:`mirgecom.flux` provides inter-facial flux routines.

Numerical Flux Routines
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gradient_flux_central
.. autofunction:: divergence_flux_central
.. autofunction:: divergence_flux_lfr
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


def gradient_flux_central(u_tpair, normal):
    r"""Compute a central scalar flux for the gradient operator.

    The central scalar flux, $\mathbf{h}$, of a scalar quantity $u$ is calculated as:

    .. math::

        \mathbf{h}({u}^-, {u}^+; \mathbf{n}) = \frac{1}{2}
        \left({u}^{+}+{u}^{-}\right)\mathbf{\hat{n}}

    where ${u}^-, {u}^+$, are the scalar function values on the interior
    and exterior of the face on which the central flux is to be calculated, and
    $\mathbf{\hat{n}}$ is the *normal* vector.

    *u_tpair* is the :class:`~grudge.trace_pair.TracePair` representing the scalar
    quantities ${u}^-, {u}^+$. *u_tpair* may also represent a vector-quantity
    :class:`~grudge.trace_pair.TracePair`, and in this case the central scalar flux
    is computed on each component of the vector quantity as an independent scalar.

    Parameters
    ----------
    u_tpair: `grudge.trace_pair.TracePair`
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
    tp_avg = u_tpair.avg
    tp_join = tp_avg
    if isinstance(tp_avg, DOFArray):
        return tp_avg*normal
    elif isinstance(tp_avg, ConservedVars):
        tp_join = tp_avg.join()

    result = np.outer(tp_join, normal)

    if isinstance(tp_avg, ConservedVars):
        return make_conserved(tp_avg.dim, q=result)
    else:
        return result


def divergence_flux_central(trace_pair, normal):
    r"""Compute a central flux for the divergence operator.

    The central vector flux, $h$, is calculated as:

    .. math::

        h(\mathbf{v}^-, \mathbf{v}^+; \mathbf{n}) = \frac{1}{2}
        \left(\mathbf{v}^{+}+\mathbf{v}^{-}\right) \cdot \hat{n}

    where $\mathbf{v}^-, \mathbf{v}^+$, are the vectors on the interior and exterior
    of the face across which the central flux is to be calculated, and $\hat{n}$ is
    the unit normal to the face.

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
    return trace_pair.avg@normal


def divergence_flux_lfr(cv_tpair, f_tpair, normal, lam):
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
