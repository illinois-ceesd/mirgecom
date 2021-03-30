""":mod:`mirgecom.flux` provides inter-facial flux routines.

Numerical Flux Routines
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: lfr_flux
"""

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
import numpy as np  # noqa
from pytools.obj_array import make_obj_array


def central_scalar_flux(trace_pair, normal):
    r"""Compute a central scalar flux.

    The central scalar flux, $h$, is calculated as:

    .. math::

        h(\mathbf{u}^-, \mathbf{u}^+; \mathbf{n}) = \frac{1}{2}
        \left(\mathbf{u}^{+}+\mathbf{u}^{-}\right)\hat{n}

    where $\mathbf{u}^-, \matbhf{u}^+$, are the vector of independent scalar
    components and scalar solution components on the interior and exterior of the
    face on which the central flux is to be calculated, and $\hat{n}$ is the normal
    vector.

    Parameters
    ----------
    trace_pair: `grudge.sym.TracePair`
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
    ncomp = 1
    if isinstance(tp_avg, np.ndarray):
        ncomp = len(tp_avg)
    if ncomp > 1:
        return make_obj_array([tp_avg[i]*normal for i in range(ncomp)])
    return trace_pair.avg*normal


def lfr_flux(q_tpair, compute_flux, normal, lam):
    r"""Compute Lax-Friedrichs/Rusanov flux after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{f}^{+} + \mathbf{f}^{-}) \cdot
        \hat{n} + \frac{\lambda}{2}(q^{-} - q^{+}),

    where $f^-, f^+$, and $q^-, q^+$ are the fluxes and scalar solution components on
    the interior and the exterior of the face on which the LFR flux is to be
    calculated. The The face normal is $\hat{n}$, and $\lambda$ is the user-supplied
    jump term coefficient.

    Parameters
    ----------
    q_tpair:

        Trace pair (grudge.symbolic.TracePair) for the face upon which flux
        calculation is to be performed

    compute_flux:

        function should return ambient dim-vector fluxes given *q* values

    normal: numpy.ndarray

        object array of :class:`meshmode.dof_array.DOFArray` with outward-pointing
        normals

    lam: :class:`meshmode.dof_array.DOFArray`

        lambda parameter for Lax-Friedrichs/Rusanov flux

    Returns
    -------
    numpy.ndarray

        object array of meshmode.dof_array.DOFArray with the Lax-Friedrichs/Rusanov
        flux.
    """
    flux_avg = 0.5*(compute_flux(q_tpair.int)
                    + compute_flux(q_tpair.ext))
    return flux_avg @ normal - 0.5*lam*(q_tpair.ext - q_tpair.int)
