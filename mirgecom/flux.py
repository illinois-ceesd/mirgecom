""":mod:`mirgecom.flux` provides inter-facial flux routines.

Numerical Flux Routines
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gradient_flux_central
.. autofunction:: divergence_flux_central
.. autofunction:: flux_lfr
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


def gradient_flux_central(u_tpair, normal):
    r"""Compute a central flux for the gradient operator.

    The central gradient flux, $\mathbf{h}$, of a scalar quantity $u$ is calculated
    as:

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
    u_tpair: :class:`~grudge.trace_pair.TracePair`
        Trace pair for the face upon which flux calculation is to be performed
    normal: numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with the flux for each
        scalar component.
    """
    from arraycontext import outer
    return outer(u_tpair.avg, normal)


def divergence_flux_central(trace_pair, normal):
    r"""Compute a central flux for the divergence operator.

    The central divergence flux, $h$, is calculated as:

    .. math::

        h(\mathbf{v}^-, \mathbf{v}^+; \mathbf{n}) = \frac{1}{2}
        \left(\mathbf{v}^{+}+\mathbf{v}^{-}\right) \cdot \hat{n}

    where $\mathbf{v}^-, \mathbf{v}^+$, are the vectors on the interior and exterior
    of the face across which the central flux is to be calculated, and $\hat{n}$ is
    the unit normal to the face.

    Parameters
    ----------
    trace_pair: :class:`~grudge.trace_pair.TracePair`
        Trace pair for the face upon which flux calculation is to be performed
    normal: numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with the flux for each
        scalar component.
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

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux Jacobian:

    .. math::
        \lambda = \text{max}\left(|\mathbb{J}_{F}(q^-)|,|\mathbb{J}_{F}(q^+)|\right)

    Here, $\lambda$ is a function parameter, leaving the responsibility for the
    calculation of the eigenvalues of the system-dependent flux Jacobian to the
    caller.

    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`meshmode.dof_array.DOFArray` with outward-pointing
        normals

    lam: :class:`~meshmode.dof_array.DOFArray`

        lambda parameter for Lax-Friedrichs/Rusanov flux

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        Lax-Friedrichs/Rusanov flux.
    """
    return flux_lfr(cv_tpair, f_tpair, normal, lam)@normal


def flux_lfr(cv_tpair, f_tpair, normal, lam):
    r"""Compute Lax-Friedrichs/Rusanov flux after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-) + \mathbf{F}(q^+))
        + \frac{\lambda}{2}(q^{-} - q^{+})\hat{\mathbf{n}},

    where $q^-, q^+$ are the scalar solution components on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the vector flux function, $\hat{\mathbf{n}}$ is the face normal, and $\lambda$
    is the user-supplied jump term coefficient.

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux jacobian:

    .. math::
        \lambda = \text{max}\left(|\mathbb{J}_{F}(q^-)|,|\mathbb{J}_{F}(q^+)|\right)

    Here, $\lambda$ is a function parameter, leaving the responsibility for the
    calculation of the eigenvalues of the system-dependent flux Jacobian to the
    caller.

    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    lam: :class:`~meshmode.dof_array.DOFArray`

        lambda parameter for Lax-Friedrichs/Rusanov flux

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        Lax-Friedrichs/Rusanov flux.
    """
    from arraycontext import outer
    return f_tpair.avg - lam*outer(cv_tpair.diff, normal)/2

# This one knows about fluid stuff
def rusanov_flux(state_tpair, normal):
    actx = state_tpair.int.array_context
    
    # This calculates the local maximum eigenvalue of the flux Jacobian
    # for a single component gas, i.e. the element-local max wavespeed |v| + c.
    w_int = \
        np.abs(np.dot(state_tpair.int.velocity, normal)
               - state_tpair.int.dv.speed_of_sound)
    w_ext = \
        np.abs(np.dot(state_tpair.ext.velocity, normal)
               + state_tpair.ext.dv.speed_of_sound)
    
    # w_int = state_tpair.int.dv.speed_of_sound + state_tpair.int.cv.speed
    # w_ext = state_tpair.ext.dv.speed_of_sound + state_tpair.ext.cv.speed
    lam = actx.np.maximum(w_int, w_ext)
    from grudge.trace_pair import TracePair
    from mirgecom.inviscid import inviscid_flux
    q_tpair = TracePair(state_tpair.dd, interior=state_tpair.int.cv,
                        exterior=state_tpair.ext.cv)
    f_tpair = TracePair(state_tpair.dd, interior=inviscid_flux(state_tpair.int),
                        exterior=inviscid_flux(state_tpair.ext))
    return flux_reconstruction_lfr(q_tpair, f_tpair, lam, normal) 


def flux_reconstruction_lfr(q_tpair, f_tpair, lam, normal):
    """Rusanov if lam=max(wavespeed), LF if lam=(gridspeed)."""
    from arraycontext import outer
    return f_tpair.avg - .5*lam*outer(q_tpair.diff, normal)


def flux_reconstruction_hll(q_tpair, f_tpair, s_tpair, normal):
    r"""Compute Harten-Lax-vanLeer (HLL) flux reconstruction

    The HLL flux is calculated as:

    .. math::

        \mathbf{F}_{\mathtt{HLL}} = \frac{S^+~\mathbf{F}^- - S^-~\mathbf{F}^+
        + S^+~S^-\left(Q^+ - Q^-\right)}{S^+ - S^-}

    where $Q^{\left{-,+\right}}, \mathbf{F}^{\left{-,+\right}}$ are the scalar
    solution components and fluxes on the left(interior) and the right(exterior) of
    the face on which the flux is to be reconstructed.

    Parameters
    ----------
    q_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    s_tpair: :class:`~grudge.trace_pair.TracePair`

        The wavespeeds at the faces for which the numerical flux is to be calculated

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        HLL reconstructed flux
    """
    from arraycontext import outer
    return (s_tpair.ext*f_tpair.int - s_tpair.int*f_tpair.ext
            + s_tpair.int * s_tpair.ext * outer(q_tpair.diff, normal)) / s_tpair.diff
