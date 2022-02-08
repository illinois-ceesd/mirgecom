""":mod:`mirgecom.flux` provides generic inter-elemental flux routines.

Low-level interfaces
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: num_flux_lfr
.. autofunction:: num_flux_central
.. autofunction:: num_flux_hll

Flux pair interfaces for operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gradient_flux_central
.. autofunction:: divergence_flux_central
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


def num_flux_lfr(f_minus, f_plus, q_minus, q_plus, lam, **kwargs):
    """Lax-Friedrichs/Rusanov low-level numerical flux."""
    return (f_minus + f_plus - lam*(q_plus - q_minus))/2


def num_flux_central(f_minus, f_plus, **kwargs):
    """Central low-level numerical flux."""
    return (f_plus + f_minus)/2


def num_flux_hll(f_minus, f_plus, q_minus, q_plus, s_minus, s_plus):
    r"""HLL low-level numerical flux.

    The Harten, Lax, van Leer approximate riemann numerical flux is calculated as:

    .. math::

        f^{*}_{\mathtt{HLL}} = \frac{\left(s^+f^--s^-f^++s^+s^-\left(q^+-q^-\right)
        \right)}{\left(s^+ - s^-\right)}

    where $f^{\mp}$, $q^{\mp}$, and $s^{\mp}$ are the interface-normal fluxes, the
    states, and the wavespeeds for the interior (-) and exterior (+) of the
    interface, respectively.
    """
    actx = q_minus.array_context
    f_star = (s_plus*f_minus - s_minus*f_plus
              + s_plus*s_minus*(q_plus - q_minus))/(s_plus - s_minus)

    # choose the correct f contribution based on the wave speeds
    f_check_minus = actx.np.greater_equal(s_minus, 0*s_minus)*(0*f_minus + 1.0)
    f_check_plus = actx.np.less_equal(s_plus, 0*s_plus)*(0*f_minus + 1.0)

    f = f_star
    f = actx.np.where(f_check_minus, f_minus, f)
    f = actx.np.where(f_check_plus, f_plus, f)

    return f


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
