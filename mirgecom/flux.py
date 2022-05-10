""":mod:`mirgecom.flux` provides generic inter-elemental flux routines.

Low-level interfaces
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: num_flux_lfr
.. autofunction:: num_flux_central
.. autofunction:: num_flux_hll

Flux pair interfaces for operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gradient_flux
.. autofunction:: divergence_flux
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


# These low-level flux functions match the presentation of them in
# the [Toro_2009]_ reference on which they are based.  These arguments
# require no data structure constructs and are presented here as pure
# functions which can easily be tested with plain ole numbers, numpy arrays
# or DOFArrays as appropriate.
#
# {{{ low-level flux interfaces

def num_flux_lfr(f_minus_normal, f_plus_normal, q_minus, q_plus, lam):
    r"""Compute Lax-Friedrichs/Rusanov flux after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:

    .. math::
        f_{\text{LFR}}=\frac{1}{2}\left(f^-+f^++\lambda\left(q^--q^+\right)\right)

    where $f^{\mp}$ and $q^{\mp}$ are the normal flux components and state components
    on the interior and the exterior of the face over which the LFR flux is to be
    calculated. $\lambda$ is the user-supplied dissipation/penalty coefficient.

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux jacobian.

    Parameters
    ----------
    f_minus_normal
        Normal component of physical flux interior to (left of) interface

    f_plus_normal
        Normal component of physical flux exterior to (right of) interface

    q_minus
        Physical state on interior of interface

    q_plus
        Physical state on exterior of interface

    lam: :class:`~meshmode.dof_array.DOFArray`
        State jump penalty parameter for dissipation term

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        Lax-Friedrichs/Rusanov numerical flux.
    """
    return (f_minus_normal + f_plus_normal + lam*(q_minus - q_plus))/2


def num_flux_central(f_minus_normal, f_plus_normal):
    r"""Central low-level numerical flux.

    The central flux is calculated as:

    .. math::
        f_{\text{central}} = \frac{\left(f^++f^-\right)}{2}

    Parameters
    ----------
    f_minus_normal
        Normal component of physical flux interior to (left of) interface

    f_plus_normal
        Normal component of physical flux exterior to (right of) interface

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        central numerical flux.
    """
    return (f_plus_normal + f_minus_normal)/2


def num_flux_hll(f_minus_normal, f_plus_normal, q_minus, q_plus, s_minus, s_plus):
    r"""HLL low-level numerical flux.

    The Harten, Lax, van Leer approximate Riemann numerical flux is calculated as:

    .. math::

        f^{*}_{\text{HLL}} = \frac{\left(s^+f^--s^-f^++s^+s^-\left(q^+-q^-\right)
        \right)}{\left(s^+ - s^-\right)}

    where $f^{\mp}$, $q^{\mp}$, and $s^{\mp}$ are the interface-normal fluxes, the
    states, and the wavespeeds for the interior (-) and exterior (+) of the
    interface, respectively.

    Details about this approximate Riemann solver can be found in Section 10.3 of
    [Toro_2009]_.

    Parameters
    ----------
    f_minus_normal
        Normal component of physical flux interior to (left of) interface

    f_plus_normal
        Normal component of physical flux exterior to (right of) interface

    q_minus
        Physical state on interior of interface

    q_plus
        Physical state on exterior of interface

    q_minus
        Physical state on interior of interface

    q_plus
        Physical state on exterior of interface

    s_minus: :class:`~meshmode.dof_array.DOFArray`
        Interface wave-speed parameter for interior of interface

    s_plus: :class:`~meshmode.dof_array.DOFArray`
        Interface wave-speed parameter for exterior of interface

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        HLL numerical flux.
    """
    actx = q_minus.array_context
    f_star = (s_plus*f_minus_normal - s_minus*f_plus_normal
              + s_plus*s_minus*(q_plus - q_minus))/(s_plus - s_minus)

    # choose the correct f contribution based on the wave speeds
    f_check_minus = \
        actx.np.greater_equal(s_minus, 0*s_minus)*(0*f_minus_normal + 1.0)
    f_check_plus = \
        actx.np.less_equal(s_plus, 0*s_plus)*(0*f_minus_normal + 1.0)

    f = f_star
    f = actx.np.where(f_check_minus, f_minus_normal, f)
    f = actx.np.where(f_check_plus, f_plus_normal, f)

    return f

# }}} low-level flux interfaces


# {{{ Tracepair flux interfaces for operators


def gradient_flux(u_tpair, normal, beta=0):
    r"""Compute a numerical flux for the gradient operator.

    The dissipative central gradient flux, $\mathbf{h}$, of a scalar quantity
    $u$ is calculated as:

    .. math::

        \mathbf{h}({u}^-, {u}^+; \mathbf{n}) = \frac{1}{2}
        \left(({u}^{+}+{u}^{-}) - \beta(u^{+} - u^{-})\right)\mathbf{\hat{n}}

    where ${u}^-, {u}^+$, are the scalar function values on the interior
    and exterior of the face on which the central flux is to be calculated, and
    $\mathbf{\hat{n}}$ is the *normal* vector. Numerical dissipation is added
    by the solution penalized by a factor $\beta$.

    *u_tpair* is the :class:`~grudge.trace_pair.TracePair` representing the scalar
    quantities ${u}^-, {u}^+$. *u_tpair* may also represent a vector-quantity
    :class:`~grudge.trace_pair.TracePair`, and in this case the central scalar flux
    is computed on each component of the vector quantity as an independent scalar.

    Parameters
    ----------
    u_tpair: :class:`~grudge.trace_pair.TracePair`
        Trace pair for the face upon which flux calculation is to be performed
    beta: Number or :class:`~meshmode.dof_array.DOFArray`
        Optional dissipation strength term
    normal: numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
        object of type *u_tpair.int* face-normal directed flux for each component of
        *u_tpair.int*.
    """
    from arraycontext import outer
    return outer(u_tpair.avg + .5*beta*u_tpair.diff, normal)


def divergence_flux(trace_pair, normal, alpha=0, beta=0):
    r"""Compute a numerical flux for the divergence operator.

    The divergence flux, $h$, is calculated as:

    .. math::

        h(\mathbf{v}^-, \mathbf{v}^+; \mathbf{n}) = \frac{1}{2}
        \left(\mathbf{v}^{+}+\mathbf{v}^{-}+\beta(\mathbf{v}^{+}-\mathbf{v}^{-})
        +\alpha\right) \cdot \hat{n}

    where $\mathbf{v}^-, \mathbf{v}^+$, are the vectors on the interior and exterior
    of the face across which the central flux is to be calculated, and $\hat{n}$ is
    the unit normal to the face.  $\alpha$ and $\beta$ are optional terms for adding
    dissipation to the otherwise central flux.

    Parameters
    ----------
    trace_pair: :class:`~grudge.trace_pair.TracePair`
        Trace pair of vector functions *v* for the face upon which flux calculation
        is to be performed
    normal: numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals
    beta: float or :class:`~meshmode.dof_array.DOFArray`
        Dissipation term strength parameter
    alpha: float or :class:`~meshmode.dof_array.DOFArray`
        Dissipation term strength parameter

    Returns
    -------
        object of type *trace_pair.int* face-normal flux for each vector function
        in *trace_pair.int*
    """
    return (trace_pair.avg + beta*trace_pair.diff/2 + alpha)@normal
# }}} Tracepair interafces for operators
