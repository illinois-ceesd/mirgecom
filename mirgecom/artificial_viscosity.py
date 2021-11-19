r""":mod:`mirgecom.artificial_viscosity` Artificial viscosity for hyperbolic systems.

Consider the following system of conservation laws of the form:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F} = 0,

where $\mathbf{Q}$ is the state vector and $\mathbf{F}$ is the vector of
fluxes. This module applies an artificial viscosity term by augmenting
the governing equations in the following way:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F} =
    \nabla\cdot{\varepsilon\nabla\mathbf{Q}},

where $\varepsilon$ is the artificial viscosity coefficient.
To evalutate the second order derivative numerically, the problem
is recast as a set of first order problems:

.. math::

    \partial_t{\mathbf{Q}} + \nabla\cdot\mathbf{F} &= \nabla\cdot\mathbf{R} \\
    \mathbf{R} &= \varepsilon\nabla\mathbf{Q}

where $\mathbf{R}$ is an auxiliary variable, and the artificial viscosity
coefficient, $\varepsilon$, is spatially dependent and calculated using the
smoothness indicator of [Persson_2012]_:

.. math::

    S_e = \frac{\langle u_{N_p} - u_{N_{p-1}}, u_{N_p} -
        u_{N_{p-1}}\rangle_e}{\langle u_{N_p}, u_{N_p} \rangle_e}

where:

- $S_e$ is the smoothness indicator
- $u_{N_p}$ is the solution in modal basis at the current polynomial order
- $u_{N_{p-1}}$ is the truncated modal represention to the polynomial order $p-1$
- The $L_2$ inner product on an element is denoted $\langle \cdot,\cdot \rangle_e$

The elementwise viscosity is then calculated:

.. math::

    \varepsilon_e =
        \begin{cases}
            0, & s_e < s_0 - \kappa \\
            \frac{1}{2}\left( 1 + \sin \frac{\pi(s_e - s_0)}{2 \kappa} \right ),
            & s_0-\kappa \le s_e \le s_0+\kappa \\
            1, & s_e > s_0+\kappa
        \end{cases}

where:

- $\varepsilon_e$ is the element viscosity
- $s_e = \log_{10}{S_e} \sim 1/p^4$ is the smoothness indicator
- $s_0$ is a reference smoothness value
- $\kappa$ controls the width of the transition between 0 to 1

Smoothness Indicator Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: smoothness_indicator

AV RHS Evaluation
^^^^^^^^^^^^^^^^^

.. autofunction:: av_operator
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

import numpy as np

from pytools import memoize_in, keyed_memoize_in
from pytools.obj_array import obj_array_vectorize

from meshmode.dof_array import thaw, DOFArray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.trace_pair import TracePair, interior_trace_pairs
from grudge.dof_desc import DD_VOLUME_MODAL, DD_VOLUME, DOFDesc


# FIXME: Remove when get_array_container_context is added to meshmode
def _get_actx(obj):
    if isinstance(obj, TracePair):
        return _get_actx(obj.int)
    if isinstance(obj, np.ndarray):
        return _get_actx(obj[0])
    elif isinstance(obj, DOFArray):
        return obj.array_context
    else:
        raise ValueError("Unknown type; can't retrieve array context.")


# Tweak the behavior of np.outer to return a lower-dimensional object if either/both
# of the arguments are scalars (np.outer always returns a matrix)
def _outer(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.outer(a, b)
    else:
        return a*b


def _facial_flux_q(discr, q_tpair):
    """Compute facial flux for each scalar component of Q."""
    actx = _get_actx(q_tpair)
    dd = q_tpair.dd

    normal = thaw(actx, discr.normal(dd))

    # This uses a central scalar flux along nhat:
    # flux = 1/2 * (Q- + Q+) * nhat
    flux_out = _outer(q_tpair.avg, normal)

    return discr.project(dd, dd.with_dtag("all_faces"), flux_out)


def _facial_flux_r(discr, r_tpair):
    """Compute facial flux for vector component of grad(Q)."""
    actx = _get_actx(r_tpair)
    dd = r_tpair.dd

    normal = thaw(actx, discr.normal(dd))

    # This uses a central vector flux along nhat:
    # flux = 1/2 * (grad(Q)- + grad(Q)+) .dot. nhat
    flux_out = r_tpair.avg @ normal

    return discr.project(dd, dd.with_dtag("all_faces"), flux_out)


def av_operator(discr, boundaries, q, alpha, boundary_kwargs=None, **kwargs):
    r"""Compute the artificial viscosity right-hand-side.

    Computes the the right-hand-side term for artificial viscosity.

    .. math::

        \mbox{RHS}_{\mbox{av}} = \nabla\cdot{\varepsilon\nabla\mathbf{Q}}

    Parameters
    ----------
    q: :class:`~meshmode.dof_array.DOFArray` or :class:`~numpy.ndarray`

        Single :class:`~meshmode.dof_array.DOFArray` for a scalar or an object array
        (:class:`~numpy.ndarray`) for a vector of
        :class:`~meshmode.dof_array.DOFArray` on which to operate.

        When used with fluid solvers, *q* is expected to be the fluid state array
        of the canonical conserved variables (mass, energy, momentum)
        for the fluid along with a vector of species masses for multi-component
        fluids.

    boundaries: float

        Dictionary of boundary functions, one for each valid boundary tag

    alpha: float

        The maximum artificial viscosity coefficient to be applied

    Returns
    -------
    numpy.ndarray

        The artificial viscosity operator applied to *q*.
    """
    if boundary_kwargs is None:
        boundary_kwargs = dict()

    quad_tag = kwargs.get("quad_tag", None)
    if quad_tag is None:
        dd = DD_VOLUME
    else:
        dd = DOFDesc("vol", quad_tag)

    dd_allfaces = dd.with_dtag("all_faces")

    kappa = kwargs.get("kappa", 1.0)
    s0 = kwargs.get("s0", -6.0)
    indicator = smoothness_indicator(
        discr,
        # Only applying the smoothness indicator to the "mass" field
        q[0] if isinstance(q, np.ndarray) else q,
        kappa=kappa,
        s0=s0
    )

    def to_quad(a, from_dd, dtag):
        return discr.project(from_dd, dd.with_dtag(dtag), a)

    # Total flux of fluid soln Q across element boundaries
    q_bnd_flux = (
        sum(
            _facial_flux_q(
                discr,
                q_tpair=TracePair(
                    pb_tpair.dd.with_discr_tag(quad_tag),
                    interior=to_quad(pb_tpair.int, pb_tpair.dd, "int_faces"),
                    exterior=to_quad(pb_tpair.ext, pb_tpair.dd, "int_faces")
                )
            ) for pb_tpair in interior_trace_pairs(discr, q)
        )
        # Boundary conditions
        + sum(
            boundaries[btag].soln_gradient_flux(discr, btag, soln=q, **boundary_kwargs)
            for btag in boundaries
        )
    )

    # Compute R
    r = discr.inverse_mass(
        -alpha * indicator * (
            discr.weak_grad(dd, discr.project("vol", dd, q))
            - discr.face_mass(dd_allfaces, q_bnd_flux)
        )
    )

    # Total flux of grad(Q) across element boundaries
    r_bnd_flux = (
        sum(
            _facial_flux_r(
                discr,
                r_tpair=TracePair(
                    pb_tpair.dd.with_discr_tag(quad_tag),
                    interior=to_quad(pb_tpair.int, pb_tpair.dd, "int_faces"),
                    exterior=to_quad(pb_tpair.ext, pb_tpair.dd, "int_faces")
                )
            ) for pb_tpair in interior_trace_pairs(discr, r)
        )
        # Boundary conditions
        + sum(
            boundaries[btag].av_flux(discr, btag, r, **boundary_kwargs)
            for btag in boundaries
        )
    )

    # Return the AV RHS term
    return discr.inverse_mass(
        (-discr.weak_div(dd, discr.project("vol", dd, r))
         + discr.face_mass(dd_allfaces, r_bnd_flux))
    )


def artificial_viscosity(discr, t, eos, boundaries, q, alpha, **kwargs):
    """Interface :function:`av_operator` with backwards-compatible API."""
    from warnings import warn
    warn("Do not call artificial_viscosity; it is now called av_operator. This"
         "function will disappear in 2021", DeprecationWarning, stacklevel=2)
    kwargs["t"] = t
    kwargs["eos"] = eos
    return av_operator(discr, boundaries, q, alpha, **kwargs)


def smoothness_indicator(discr, u, kappa=1.0, s0=-6.0):
    r"""Calculate the smoothness indicator.

    Parameters
    ----------
    u: meshmode.dof_array.DOFArray
        The field that is used to calculate the smoothness indicator.

    kappa
        An optional argument that controls the width of the transition from 0 to 1.
    s0
        An optional argument that sets the smoothness level to limit
        on. Values in the range $(-\infty,0]$ are allowed, where $-\infty$ results in
        all cells being tagged and 0 results in none.

    Returns
    -------
    meshmode.dof_array.DOFArray
        The elementwise constant values between 0 and 1 which indicate the smoothness
        of a given element.
    """
    if not isinstance(u, DOFArray):
        raise ValueError("u argument must be a DOFArray.")

    actx = u.array_context

    @memoize_in(actx, (smoothness_indicator, "smooth_comp_knl"))
    def indicator_prg():
        """Compute the smoothness indicator for all elements."""
        from arraycontext import make_loopy_program
        from meshmode.transform_metadata import (ConcurrentElementInameTag,
                                                 ConcurrentDOFInameTag)
        import loopy as lp
        t_unit = make_loopy_program([
            "{[iel]: 0 <= iel < nelements}",
            "{[idof]: 0 <= idof < ndiscr_nodes_in}",
            "{[jdof]: 0 <= jdof < ndiscr_nodes_in}",
            "{[kdof]: 0 <= kdof < ndiscr_nodes_in}"
            ],
            """
                result[iel,idof] = sum(kdof, vec[iel, kdof]               \
                                             * vec[iel, kdof]             \
                                             * modes_active_flag[kdof]) / \
                                   sum(jdof, vec[iel, jdof]               \
                                             * vec[iel, jdof]             \
                                             + 1.0e-12 / ndiscr_nodes_in)
            """,
            name="smooth_comp",
        )
        return lp.tag_inames(t_unit, {"iel": ConcurrentElementInameTag(),
                                      "idof": ConcurrentDOFInameTag()})

    @keyed_memoize_in(actx, (smoothness_indicator,
                             "highest_mode"),
                      lambda grp: grp.discretization_key())
    def highest_mode(grp):
        return actx.from_numpy(
            np.asarray([1 if sum(mode_id) == grp.order
                        else 0
                        for mode_id in grp.mode_ids()])
        )

    # Convert to modal solution representation
    modal_map = discr.connection_from_dds(DD_VOLUME, DD_VOLUME_MODAL)
    uhat = modal_map(u)

    # Compute smoothness indicator value
    indicator = DOFArray(
        actx,
        data=tuple(
            actx.call_loopy(
                indicator_prg(),
                vec=uhat[grp.index],
                modes_active_flag=highest_mode(grp))["result"]
            for grp in discr.discr_from_dd("vol").groups
        )
    )
    indicator = actx.np.log10(indicator + 1.0e-12)

    # Compute artificial viscosity percentage based on indicator and set parameters
    yesnol = actx.np.greater(indicator, (s0 - kappa))
    yesnou = actx.np.greater(indicator, (s0 + kappa))
    sin_indicator = actx.np.where(
        yesnol,
        0.5 * (1.0 + actx.np.sin(np.pi * (indicator - s0) / (2.0 * kappa))),
        0.0 * indicator,
    )
    indicator = actx.np.where(yesnou, 1.0 + 0.0 * indicator, sin_indicator)

    return indicator
