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

where $\mathbf{R}$ is an auxiliary variable, and the artitifial viscosity
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

The elementwise viscoisty is then calculated:

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
from pytools.obj_array import obj_array_vectorize
from meshmode.dof_array import thaw, DOFArray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair
from grudge.dof_desc import DD_VOLUME_MODAL, DD_VOLUME


def _facial_flux_q(discr, q_tpair):
    """Compute facial flux for each scalar component of Q."""
    flux_dis = q_tpair.avg
    if isinstance(flux_dis, np.ndarray):
        actx = flux_dis[0].array_context
        flux_dis = flux_dis.reshape(-1, 1)
    else:
        actx = flux_dis.array_context

    normal = thaw(actx, discr.normal(q_tpair.dd))

    # This uses a central scalar flux along nhat:
    # flux = 1/2 * (Q- + Q+) * nhat
    flux_out = flux_dis * normal

    return discr.project(q_tpair.dd, "all_faces", flux_out)


def _facial_flux_r(discr, r_tpair):
    """Compute facial flux for vector component of grad(Q)."""
    flux_dis = r_tpair.avg
    if isinstance(flux_dis[0], np.ndarray):
        actx = flux_dis[0][0].array_context
    else:
        actx = flux_dis[0].array_context

    normal = thaw(actx, discr.normal(r_tpair.dd))

    # This uses a central vector flux along nhat:
    # flux = 1/2 * (grad(Q)- + grad(Q)+) .dot. nhat
    flux_out = flux_dis @ normal

    return discr.project(r_tpair.dd, "all_faces", flux_out)


def av_operator(discr, t, eos, boundaries, q, alpha, **kwargs):
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

    t: float

        Time

    alpha: float

       The maximum artificial viscosity coefficient to be applied

    eos: :class:`~mirgecom.eos.GasEOS`

       Only used as a pass through to the boundary conditions.

    Returns
    -------
    numpy.ndarray

        The artificial viscosity operator applied to *q*.
    """
    # Get smoothness indicator based on first component
    indicator_field = q[0] if isinstance(q, np.ndarray) else q
    indicator = smoothness_indicator(discr, indicator_field, **kwargs)

    # R=Grad(Q) volume part
    if isinstance(q, np.ndarray):
        grad_q_vol = np.stack(obj_array_vectorize(discr.weak_grad, q), axis=0)
    else:
        grad_q_vol = discr.weak_grad(q)

    # R=Grad(Q) Q flux over interior faces
    q_flux_int = _facial_flux_q(discr, q_tpair=interior_trace_pair(discr, q))
    # R=Grad(Q) Q flux interior faces on partition boundaries
    q_flux_pb = sum(_facial_flux_q(discr, q_tpair=pb_tpair)
                    for pb_tpair in cross_rank_trace_pairs(discr, q))
    # R=Grad(Q) Q flux domain boundary part (i.e. BCs)
    q_flux_db = 0
    for btag in boundaries:
        q_tpair = TracePair(
            btag,
            interior=discr.project("vol", btag, q),
            exterior=boundaries[btag].exterior_q(discr, btag=btag, t=t,
                                                 q=q, eos=eos))
        q_flux_db = q_flux_db + _facial_flux_q(discr, q_tpair=q_tpair)
    # Total Q flux across element boundaries
    q_bnd_flux = q_flux_int + q_flux_pb + q_flux_db

    # Compute R
    r = discr.inverse_mass(
        -alpha * indicator * (grad_q_vol - discr.face_mass(q_bnd_flux))
    )

    # RHS_av = div(R) volume part
    div_r_vol = discr.weak_div(r)
    # RHS_av = div(R): grad(Q) flux interior faces part
    r_flux_int = _facial_flux_r(discr, r_tpair=interior_trace_pair(discr, r))
    # RHS_av = div(R): grad(Q) flux interior faces on the partition boundaries
    r_flux_pb = sum(_facial_flux_r(discr, r_tpair=pb_tpair)
                    for pb_tpair in cross_rank_trace_pairs(discr, r))
    # RHS_av = div(R): grad(Q) flux domain boundary part (BCs)
    r_flux_db = 0
    for btag in boundaries:
        r_tpair = TracePair(
            btag,
            interior=discr.project("vol", btag, r),
            exterior=boundaries[btag].exterior_grad_q(discr, btag=btag, t=t,
                                                      grad_q=r, eos=eos))
        r_flux_db = r_flux_db + _facial_flux_r(discr, r_tpair=r_tpair)
    # Total grad(Q) flux element boundaries
    r_flux_bnd = r_flux_int + r_flux_pb + r_flux_db

    # Return the AV RHS term
    return discr.inverse_mass(-div_r_vol + discr.face_mass(r_flux_bnd))


def artificial_viscosity(discr, t, eos, boundaries, q, alpha, **kwargs):
    """Interface :function:`av_operator` with backwards-compatible API."""
    from warnings import warn
    warn("Do not call artificial_viscosity; it is now called av_operator. This"
         "function will disappear in 2021", DeprecationWarning, stacklevel=2)
    return av_operator(discr=discr, eos=eos, boundaries=boundaries,
                       q=q, alpha=alpha, t=t)


def compute_smoothness_indicator():
    """Compute the smoothness indicator for all elements."""
    from meshmode.array_context import make_loopy_program

    knl = make_loopy_program(
        """{[iel,idof,j,k]:
        0<=iel<nelements and
        0<=idof<ndiscr_nodes_out and
        0<=j<ndiscr_nodes_in and
        0<=k<ndiscr_nodes_in}""",
        "result[iel,idof] = "
        "sum(k,vec[iel,k]*vec[iel,k]*modes[k])/sum(j,"
        " vec[iel,j]*vec[iel,j]+1.0e-12/ndiscr_nodes_in)",
        name="smooth_comp",
    )
    return knl


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
    assert isinstance(u, DOFArray)

    def get_indicator():
        return compute_smoothness_indicator()

    # Convert to modal solution representation
    modal_map = discr.connection_from_dds(DD_VOLUME, DD_VOLUME_MODAL)
    uhat = modal_map(u)

    # Compute smoothness indicator value
    actx = u.array_context
    indicator = discr.empty(actx, dtype=u.entry_dtype)
    for group in discr.discr_from_dd("vol").groups:
        mode_ids = group.mode_ids()
        modes = len(mode_ids)
        order = group.order
        highest_mode = np.ones(modes)
        for mode_index, mode_id in enumerate(mode_ids):
            highest_mode[mode_index] = highest_mode[mode_index] * (
                sum(mode_id) == order
            )

        actx.call_loopy(
            get_indicator(),
            result=indicator[group.index],
            vec=uhat[group.index],
            modes=actx.from_numpy(highest_mode),
        )
    indicator = actx.np.log10(indicator + 1.0e-12)

    # Compute artificial viscosity percentage based on indicator and set parameters
    yesnol = indicator > (s0 - kappa)
    yesnou = indicator > (s0 + kappa)
    sin_indicator = actx.np.where(
        yesnol,
        0.5 * (1.0 + actx.np.sin(np.pi * (indicator - s0) / (2.0 * kappa))),
        0.0 * indicator,
    )
    indicator = actx.np.where(yesnou, 1.0 + 0.0 * indicator, sin_indicator)

    return indicator
