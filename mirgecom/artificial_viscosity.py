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

.. autofunction:: av_laplacian_operator
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
from dataclasses import replace

from pytools import memoize_in, keyed_memoize_in

from meshmode.dof_array import thaw, DOFArray
from meshmode.discretization.connection import FACE_RESTR_ALL

from mirgecom.flux import gradient_flux_central, divergence_flux_central
from mirgecom.operators import div_operator, grad_operator

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs
)
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    DISCR_TAG_MODAL,
    as_dofdesc,
)

import grudge.op as op


class _AVCVTag:
    pass


class _AVRTag:
    pass


def av_laplacian_operator(discr, boundaries, fluid_state, alpha,
                          boundary_kwargs=None, quadrature_tag=DISCR_TAG_BASE,
                          volume_dd=DD_VOLUME_ALL, **kwargs):
    r"""Compute the artificial viscosity right-hand-side.

    Computes the the right-hand-side term for artificial viscosity.

    .. math::

        \mbox{RHS}_{\mbox{av}} = \nabla\cdot{\varepsilon\nabla\mathbf{Q}}

    Parameters
    ----------
    fluid_state: :class:`mirgecom.gas_model.FluidState`
        Fluid state object with the conserved and thermal state.

    boundaries: float
        Dictionary of boundary functions, one for each valid boundary tag

    alpha: float
        The maximum artificial viscosity coefficient to be applied

    boundary_kwargs: :class:`dict`
        dictionary of extra arguments to pass through to the boundary conditions

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.

    volume_dd: grudge.dof_desc.DOFDesc
        The DOF descriptor of the volume on which to apply the operator.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`
        The artificial viscosity operator applied to *q*.
    """
    if boundary_kwargs is None:
        boundary_kwargs = dict()

    boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in boundaries.items()}

    cv = fluid_state.cv
    actx = cv.array_context

    dd_base = volume_dd
    dd_vol = dd_base.with_discr_tag(quadrature_tag)
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)

    def interp_to_vol_quad(u):
        return op.project(discr, dd_base, dd_vol, u)

    def interp_to_surf_quad(utpair):
        dd_trace = utpair.dd
        dd_trace_quad = dd_trace.with_discr_tag(quadrature_tag)
        return TracePair(
            dd_trace_quad,
            interior=op.project(discr, dd_trace, dd_trace_quad, utpair.int),
            exterior=op.project(discr, dd_trace, dd_trace_quad, utpair.ext)
        )

    # Get smoothness indicator based on mass component
    kappa = kwargs.get("kappa", 1.0)
    s0 = kwargs.get("s0", -6.0)
    indicator = smoothness_indicator(
        discr, cv.mass, kappa=kappa, s0=s0, volume_dd=dd_vol)

    def central_flux(utpair):
        dd_trace = utpair.dd
        dd_allfaces = dd_trace.with_domain_tag(
            replace(dd_trace.domain_tag, tag=FACE_RESTR_ALL))
        normal = thaw(actx, discr.normal(dd_trace))
        return op.project(discr, dd_trace, dd_allfaces,
                          # This uses a central scalar flux along nhat:
                          # flux = 1/2 * (Q- + Q+) * nhat
                          gradient_flux_central(utpair, normal))

    cv_bnd = (
        # Rank-local and cross-rank (across parallel partitions) contributions
        sum(
            central_flux(interp_to_surf_quad(tpair))
            for tpair in interior_trace_pairs(
                discr, cv, volume_dd=dd_base, tag=_AVCVTag))
        # Contributions from boundary fluxes
        + sum(
            bdry.soln_gradient_flux(
                discr,
                dd_vol=dd_vol,
                dd_bdry=dd_vol.with_domain_tag(bdtag),
                fluid_state=fluid_state, **boundary_kwargs)
            for bdtag, bdry in boundaries.items())
    )

    # Compute R = alpha*grad(Q)
    r = -alpha * indicator \
        * grad_operator(
            discr, dd_vol, dd_allfaces, interp_to_vol_quad(cv), cv_bnd)

    def central_flux_div(utpair):
        dd_trace = utpair.dd
        dd_allfaces = dd_trace.with_domain_tag(
            replace(dd_trace.domain_tag, tag=FACE_RESTR_ALL))
        normal = thaw(actx, discr.normal(dd_trace))
        return op.project(discr, dd_trace, dd_allfaces,
                          # This uses a central vector flux along nhat:
                          # flux = 1/2 * (grad(Q)- + grad(Q)+) .dot. nhat
                          divergence_flux_central(utpair, normal))

    # Total flux of grad(Q) across element boundaries
    r_bnd = (
        # Rank-local and cross-rank (across parallel partitions) contributions
        sum(
            central_flux_div(interp_to_surf_quad(tpair))
            for tpair in interior_trace_pairs(
                discr, r, volume_dd=dd_base, tag=_AVRTag))
        # Contributions from boundary fluxes
        + sum(
            bdry.av_flux(
                discr,
                dd_vol=dd_vol,
                dd_bdry=dd_vol.with_domain_tag(bdtag),
                diffusion=r, **boundary_kwargs)
            for bdtag, bdry in boundaries.items())
    )

    # Return the AV RHS term
    return -div_operator(discr, dd_vol, dd_allfaces, interp_to_vol_quad(r), r_bnd)


def smoothness_indicator(discr, u, kappa=1.0, s0=-6.0, volume_dd=DD_VOLUME_ALL):
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

    @memoize_in(actx, (smoothness_indicator, "smooth_comp_knl", volume_dd))
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
                             "highest_mode", volume_dd),
                      lambda grp: grp.discretization_key())
    def highest_mode(grp):
        return actx.from_numpy(
            np.asarray([1 if sum(mode_id) == grp.order
                        else 0
                        for mode_id in grp.mode_ids()])
        )

    # Convert to modal solution representation
    dd_vol = volume_dd
    dd_modal = dd_vol.with_discr_tag(DISCR_TAG_MODAL)
    modal_map = discr.connection_from_dds(dd_vol, dd_modal)
    uhat = modal_map(u)

    # Compute smoothness indicator value
    indicator = DOFArray(
        actx,
        data=tuple(
            actx.call_loopy(
                indicator_prg(),
                vec=uhat[grp.index],
                modes_active_flag=highest_mode(grp))["result"]
            for grp in discr.discr_from_dd(dd_vol).groups
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
