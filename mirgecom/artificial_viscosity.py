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


Boundary Conditions
^^^^^^^^^^^^^^^^^^^

The artificial viscosity operator as currently implemented re-uses the fluid
solution gradient $\nabla{\mathbf{Q}}$ for the auxiliary equation:

.. math::

    \mathbf{R} = \varepsilon\nabla\mathbf{Q}_\text{fluid}

As such, the fluid-system imposes the appropriate boundary solution $\mathbf{Q}^+$
for the comptuation of $\nabla{\mathbf{Q}}$.  This approach leaves the boundary
condition on $\mathbf{R}$ to be imposed by boundary treatment for the operator when
computing the divergence for the RHS, $\nabla \cdot \mathbf{R}$.

Similar to the fluid boundary treatments; when no boundary conditions are imposed
on $\mathbf{R}$, the interior solution is simply extrapolated to the boundary,
(i.e., $\mathbf{R}^+ = \mathbf{R}^-$).  If such a boundary condition is imposed,
usually for selected components of $\mathbf{R}$, then such boundary conditions
are used directly:  $\mathbf{R}^+ = \mathbf{R}_\text{bc}$.

A central numerical flux is then employed to transmit the boundary condition to
the domain for the divergence operator:

.. math::

    \mathbf{R} \cdot \hat{mathbf{n}} = \frac{1}{2}\left(\mathbf{R}^-
    + \mathbf{R}^+\right) \cdot \hat{\mathbf{n}}


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

from pytools import memoize_in, keyed_memoize_in
from functools import partial
from meshmode.dof_array import DOFArray
from meshmode.discretization.connection import FACE_RESTR_ALL

from mirgecom.flux import num_flux_central
from mirgecom.operators import div_operator

from grudge.trace_pair import (
    interior_trace_pairs,
    tracepair_with_discr_tag
)

from grudge.dof_desc import (
    DD_VOLUME_ALL,
    VolumeDomainTag,
    DISCR_TAG_BASE,
    DISCR_TAG_MODAL,
)

from mirgecom.utils import normalize_boundaries
from arraycontext import get_container_context_recursively
from grudge.dof_desc import as_dofdesc
from grudge.trace_pair import TracePair
import grudge.op as op

from mirgecom.boundary import (
    AdiabaticNoslipWallBoundary,
    PrescribedFluidBoundary,
    IsothermalWallBoundary
)


class _AVRTag:
    pass


def _identical_grad_av(grad_av_minus, **kwargs):
    return grad_av_minus


class AdiabaticNoSlipWallAV(AdiabaticNoslipWallBoundary):
    r"""Interface to a prescribed adiabatic noslip fluid boundary with AV.

    .. automethod:: __init__
    .. automethod:: av_flux
    """

    def __init__(self, boundary_grad_av_func=_identical_grad_av,
                 av_num_flux_func=num_flux_central, **kwargs):
        """Initialize the PrescribedFluidBoundaryAV and methods."""
        self._bnd_grad_av_func = boundary_grad_av_func
        self._av_num_flux_func = av_num_flux_func
        AdiabaticNoslipWallBoundary.__init__(self, **kwargs)

    def _boundary_quantity(self, dcoll, dd_bdry, quantity, local=False, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        dd_allfaces = dd_bdry.with_boundary_tag(FACE_RESTR_ALL)
        return quantity if local else op.project(dcoll,
            dd_bdry, dd_allfaces, quantity)

    def av_flux(self, dcoll, dd_bdry, diffusion, **kwargs):
        """Get the diffusive fluxes for the AV operator API."""
        dd_bdry = as_dofdesc(dd_bdry)
        grad_av_minus = op.project(dcoll, dd_bdry.untrace(), dd_bdry, diffusion)
        actx = get_container_context_recursively(grad_av_minus)
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        grad_av_plus = grad_av_minus
        bnd_grad_pair = TracePair(dd_bdry, interior=grad_av_minus,
                                  exterior=grad_av_plus)
        num_flux = self._av_num_flux_func(bnd_grad_pair.int, bnd_grad_pair.ext)@nhat
        return self._boundary_quantity(dcoll, dd_bdry, num_flux, **kwargs)


class IsothermalWallAV(IsothermalWallBoundary):
    r"""Interface to a prescribed isothermal noslip wall boundary with AV.

    .. automethod:: __init__
    .. automethod:: av_flux
    """

    def __init__(self, boundary_grad_av_func=_identical_grad_av,
                 av_num_flux_func=num_flux_central, **kwargs):
        """Initialize the PrescribedFluidBoundaryAV and methods."""
        self._bnd_grad_av_func = boundary_grad_av_func
        self._av_num_flux_func = av_num_flux_func
        IsothermalWallBoundary.__init__(self, **kwargs)

    def _boundary_quantity(self, dcoll, dd_bdry, quantity, local=False, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        dd_allfaces = dd_bdry.with_boundary_tag(FACE_RESTR_ALL)
        return quantity if local else op.project(dcoll,
            dd_bdry, dd_allfaces, quantity)

    def av_flux(self, dcoll, dd_bdry, diffusion, **kwargs):
        """Get the diffusive fluxes for the AV operator API."""
        dd_bdry = as_dofdesc(dd_bdry)
        grad_av_minus = op.project(dcoll, dd_bdry.untrace(), dd_bdry, diffusion)
        actx = get_container_context_recursively(grad_av_minus)
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        grad_av_plus = grad_av_minus
        bnd_grad_pair = TracePair(dd_bdry, interior=grad_av_minus,
                                  exterior=grad_av_plus)
        num_flux = self._av_num_flux_func(bnd_grad_pair.int, bnd_grad_pair.ext)@nhat
        return self._boundary_quantity(dcoll, dd_bdry, num_flux, **kwargs)


# This class is a FluidBoundary that provides default implementations of
# the abstract methods in FluidBoundary. This class will be eliminated
# by resolution of https://github.com/illinois-ceesd/mirgecom/issues/576.
# TODO: Don't do this. Make every boundary condition implement its own
# version of the FluidBoundary methods.
class PrescribedFluidBoundaryAV(PrescribedFluidBoundary):
    r"""Interface to a prescribed fluid boundary treatment with AV.

    .. automethod:: __init__
    .. automethod:: av_flux
    """

    def __init__(self, boundary_grad_av_func=_identical_grad_av,
                 av_num_flux_func=num_flux_central, **kwargs):
        """Initialize the PrescribedFluidBoundaryAV and methods."""
        self._bnd_grad_av_func = boundary_grad_av_func
        self._av_num_flux_func = av_num_flux_func
        PrescribedFluidBoundary.__init__(self, **kwargs)

    def av_flux(self, dcoll, dd_bdry, diffusion, **kwargs):
        """Get the diffusive fluxes for the AV operator API."""
        dd_bdry = as_dofdesc(dd_bdry)
        grad_av_minus = op.project(dcoll, dd_bdry.untrace(), dd_bdry, diffusion)
        actx = get_container_context_recursively(grad_av_minus)
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        grad_av_plus = self._bnd_grad_av_func(
            dcoll=dcoll, dd_bdry=dd_bdry, grad_av_minus=grad_av_minus, **kwargs)
        bnd_grad_pair = TracePair(dd_bdry, interior=grad_av_minus,
                                  exterior=grad_av_plus)
        num_flux = self._av_num_flux_func(bnd_grad_pair.int, bnd_grad_pair.ext)@nhat
        return self._boundary_quantity(dcoll, dd_bdry, num_flux, **kwargs)

    # }}}


def av_laplacian_operator(dcoll, boundaries, fluid_state, alpha, gas_model=None,
                          kappa=1., s0=-6., time=0, quadrature_tag=DISCR_TAG_BASE,
                          dd=DD_VOLUME_ALL, boundary_kwargs=None, indicator=None,
                          divergence_numerical_flux=num_flux_central, comm_tag=None,
                          operator_states_quad=None,
                          grad_cv=None,
                          **kwargs):
    r"""Compute the artificial viscosity right-hand-side.

    Computes the the right-hand-side term for artificial viscosity.

    .. math::

        \mbox{RHS}_{\mbox{av}} = \nabla\cdot{\varepsilon\nabla\mathbf{Q}}

    Parameters
    ----------
    fluid_state: :class:`mirgecom.gas_model.FluidState`
        Fluid state object with the conserved and thermal state.

    boundaries: dict
        Dictionary of boundary functions, one for each valid boundary tag

    alpha: float
        The maximum artificial viscosity coefficient to be applied

    indicator: :class:`~meshmode.dof_array.DOFArray`
        The indicator field used for locating where AV should be applied. If not
        supplied by the user, then
        :func:`~mirgecom.artificial_viscosity.smoothness_indicator` will be used
        with fluid mass density as the indicator field.

    kappa
        An optional argument that controls the width of the transition from 0 to 1,
        $\kappa$. This parameter defaults to $\kappa=1$.

    s0
        An optional argument that sets the smoothness level to limit
        on, $s_0$. Values in the range $(-\infty,0]$ are allowed, where $-\infty$
        results in all cells being tagged and 0 results in none.  This parameter
        defaults to $s_0=-6$.

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.

    dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *fluid_state* lives.
        Must be a volume on the base discretization.

    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`
        The artificial viscosity operator applied to *q*.
    """
    boundaries = normalize_boundaries(boundaries)

    cv = fluid_state.cv
    actx = cv.array_context

    if not isinstance(dd.domain_tag, VolumeDomainTag):
        raise TypeError("dd must represent a volume")
    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    from warnings import warn

    if boundary_kwargs is not None:
        warn("The AV boundary_kwargs interface is deprecated, please pass gas_model"
             " and time directly.")
        if gas_model is None:
            gas_model = boundary_kwargs["gas_model"]
            if "time" in boundary_kwargs:
                time = boundary_kwargs["time"]

    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    def interp_to_vol_quad(u):
        return op.project(dcoll, dd_vol, dd_vol_quad, u)

    if operator_states_quad is None:
        from mirgecom.gas_model import make_operator_fluid_states
        operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries, quadrature_tag,
            dd=dd_vol, comm_tag=comm_tag)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    # Get smoothness indicator based on mass component
    if indicator is None:
        indicator = smoothness_indicator(dcoll, fluid_state.mass_density,
                                         kappa=kappa, s0=s0, dd=dd_vol)

    if grad_cv is None:
        from mirgecom.navierstokes import grad_cv_operator
        grad_cv = grad_cv_operator(dcoll, gas_model, boundaries, fluid_state,
                                   time=time, quadrature_tag=quadrature_tag,
                                   dd=dd_vol,
                                   comm_tag=comm_tag,
                                   operator_states_quad=operator_states_quad)

    # Compute R = alpha*grad(Q)
    r = -alpha * indicator * grad_cv

    def central_flux_div(utpair_quad):
        dd_trace_quad = utpair_quad.dd
        dd_allfaces_quad = dd_trace_quad.with_boundary_tag(FACE_RESTR_ALL)
        normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
        return op.project(dcoll, dd_trace_quad, dd_allfaces_quad,
                          # This uses a central vector flux along nhat:
                          # flux = 1/2 * (grad(Q)- + grad(Q)+) .dot. nhat
                          divergence_numerical_flux(
                              utpair_quad.int, utpair_quad.ext)@normal_quad)

    # Total flux of grad(Q) across element boundaries
    r_bnd = (
        # Rank-local and cross-rank (across parallel partitions) contributions
        sum(
            central_flux_div(interp_to_surf_quad(tpair=tpair))
            for tpair in interior_trace_pairs(
                dcoll, r, volume_dd=dd_vol, comm_tag=(_AVRTag, comm_tag)))

        # Contributions from boundary fluxes
        + sum(
            bdry.av_flux(
                dcoll,
                dd_bdry=dd_vol.with_domain_tag(bdtag),
                diffusion=r)
            for bdtag, bdry in boundaries.items())
    )

    # Return the AV RHS term
    return -div_operator(
        dcoll, dd_vol_quad, dd_allfaces_quad, interp_to_vol_quad(r), r_bnd)


def smoothness_indicator(dcoll, u, kappa=1.0, s0=-6.0, dd=DD_VOLUME_ALL):
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

    @memoize_in(actx, (smoothness_indicator, "smooth_comp_knl", dd))
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

    @keyed_memoize_in(actx, (smoothness_indicator, "highest_mode", dd),
                      lambda grp: grp.discretization_key())
    def highest_mode(grp):
        return actx.from_numpy(
            np.asarray([1 if sum(mode_id) == grp.order
                        else 0
                        for mode_id in grp.mode_ids()])
        )

    # Convert to modal solution representation
    dd_vol = dd
    dd_modal = dd_vol.with_discr_tag(DISCR_TAG_MODAL)
    modal_map = dcoll.connection_from_dds(dd_vol, dd_modal)
    uhat = modal_map(u)

    # Compute smoothness indicator value
    if actx.supports_nonscalar_broadcasting:
        from meshmode.transform_metadata import DiscretizationDOFAxisTag
        indicator = DOFArray(
            actx,
            data=tuple(
                actx.tag_axis(
                    1,
                    DiscretizationDOFAxisTag(),
                    actx.np.broadcast_to(
                        ((actx.einsum("ek,k->e",
                                      uhat[igrp]**2,
                                      highest_mode(grp))
                          / (actx.einsum("ej->e",
                                         (uhat[igrp]**2+(1e-12/grp.nunit_dofs))
                                         )))
                         .reshape(-1, 1)),
                        uhat[igrp].shape))
                for igrp, grp in enumerate(dcoll.discr_from_dd(dd_vol).groups)
            )
        )
    else:
        indicator = DOFArray(
            actx,
            data=tuple(
                actx.call_loopy(
                    indicator_prg(),
                    vec=uhat[igrp],
                    modes_active_flag=highest_mode(grp)
                )["result"]
                for igrp, grp in enumerate(dcoll.discr_from_dd(dd_vol).groups)
            )
        )

    indicator = actx.np.log10(indicator + 1.0e-12)

    # Compute artificial viscosity percentage based on indicator and set parameters
    yesnol = actx.np.greater(indicator, (s0 - kappa))
    yesnou = actx.np.greater(indicator, (s0 + kappa))
    saintly_value = 1.0
    sin_indicator = actx.np.where(
        yesnol,
        0.5 * (1.0 + actx.np.sin(np.pi * (indicator - s0) / (2.0 * kappa))),
        0.0 * indicator,
    )
    indicator = actx.np.where(yesnou, saintly_value, sin_indicator)

    return indicator
