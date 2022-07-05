r""":mod:`mirgecom.navierstokes` methods and utils for compressible Navier-Stokes.

Compressible Navier-Stokes equations:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F}_{I} = \nabla\cdot\mathbf{F}_{V}

where:

-  fluid state $\mathbf{Q} = [\rho, \rho{E}, \rho\mathbf{v}, \rho{Y}_\alpha]$
-  with fluid density $\rho$, flow energy $E$, velocity $\mathbf{v}$, and vector
   of species mass fractions ${Y}_\alpha$, where $1\le\alpha\le\mathtt{nspecies}$.
-  inviscid flux $\mathbf{F}_{I} = [\rho\mathbf{v},(\rho{E} + p)\mathbf{v}
   ,(\rho(\mathbf{v}\otimes\mathbf{v})+p\mathbf{I}), \rho{Y}_\alpha\mathbf{v}]$
-  viscous flux $\mathbf{F}_V = [0,((\tau\cdot\mathbf{v})-\mathbf{q}),\tau_{:i}
   ,J_{\alpha}]$
-  viscous stress tensor $\mathbf{\tau} = \mu(\nabla\mathbf{v}+(\nabla\mathbf{v})^T)
   + (\mu_B - \frac{2}{3}\mu)(\nabla\cdot\mathbf{v})$
-  diffusive flux for each species $J_\alpha = -\rho{D}_{\alpha}\nabla{Y}_{\alpha}$
-  total heat flux $\mathbf{q}=\mathbf{q}_c+\mathbf{q}_d$, is the sum of:
    -  conductive heat flux $\mathbf{q}_c = -\kappa\nabla{T}$
    -  diffusive heat flux $\mathbf{q}_d = \sum{h_{\alpha} J_{\alpha}}$
-  fluid pressure $p$, temperature $T$, and species specific enthalpies $h_\alpha$
-  fluid viscosity $\mu$, bulk viscosity $\mu_{B}$, fluid heat conductivity $\kappa$,
   and species diffusivities $D_{\alpha}$.

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: grad_cv_operator
.. autofunction:: grad_t_operator
.. autofunction:: ns_operator
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

from dataclasses import replace
from functools import partial

from meshmode.discretization.connection import FACE_RESTR_ALL

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs,
    tracepair_with_discr_tag
)
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    as_dofdesc,
)

import grudge.op as op

from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux_rusanov,
    inviscid_flux_on_element_boundary
)
from mirgecom.viscous import (
    viscous_flux,
    viscous_facial_flux_central,
    viscous_flux_on_element_boundary
)
from mirgecom.flux import num_flux_central

from mirgecom.operators import (
    div_operator, grad_operator
)
from mirgecom.gas_model import make_operator_fluid_states


class _NSGradCVTag:
    pass


class _NSGradTemperatureTag:
    pass


def _gradient_flux_interior(discr, numerical_flux_func, tpair):
    """Compute interior face flux for gradient operator."""
    from arraycontext import outer
    actx = tpair.int.array_context
    dd_trace = tpair.dd
    dd_allfaces = dd_trace.with_domain_tag(
        replace(dd_trace.domain_tag, tag=FACE_RESTR_ALL))
    normal = actx.thaw(discr.normal(dd_trace))
    flux = outer(numerical_flux_func(tpair.int, tpair.ext), normal)
    return op.project(discr, dd_trace, dd_allfaces, flux)


def grad_cv_operator(
        discr, gas_model, boundaries, state, *, time=0.0,
        numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE, volume_dd=DD_VOLUME_ALL,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        operator_states_quad=None):
    r"""Compute the gradient of the fluid conserved variables.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions, one for each valid
        :class:`~grudge.dof_desc.BoundaryDomainTag`

    time
        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    numerical_flux_func:

       Optional callable function to return the numerical flux to be used when
       computing gradients. Defaults to :class:`~mirgecom.flux.num_flux_central`.

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    volume_dd: grudge.dof_desc.DOFDesc
        The DOF descriptor of the volume on which to apply the operator.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        CV object with vector components representing the gradient of the fluid
        conserved variables.
    """
    boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in boundaries.items()}

    dd_vol_base = volume_dd
    dd_vol_quad = dd_vol_base.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    if operator_states_quad is None:
        operator_states_quad = make_operator_fluid_states(
            discr, state, gas_model, boundaries, quadrature_tag,
            volume_dd=dd_vol_base)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    get_interior_flux = partial(
        _gradient_flux_interior, discr, numerical_flux_func)

    cv_interior_pairs = [TracePair(state_pair.dd,
                                   interior=state_pair.int.cv,
                                   exterior=state_pair.ext.cv)
                         for state_pair in inter_elem_bnd_states_quad]

    cv_flux_bnd = (

        # Domain boundaries
        sum(op.project(
            discr, dd_vol_quad.with_domain_tag(bdtag),
            dd_allfaces_quad,
            bdry.cv_gradient_flux(
                discr,
                dd_vol_quad.with_domain_tag(bdtag),
                gas_model=gas_model,
                state_minus=domain_bnd_states_quad[bdtag],
                time=time,
                numerical_flux_func=numerical_flux_func))
            for bdtag, bdry in boundaries.items())

        # Interior boundaries
        + sum(get_interior_flux(tpair) for tpair in cv_interior_pairs)
    )

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    return grad_operator(
        discr, dd_vol_quad, dd_allfaces_quad, vol_state_quad.cv, cv_flux_bnd)


def grad_t_operator(
        discr, gas_model, boundaries, state, *, time=0.0,
        numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE, volume_dd=DD_VOLUME_ALL,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        operator_states_quad=None):
    r"""Compute the gradient of the fluid temperature.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions keyed by btags

    time
        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    numerical_flux_func:

       Optional callable function to return the numerical flux to be used when
       computing gradients. Defaults to :class:`~mirgecom.flux.num_flux_central`.

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    volume_dd: grudge.dof_desc.DOFDesc
        The DOF descriptor of the volume on which to apply the operator.

    Returns
    -------
    :class:`numpy.ndarray`

        Array of :class:`~meshmode.dof_array.DOFArray` representing the gradient of
        the fluid temperature.
    """
    boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in boundaries.items()}

    dd_vol_base = volume_dd
    dd_vol_quad = dd_vol_base.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    if operator_states_quad is None:
        operator_states_quad = make_operator_fluid_states(
            discr, state, gas_model, boundaries, quadrature_tag,
            volume_dd=dd_vol_base)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    get_interior_flux = partial(
        _gradient_flux_interior, discr, numerical_flux_func)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (4c)
    # Capture the temperature for the interior faces for grad(T) calc
    # Note this is *all interior faces*, including partition boundaries
    # due to the use of *interior_state_pairs*.
    t_interior_pairs = [TracePair(state_pair.dd,
                                  interior=state_pair.int.temperature,
                                  exterior=state_pair.ext.temperature)
                        for state_pair in inter_elem_bnd_states_quad]

    t_flux_bnd = (

        # Domain boundaries
        sum(op.project(
            discr, dd_vol_quad.with_domain_tag(bdtag),
            dd_allfaces_quad,
            bdry.temperature_gradient_flux(
                discr,
                dd_vol_quad.with_domain_tag(bdtag),
                gas_model=gas_model,
                state_minus=domain_bnd_states_quad[bdtag],
                time=time,
                numerical_flux_func=numerical_flux_func))
            for bdtag, bdry in boundaries.items())

        # Interior boundaries
        + sum(get_interior_flux(tpair) for tpair in t_interior_pairs)
    )

    # Fluxes in-hand, compute the gradient of temperature
    return grad_operator(
        discr, dd_vol_quad, dd_allfaces_quad, vol_state_quad.temperature, t_flux_bnd)


def ns_operator(discr, gas_model, state, boundaries, *, time=0.0,
                inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
                gradient_numerical_flux_func=num_flux_central,
                viscous_numerical_flux_func=viscous_facial_flux_central,
                return_gradients=False, quadrature_tag=DISCR_TAG_BASE,
                volume_dd=DD_VOLUME_ALL,
                # Added to avoid repeated computation
                # FIXME: See if there's a better way to do this
                operator_states_quad=None,
                grad_cv=None, grad_t=None):
    r"""Compute RHS of the Navier-Stokes equations.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions keyed by btags

    time
        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    inviscid_numerical_flux_func:
        Optional callable function providing the face-normal flux to be used
        for the divergence of the inviscid transport flux.  This defaults to
        :func:`~mirgecom.inviscid.inviscid_facial_flux_rusanov`.

    viscous_numerical_flux_func:
        Optional callable function providing the face-normal flux to be used
        for the divergence of the viscous transport flux.  This defaults to
        :func:`~mirgecom.viscous.viscous_facial_flux_central`.

    gradient_numerical_flux_func:
       Optional callable function to return the numerical flux to be used when
       computing gradients in the Navier-Stokes operator.

    return_gradients
        Optional boolean (defaults to false) indicating whether to return
        $\nabla(\text{CV})$ and $\nabla(T)$ along with the RHS for the Navier-Stokes
        equations.  Useful for debugging and visualization.

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    volume_dd: grudge.dof_desc.DOFDesc
        The DOF descriptor of the volume on which to apply the operator.

    operator_states_quad
        Optional iterable container providing the full fluid states
        (:class:`~mirgecom.gas_model.FluidState`) on the quadrature
        domain (if any) on each of the volume, internal faces tracepairs
        (including partition boundaries), and minus side of domain boundary faces.
        If this data structure is not provided, it will be calculated with
        :func:`~mirgecom.gas_model.make_operator_fluid_states`.

    grad_cv: :class:`~mirgecom.fluid.ConservedVars`
        Optional CV object containing the gradient of the fluid conserved quantities.
        If not provided, the operator will calculate it with
        :func:`~mirgecom.navierstokes.grad_cv_operator`

    grad_t: numpy.ndarray
        Optional array containing the gradient of the fluid temperature. If not
        provided, the operator will calculate it with
        :func:`~mirgecom.navierstokes.grad_t_operator`.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)
    """
    if not state.is_viscous:
        raise ValueError("Navier-Stokes operator expects viscous gas model.")

    boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in boundaries.items()}

    dd_vol_base = volume_dd
    dd_vol_quad = dd_vol_base.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    # Make model-consistent fluid state data (i.e. CV *and* DV) for:
    # - Volume: vol_state_quad
    # - Element-element boundary face trace pairs: inter_elem_bnd_states_quad
    # - Interior states (Q_minus) on the domain boundary: domain_bnd_states_quad
    #
    # Note: these states will live on the quadrature domain if one is given,
    # otherwise they stay on the interpolatory/base domain.
    if operator_states_quad is None:
        operator_states_quad = make_operator_fluid_states(
            discr, state, gas_model, boundaries, quadrature_tag,
            volume_dd=dd_vol_base)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    # {{{ Local utilities

    # transfer trace pairs to quad grid, update pair dd
    interp_to_surf_quad = partial(tracepair_with_discr_tag, discr, quadrature_tag)

    # }}}

    # {{{ === Compute grad(CV) ===

    if grad_cv is None:
        grad_cv = grad_cv_operator(
            discr, gas_model, boundaries, state, time=time,
            numerical_flux_func=gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag, volume_dd=dd_vol_base,
            operator_states_quad=operator_states_quad)

    # Communicate grad(CV) and put it on the quadrature domain
    grad_cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(
            discr, grad_cv, volume_dd=dd_vol_base, tag=_NSGradCVTag)
    ]

    # }}} Compute grad(CV)

    # {{{ === Compute grad(temperature) ===

    if grad_t is None:
        grad_t = grad_t_operator(
            discr, gas_model, boundaries, state, time=time,
            numerical_flux_func=gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag, volume_dd=dd_vol_base,
            operator_states_quad=operator_states_quad)

    # Create the interior face trace pairs, perform MPI exchange, interp to quad
    grad_t_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(
            discr, grad_t, volume_dd=dd_vol_base, tag=_NSGradTemperatureTag)
    ]

    # }}} compute grad(temperature)

    # {{{ === Navier-Stokes RHS ===

    # Compute the volume term for the divergence operator
    vol_term = (

        # Compute the volume contribution of the viscous flux terms
        # using field values on the quadrature grid
        viscous_flux(state=vol_state_quad,
                     # Interpolate gradients to the quadrature grid
                     grad_cv=op.project(discr, dd_vol_base, dd_vol_quad, grad_cv),
                     grad_t=op.project(discr, dd_vol_base, dd_vol_quad, grad_t))

        # Compute the volume contribution of the inviscid flux terms
        # using field values on the quadrature grid
        - inviscid_flux(state=vol_state_quad)
    )

    # Compute the boundary terms for the divergence operator
    bnd_term = (

        # All surface contributions from the viscous fluxes
        viscous_flux_on_element_boundary(
            discr, gas_model, boundaries, inter_elem_bnd_states_quad,
            domain_bnd_states_quad, grad_cv, grad_cv_interior_pairs,
            grad_t, grad_t_interior_pairs, quadrature_tag=quadrature_tag,
            numerical_flux_func=viscous_numerical_flux_func, time=time,
            volume_dd=dd_vol_base)

        # All surface contributions from the inviscid fluxes
        - inviscid_flux_on_element_boundary(
            discr, gas_model, boundaries, inter_elem_bnd_states_quad,
            domain_bnd_states_quad, quadrature_tag=quadrature_tag,
            numerical_flux_func=inviscid_numerical_flux_func, time=time,
            volume_dd=dd_vol_base)

    )
    ns_rhs = div_operator(discr, dd_vol_quad, dd_allfaces_quad, vol_term, bnd_term)
    if return_gradients:
        return ns_rhs, grad_cv, grad_t
    return ns_rhs

    # }}} NS RHS
