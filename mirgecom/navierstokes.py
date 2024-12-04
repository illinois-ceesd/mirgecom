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

from functools import partial
from warnings import warn

from meshmode.discretization.connection import FACE_RESTR_ALL

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs,
    tracepair_with_discr_tag
)
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    VolumeDomainTag,
    DISCR_TAG_BASE,
)

import grudge.geometry as geo
import grudge.op as op

from mirgecom.euler import (
    euler_operator,
    entropy_stable_euler_operator
)
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux_rusanov,
    inviscid_flux_on_element_boundary,
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
from mirgecom.utils import normalize_boundaries


class _NSGradCVTag:
    pass


class _NSGradTemperatureTag:
    pass


class _ESFluidCVTag():
    pass


class _ESFluidTemperatureTag():
    pass


def _gradient_flux_interior(dcoll, numerical_flux_func, tpair):
    """Compute interior face flux for gradient operator."""
    from arraycontext import outer
    actx = tpair.int.array_context
    dd_trace = tpair.dd
    dd_allfaces = dd_trace.with_boundary_tag(FACE_RESTR_ALL)
    normal = geo.normal(actx, dcoll, dd_trace)
    flux = outer(numerical_flux_func(tpair.int, tpair.ext), normal)
    return op.project(dcoll, dd_trace, dd_allfaces, flux)


def grad_cv_operator(
        dcoll, gas_model, boundaries, state, *, time=0.0,
        numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE, dd=DD_VOLUME_ALL, comm_tag=None,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        operator_states_quad=None, limiter_func=None,
        use_esdg=False):
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

    dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *state* lives. Must be a
        volume on the base discretization.

    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        CV object with vector components representing the gradient of the fluid
        conserved variables.
    """
    boundaries = normalize_boundaries(boundaries)

    if not isinstance(dd.domain_tag, VolumeDomainTag):
        raise TypeError("dd must represent a volume")
    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    if operator_states_quad is None:
        if state.is_mixture and limiter_func is None:
            warn("Mixtures often require species limiting, and a non-limited "
                 "state is being created for this operator. For mixtures, "
                 "one should pass the operator_states_quad argument with "
                 "limited states or pass a limiter_func to this operator.")
        operator_states_quad = make_operator_fluid_states(
            dcoll, state, gas_model, boundaries, quadrature_tag,
            dd=dd_vol, comm_tag=comm_tag, limiter_func=limiter_func,
            entropy_stable=use_esdg)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    get_interior_flux = partial(
        _gradient_flux_interior, dcoll, numerical_flux_func)

    cv_interior_pairs = [TracePair(state_pair.dd,
                                   interior=state_pair.int.cv,
                                   exterior=state_pair.ext.cv)
                         for state_pair in inter_elem_bnd_states_quad]

    cv_flux_bnd = (

        # Domain boundaries
        sum(op.project(
            dcoll, dd_vol_quad.with_domain_tag(bdtag),
            dd_allfaces_quad,
            bdry.cv_gradient_flux(
                dcoll,
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
        dcoll, dd_vol_quad, dd_allfaces_quad, vol_state_quad.cv, cv_flux_bnd)


def grad_t_operator(
        dcoll, gas_model, boundaries, state, *, time=0.0,
        numerical_flux_func=num_flux_central,
        quadrature_tag=DISCR_TAG_BASE, dd=DD_VOLUME_ALL, comm_tag=None,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        operator_states_quad=None, limiter_func=None, use_esdg=False):
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

    dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *state* lives. Must be a
        volume on the base discretization.

    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    :class:`numpy.ndarray`

        Array of :class:`~meshmode.dof_array.DOFArray` representing the gradient of
        the fluid temperature.
    """
    boundaries = normalize_boundaries(boundaries)

    if not isinstance(dd.domain_tag, VolumeDomainTag):
        raise TypeError("dd must represent a volume")
    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    if operator_states_quad is None:
        if state.is_mixture and limiter_func is None:
            warn("Mixtures often require species limiting, and a non-limited "
                 "state is being created for this operator. For mixtures, "
                 "one should pass the operator_states_quad argument with "
                 "limited states or pass a limiter_func to this operator.")
        operator_states_quad = make_operator_fluid_states(
            dcoll, state, gas_model, boundaries, quadrature_tag,
            dd=dd_vol, comm_tag=comm_tag, limiter_func=limiter_func,
            entropy_stable=use_esdg)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    get_interior_flux = partial(
        _gradient_flux_interior, dcoll, numerical_flux_func)

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
            dcoll, dd_vol_quad.with_domain_tag(bdtag),
            dd_allfaces_quad,
            bdry.temperature_gradient_flux(
                dcoll,
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
        dcoll, dd_vol_quad, dd_allfaces_quad, vol_state_quad.temperature, t_flux_bnd)


def ns_operator(dcoll, gas_model, state, boundaries, *, time=0.0,
                inviscid_fluid_operator=None,
                inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
                gradient_numerical_flux_func=num_flux_central,
                viscous_numerical_flux_func=viscous_facial_flux_central,
                return_gradients=False, quadrature_tag=DISCR_TAG_BASE,
                dd=DD_VOLUME_ALL, comm_tag=None, limiter_func=None,
                # Added to avoid repeated computation
                # FIXME: See if there's a better way to do this
                operator_states_quad=None, use_esdg=False,
                grad_cv=None, grad_t=None, inviscid_terms_on=True,
                entropy_conserving_flux_func=None):
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
        equations. Useful for debugging and visualization.

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *state* lives. Must be a
        volume on the base discretization.

    comm_tag: Hashable
        Tag for distributed communication

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

    inviscid_terms_on
        Optional boolean to en/disable inviscid terms in this operator.
        Defaults to ON (True).

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)
    """
    if not state.is_viscous:
        raise ValueError("Navier-Stokes operator expects viscous gas model.")

    boundaries = normalize_boundaries(boundaries)

    if not isinstance(dd.domain_tag, VolumeDomainTag):
        raise TypeError("dd must represent a volume")
    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    if inviscid_fluid_operator is None and use_esdg:
        inviscid_fluid_operator = entropy_stable_euler_operator
    elif use_esdg and inviscid_fluid_operator == euler_operator:
        raise RuntimeError("Standard Euler operator is incompatible with ESDG.")

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    # Make model-consistent fluid state data (i.e. CV *and* DV) for:
    # - Volume: vol_state_quad
    # - Element-element boundary face trace pairs: inter_elem_bnd_states_quad
    # - Interior states (Q_minus) on the domain boundary: domain_bnd_states_quad
    #
    # Note: these states will live on the quadrature domain if one is given,
    # otherwise they stay on the interpolatory/base domain.
    if operator_states_quad is None:
        if state.is_mixture and limiter_func is None:
            warn("Mixtures often require species limiting, and a non-limited "
                 "state is being created for this operator. For mixtures, "
                 "one should pass the operator_states_quad argument with "
                 "limited states or provide a limiter_func to this operator.")
        operator_states_quad = make_operator_fluid_states(
            dcoll, state, gas_model, boundaries, quadrature_tag,
            limiter_func=limiter_func, dd=dd_vol, comm_tag=comm_tag)

    vol_state_quad, inter_elem_bnd_states_quad, domain_bnd_states_quad = \
        operator_states_quad

    # {{{ Local utilities

    # transfer trace pairs to quad grid, update pair dd
    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    # }}}

    # {{{ === Compute grad(CV) ===

    if grad_cv is None:
        grad_cv = grad_cv_operator(
            dcoll, gas_model, boundaries, state, time=time,
            numerical_flux_func=gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag, dd=dd_vol,
            operator_states_quad=operator_states_quad, comm_tag=comm_tag,
            use_esdg=use_esdg, limiter_func=limiter_func)

    # Communicate grad(CV) and put it on the quadrature domain
    grad_cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(
            dcoll, grad_cv, volume_dd=dd_vol, comm_tag=(_NSGradCVTag, comm_tag))
    ]

    # }}} Compute grad(CV)

    # {{{ === Compute grad(temperature) ===

    if grad_t is None:
        grad_t = grad_t_operator(
            dcoll, gas_model, boundaries, state, time=time,
            numerical_flux_func=gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag, dd=dd_vol,
            operator_states_quad=operator_states_quad, comm_tag=comm_tag,
            use_esdg=use_esdg, limiter_func=limiter_func)

    # Create the interior face trace pairs, perform MPI exchange, interp to quad
    grad_t_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(
            dcoll, grad_t, volume_dd=dd_vol,
            comm_tag=(_NSGradTemperatureTag, comm_tag))
    ]

    # }}} compute grad(temperature)

    # {{{ === Navier-Stokes RHS ===

    # Physical viscous flux in the element volume
    vol_term = viscous_flux(state=vol_state_quad,
                     # Interpolate gradients to the quadrature grid
                     grad_cv=op.project(dcoll, dd_vol, dd_vol_quad, grad_cv),
                     grad_t=op.project(dcoll, dd_vol, dd_vol_quad, grad_t))

    # Physical viscous flux (f .dot. n) is the boundary term for the div op
    bnd_term = viscous_flux_on_element_boundary(
        dcoll, gas_model, boundaries, inter_elem_bnd_states_quad,
        domain_bnd_states_quad, grad_cv, grad_cv_interior_pairs,
        grad_t, grad_t_interior_pairs, quadrature_tag=quadrature_tag,
        numerical_flux_func=viscous_numerical_flux_func, time=time,
        dd=dd_vol)

    # Add corresponding inviscid parts if enabled
    # Note that this is the default, and highest performing path.
    # To get separate inviscid operator, set inviscid_fluid_operator explicitly
    # or use ESDG.
    if inviscid_terms_on and inviscid_fluid_operator is None:
        vol_term = vol_term - inviscid_flux(state=vol_state_quad,
                                            gas_model=gas_model)
        bnd_term = bnd_term - inviscid_flux_on_element_boundary(
            dcoll, gas_model, boundaries, inter_elem_bnd_states_quad,
            domain_bnd_states_quad, quadrature_tag=quadrature_tag,
            numerical_flux_func=inviscid_numerical_flux_func, time=time,
            dd=dd_vol)

    ns_rhs = div_operator(dcoll, dd_vol_quad, dd_allfaces_quad, vol_term, bnd_term)

    # Call an external operator for the inviscid terms (Euler by default)
    # ESDG *always* uses this branch
    if inviscid_terms_on and inviscid_fluid_operator is not None:
        ns_rhs = ns_rhs + inviscid_fluid_operator(
            dcoll, state=state, gas_model=gas_model, boundaries=boundaries,
            time=time, dd=dd, comm_tag=comm_tag, quadrature_tag=quadrature_tag,
            operator_states_quad=operator_states_quad)

    if return_gradients:
        return ns_rhs, grad_cv, grad_t
    return ns_rhs

    # }}} NS RHS
