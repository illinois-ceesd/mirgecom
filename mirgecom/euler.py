r""":mod:`mirgecom.euler` helps solve Euler's equations of gas dynamics.

Euler's equations of gas dynamics:

.. math::

    \partial_t \mathbf{Q} = -\nabla\cdot{\mathbf{F}} +
    (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

where:

-  state $\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V}, \rho{Y}_\alpha]$
-  flux $\mathbf{F} = [\rho\vec{V},(\rho{E} + p)\vec{V},
   (\rho(\vec{V}\otimes\vec{V}) + p*\mathbf{I}), \rho{Y}_\alpha\vec{V}]$,
-  unit normal $\hat{n}$ to the domain boundary $\partial\Omega$,
-  vector of species mass fractions ${Y}_\alpha$,
   with $1\le\alpha\le\mathtt{nspecies}$.

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: euler_operator

Logging Helpers
^^^^^^^^^^^^^^^

.. autofunction:: units_for_logging
.. autofunction:: extract_vars_for_logging
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
from warnings import warn

from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    VolumeDomainTag,
    DISCR_TAG_BASE,
)

from mirgecom.gas_model import make_operator_fluid_states
from mirgecom.inviscid import (  # noqa
    inviscid_flux,
    inviscid_facial_flux_rusanov,
    inviscid_flux_on_element_boundary,
    entropy_stable_inviscid_facial_flux_rusanov,
    entropy_stable_inviscid_facial_flux,
    entropy_conserving_flux_chandrashekar,
    entropy_conserving_flux_renac
)

from mirgecom.operators import div_operator
from mirgecom.utils import normalize_boundaries
from arraycontext import map_array_container
from mirgecom.gas_model import (
    project_fluid_state,
    make_fluid_state_trace_pairs,
    make_entropy_projected_fluid_state,
    conservative_to_entropy_vars,
    entropy_to_conservative_vars
)

from meshmode.dof_array import DOFArray

from functools import partial

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs,
    tracepair_with_discr_tag
)

import grudge.op as op


class _ESFluidCVTag():
    pass


class _ESFluidTemperatureTag():
    pass


def entropy_stable_euler_operator(
        dcoll, gas_model, state, boundaries, time=0.0,
        inviscid_numerical_flux_func=None,
        entropy_conserving_flux_func=None,
        operator_states_quad=None,
        dd=DD_VOLUME_ALL, quadrature_tag=None, comm_tag=None,
        limiter_func=None):
    """Compute RHS of the Euler flow equations using flux-differencing.

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

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`
        Agglomerated object array of DOF arrays representing the RHS of the Euler
        flow equations.
    """
    boundaries = normalize_boundaries(boundaries)

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    # NOTE: For single-gas this is just a fixed scalar.
    # However, for mixtures, gamma is a DOFArray. For now,
    # we are re-using gamma from here and *not* recomputing
    # after applying entropy projections. It is unclear at this
    # time whether it's strictly necessary or if this is good enough
    gamma_base = gas_model.eos.gamma(state.cv, state.temperature)

    # Interpolate state to vol quad grid
    if operator_states_quad is not None:
        state_quad = operator_states_quad[0]
    else:
        if state.is_mixture and limiter_func is None:
            warn("Mixtures often require species limiting, and a non-limited "
                 "state is being created for this operator. For mixtures, "
                 "one should pass the operator_states_quad argument with "
                 "limited states, or least pass a limiter_func to this operator.")
        state_quad = project_fluid_state(
            dcoll, dd_vol, dd_vol_quad, state, gas_model, limiter_func=limiter_func,
            entropy_stable=True)

    gamma_quad = gas_model.eos.gamma(state_quad.cv, state_quad.temperature)

    # Compute the projected (nodal) entropy variables
    from grudge.projection import volume_quadrature_project  \
        # pylint: disable=no-name-in-module

    entropy_vars = volume_quadrature_project(
        dcoll, dd_vol_quad,
        # Map to entropy variables
        conservative_to_entropy_vars(gamma_quad, state_quad))

    modified_conserved_fluid_state = \
        make_entropy_projected_fluid_state(dcoll, dd_vol_quad, dd_allfaces_quad,
                                           state, entropy_vars, gamma_base,
                                           gas_model)

    def _reshape(shape, ary):
        if not isinstance(ary, DOFArray):
            return map_array_container(partial(_reshape, shape), ary)

        return DOFArray(ary.array_context, data=tuple(
            subary.reshape(grp.nelements, *shape)
            # Just need group for determining the number of elements
            for grp, subary in zip(dcoll.discr_from_dd(dd_vol).groups, ary)))

    if entropy_conserving_flux_func is None:
        entropy_conserving_flux_func = \
            (entropy_conserving_flux_renac if state.is_mixture
             else entropy_conserving_flux_chandrashekar)
        flux_func = "renac" if state.is_mixture else "chandrashekar"
        warn("No entropy_conserving_flux_func was given for ESDG. "
             f"Setting EC flux to entropy_conserving_flux_{flux_func}.")

    flux_matrices = entropy_conserving_flux_func(
        gas_model,
        _reshape((1, -1), modified_conserved_fluid_state),
        _reshape((-1, 1), modified_conserved_fluid_state))

    # Compute volume derivatives using flux differencing
    from grudge.flux_differencing import volume_flux_differencing  \
        # pylint: disable=no-name-in-module,import-error

    inviscid_vol_term = \
        -volume_flux_differencing(dcoll, dd_vol_quad, dd_allfaces_quad,
                                  flux_matrices)

    # transfer trace pairs to quad grid, update pair dd
    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    tseed_interior_pairs = None
    if state.is_mixture:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the inviscid flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        tseed_interior_pairs = [
            # Get the interior trace pairs onto the surface quadrature
            # discretization (if any)
            interp_to_surf_quad(tpair)
            for tpair in interior_trace_pairs(dcoll, state.temperature,
                                              volume_dd=dd_vol,
                                              comm_tag=(_ESFluidTemperatureTag,
                                                        comm_tag))
        ]

    def _interp_to_surf_modified_conservedvars(gamma, ev_pair):
        # Takes a trace pair containing the projected entropy variables
        # and converts them into conserved variables on the quadrature grid.
        local_dd = ev_pair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)

        # Interpolate entropy variables to the surface quadrature grid
        ev_pair_surf = op.project(dcoll, local_dd, local_dd_quad, ev_pair)

        if isinstance(gamma, DOFArray):
            gamma = op.project(dcoll, dd_vol, local_dd_quad, gamma)

        return TracePair(
            local_dd_quad,
            # Convert interior and exterior states to conserved variables
            interior=entropy_to_conservative_vars(gamma, ev_pair_surf.int),
            exterior=entropy_to_conservative_vars(gamma, ev_pair_surf.ext)
        )

    cv_interior_pairs = [
        # Compute interior trace pairs using modified conservative
        # variables on the quadrature grid
        # (obtaining state from projected entropy variables)
        _interp_to_surf_modified_conservedvars(gamma_base, tpair)
        for tpair in interior_trace_pairs(dcoll, entropy_vars, volume_dd=dd_vol,
                                          comm_tag=(_ESFluidCVTag, comm_tag))]

    boundary_states = {
        # TODO: Use modified conserved vars as the input state?
        # Would need to make an "entropy-projection" variant
        # of *project_fluid_state*
        bdtag: project_fluid_state(
            dcoll, dd_vol,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            dd_vol_quad.with_domain_tag(bdtag),
            state, gas_model, entropy_stable=True) for bdtag in boundaries
    }

    # Interior interface state pairs consisting of modified conservative
    # variables and the corresponding temperature seeds
    interior_states = make_fluid_state_trace_pairs(cv_interior_pairs,
                                                   gas_model,
                                                   tseed_interior_pairs)

    if inviscid_numerical_flux_func is None:
        inviscid_numerical_flux_func = \
            partial(entropy_stable_inviscid_facial_flux_rusanov,
                    entropy_conserving_flux_func=entropy_conserving_flux_func)
        warn("No inviscid_numerical_flux_func was given for ESDG. "
             "Automatically setting facial flux to entropy-stable Rusanov "
             "(entropy_stable_inviscid_facial_flux_rusanov).")
    elif inviscid_numerical_flux_func not in \
         [entropy_stable_inviscid_facial_flux_rusanov,
          entropy_stable_inviscid_facial_flux]:
        warn("Unrecognized inviscid_numerical_flux_func for ESDG. Proceed only "
             "if you know what you are doing. An ESDG-compatible facial flux "
             "function *must* be used with ESDG. Valid built-in choices are:\n"
             "* entropy_stable_inviscid_facial_flux_rusanov, -or-\n"
             "* entropy_stable_inviscid_facial_flux\n")

    # Compute interface contributions
    inviscid_flux_bnd = inviscid_flux_on_element_boundary(
        dcoll, gas_model, boundaries, interior_states,
        boundary_states, quadrature_tag=quadrature_tag,
        numerical_flux_func=inviscid_numerical_flux_func, time=time,
        dd=dd_vol)

    return op.inverse_mass(
        dcoll,
        dd_vol,
        inviscid_vol_term - op.face_mass(dcoll, dd_allfaces_quad,
                                         inviscid_flux_bnd)
    )


def euler_operator(dcoll, state, gas_model, boundaries, time=0.0,
                   inviscid_numerical_flux_func=None,
                   quadrature_tag=DISCR_TAG_BASE, dd=DD_VOLUME_ALL,
                   comm_tag=None, use_esdg=False, operator_states_quad=None,
                   entropy_conserving_flux_func=None, limiter_func=None):
    r"""Compute RHS of the Euler flow equations.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        The right-hand-side of the Euler flow equations:

        .. math::

            \dot{\mathbf{q}} = - \nabla\cdot\mathbf{F} +
                (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

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

    quadrature_tag

        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.

    dd: grudge.dof_desc.DOFDesc

        the DOF descriptor of the discretization on which *state* lives. Must be a
        volume on the base discretization.

    comm_tag: Hashable

        Tag for distributed communication
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

    if use_esdg:
        return entropy_stable_euler_operator(
            dcoll, gas_model=gas_model, state=state, boundaries=boundaries,
            time=time, operator_states_quad=operator_states_quad, dd=dd,
            inviscid_numerical_flux_func=inviscid_numerical_flux_func,
            entropy_conserving_flux_func=entropy_conserving_flux_func,
            quadrature_tag=quadrature_tag, comm_tag=comm_tag)

    if inviscid_numerical_flux_func is None:
        warn("inviscid_numerical_flux_func unspecified, defaulting to "
             "inviscid_facial_flux_rusanov.")
        inviscid_numerical_flux_func = inviscid_facial_flux_rusanov

    volume_state_quad, interior_state_pairs_quad, domain_boundary_states_quad = \
        operator_states_quad

    # Compute volume contributions
    inviscid_flux_vol = inviscid_flux(volume_state_quad)

    # Compute interface contributions
    inviscid_flux_bnd = inviscid_flux_on_element_boundary(
        dcoll, gas_model, boundaries, interior_state_pairs_quad,
        domain_boundary_states_quad, quadrature_tag=quadrature_tag,
        numerical_flux_func=inviscid_numerical_flux_func, time=time,
        dd=dd_vol)

    return -div_operator(dcoll, dd_vol_quad, dd_allfaces_quad,
                         inviscid_flux_vol, inviscid_flux_bnd)


# By default, run unitless
NAME_TO_UNITS = {
    "mass": "",
    "energy": "",
    "momentum": "",
    "temperature": "",
    "pressure": ""
}


def units_for_logging(quantity: str) -> str:
    """Return unit for quantity."""
    return NAME_TO_UNITS[quantity]


def extract_vars_for_logging(dim: int, state, eos) -> dict:
    """Extract state vars."""
    dv = eos.dependent_vars(state)

    from mirgecom.utils import asdict_shallow
    name_to_field = asdict_shallow(state)
    name_to_field.update(asdict_shallow(dv))
    return name_to_field
