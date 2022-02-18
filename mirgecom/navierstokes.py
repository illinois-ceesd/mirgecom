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
-  diffusive flux for each species $J_\alpha = \rho{D}_{\alpha}\nabla{Y}_{\alpha}$
-  total heat flux $\mathbf{q}=\mathbf{q}_c+\mathbf{q}_d$, is the sum of:
    -  conductive heat flux $\mathbf{q}_c = -\kappa\nabla{T}$
    -  diffusive heat flux $\mathbf{q}_d = \sum{h_{\alpha} J_{\alpha}}$
-  fluid pressure $p$, temperature $T$, and species specific enthalpies $h_\alpha$
-  fluid viscosity $\mu$, bulk viscosity $\mu_{B}$, fluid heat conductivity $\kappa$,
   and species diffusivities $D_{\alpha}$.

RHS Evaluation
^^^^^^^^^^^^^^

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

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs
)
from grudge.dof_desc import DOFDesc, as_dofdesc

import grudge.op as op

from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_flux_rusanov,
    inviscid_boundary_flux_for_divergence_operator
)
from mirgecom.viscous import (
    viscous_flux,
    viscous_flux_central,
    viscous_boundary_flux_for_divergence_operator
)
from mirgecom.flux import (
    gradient_flux_central
)
from mirgecom.operators import (
    div_operator, grad_operator
)
from mirgecom.gas_model import make_operator_fluid_states

from arraycontext import thaw


class _NSGradCVTag:
    pass


class _NSGradTemperatureTag:
    pass


def ns_operator(discr, gas_model, state, boundaries, time=0.0,
                inviscid_numerical_flux_func=inviscid_flux_rusanov,
                gradient_numerical_flux_func=gradient_flux_central,
                viscous_numerical_flux_func=viscous_flux_central,
                quadrature_tag=None):
    r"""Compute RHS of the Navier-Stokes equations.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries
        Dictionary of boundary functions keyed by btags

    time
        Time

    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
        Implementing the transport properties including heat conductivity,
        and species diffusivities type(mirgecom.transport.TransportModel).

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        Agglomerated object array of DOF arrays representing the RHS of the
        Navier-Stokes equations.
    """
    if not state.is_viscous:
        raise ValueError("Navier-Stokes operator expects viscous gas model.")

    actx = state.array_context
    dd_base = as_dofdesc("vol")
    dd_vol_quad = DOFDesc("vol", quadrature_tag)
    dd_faces_quad = DOFDesc("all_faces", quadrature_tag)

    # Make model-consistent fluid state data (i.e. CV *and* DV) for:
    # - Volume: volume_state_quad
    # - Interior face trace pairs: interior_boundary_states_quad
    # - Interior states on the domain boundary: domain_boundary_states_quad
    #
    # Note: these states will live on the quadrature domain if one is given,
    # otherwise they stay on the interpolatory/base domain.
    volume_state_quad, interior_boundary_states_quad, domain_boundary_states_quad = \
        make_operator_fluid_states(discr, state, gas_model, boundaries,
                                    quadrature_tag)

    # {{{ Local utilities

    # compute interior face flux for gradient operator
    def _gradient_flux_interior(tpair):
        dd = tpair.dd
        normal = thaw(discr.normal(dd), actx)
        flux = gradient_numerical_flux_func(tpair, normal)
        return op.project(discr, dd, dd.with_dtag("all_faces"), flux)

    # transfer trace pairs to quad grid, update pair dd
    def _interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(discr, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(discr, local_dd, local_dd_quad, utpair.ext)
        )

    # }}}

    # {{{ === Compute grad(CV) ===

    cv_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].cv_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=domain_boundary_states_quad[btag],
            time=time,
            numerical_flux_func=gradient_numerical_flux_func)
            for btag in domain_boundary_states_quad)

        # Interior boundaries
        + sum(_gradient_flux_interior(TracePair(tpair.dd,
                                               interior=tpair.int.cv,
                                               exterior=tpair.ext.cv))
              for tpair in interior_boundary_states_quad)
    )

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = grad_operator(discr, dd_vol_quad, dd_faces_quad,
                            volume_state_quad.cv, cv_flux_bnd)

    # Communicate grad(CV) and put it on the quadrature domain
    # FIXME/ReviewQuestion: communicate grad_cv - already on quadrature dom?
    grad_cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        _interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, grad_cv, tag=_NSGradCVTag)
    ]

    # }}} Compute grad(CV)

    # {{{ === Compute grad(temperature) ===

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # Capture the temperature for the interior faces for grad(T) calc
    # Note this is *all interior faces*, including partition boundaries
    # due to the use of *interior_state_pairs*.
    t_interior_pairs = [TracePair(state_pair.dd,
                                  interior=state_pair.int.temperature,
                                  exterior=state_pair.ext.temperature)
                        for state_pair in interior_boundary_states_quad]

    t_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].temperature_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=domain_boundary_states_quad[btag],
            time=time)
            for btag in boundaries)

        # Interior boundaries
        + sum(_gradient_flux_interior(tpair) for tpair in t_interior_pairs)
    )

    # Fluxes in-hand, compute the gradient of temperaturet
    grad_t = grad_operator(discr, dd_vol_quad, dd_faces_quad,
                           volume_state_quad.temperature, t_flux_bnd)

    # Create the interior face trace pairs, perform MPI exchange, interp to quad
    grad_t_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        _interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, grad_t, tag=_NSGradTemperatureTag)
    ]

    # }}} compute grad(temperature)

    # {{{ === Navier-Stokes RHS ===

    # Compute the volume term for the divergence operator
    vol_term = (

        # Compute the volume contribution of the viscous flux terms
        # using field values on the quadrature grid
        viscous_flux(state=volume_state_quad,
                     # Interpolate gradients to the quadrature grid
                     grad_cv=op.project(discr, dd_base, dd_vol_quad, grad_cv),
                     grad_t=op.project(discr, dd_base, dd_vol_quad, grad_t))

        # Compute the volume contribution of the inviscid flux terms
        # using field values on the quadrature grid
        - inviscid_flux(state=volume_state_quad)
    )

    # Compute the boundary terms for the divergence operator
    bnd_term = (

        # All surface contributions from the viscous fluxes
        viscous_boundary_flux_for_divergence_operator(
            discr, gas_model, boundaries, interior_boundary_states_quad,
            domain_boundary_states_quad, grad_cv, grad_cv_interior_pairs,
            grad_t, grad_t_interior_pairs, quadrature_tag=quadrature_tag,
            numerical_flux_func=viscous_numerical_flux_func, time=time)

        # All surface contributions from the inviscid fluxes
        - inviscid_boundary_flux_for_divergence_operator(
            discr, gas_model, boundaries, interior_boundary_states_quad,
            domain_boundary_states_quad, quadrature_tag=quadrature_tag,
            numerical_flux_func=inviscid_numerical_flux_func, time=time)

    )

    return div_operator(discr, dd_vol_quad, dd_faces_quad, vol_term, bnd_term)

    # }}} NS RHS
