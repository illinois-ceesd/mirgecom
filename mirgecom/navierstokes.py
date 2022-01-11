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
    inviscid_facial_flux,
    inviscid_flux_rusanov
)
from mirgecom.viscous import (
    viscous_flux,
    viscous_facial_flux,
    viscous_flux_central
)
from mirgecom.flux import (
    gradient_flux_central
)
from mirgecom.operators import (
    div_operator, grad_operator
)
from mirgecom.gas_model import (
    project_fluid_state,
    make_fluid_state_trace_pairs
)

from arraycontext import thaw


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
        Dictionary of boundary functions, one for each valid btag

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
    dd_vol = DOFDesc("vol", quadrature_tag)
    dd_faces = DOFDesc("all_faces", quadrature_tag)

    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(discr, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(discr, local_dd, local_dd_quad, utpair.ext)
        )

    boundary_states = {
        btag: project_fluid_state(
            discr, dd_base,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            state, gas_model) for btag in boundaries
    }

    cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, state.cv)
    ]

    tseed_interior_pairs = None
    if state.is_mixture:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        tseed_interior_pairs = [
            # Get the interior trace pairs onto the surface quadrature
            # discretization (if any)
            interp_to_surf_quad(tpair)
            for tpair in interior_trace_pairs(discr, state.temperature)
        ]

    quadrature_state = \
        project_fluid_state(discr, dd_base, dd_vol, state, gas_model)
    interior_state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs,
                                                        gas_model,
                                                        tseed_interior_pairs)

    def gradient_flux_interior(tpair):
        dd = tpair.dd
        normal = thaw(discr.normal(dd), actx)
        flux = gradient_numerical_flux_func(tpair, normal)
        return op.project(discr, dd, dd.with_dtag("all_faces"), flux)

    cv_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].cv_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=boundary_states[btag],
            time=time,
            numerical_flux_func=gradient_numerical_flux_func)
            for btag in boundary_states)

        # Interior boundaries
        + sum(gradient_flux_interior(tpair) for tpair in cv_interior_pairs)
    )

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = grad_operator(discr, dd_vol, dd_faces,
                            quadrature_state.cv, cv_flux_bnd)

    grad_cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, grad_cv)
    ]

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # Capture the temperature for the interior faces for grad(T) calc
    # Note this is *all interior faces*, including partition boundaries
    # due to the use of *interior_state_pairs*.
    t_interior_pairs = [TracePair(state_pair.dd,
                                  interior=state_pair.int.temperature,
                                  exterior=state_pair.ext.temperature)
                        for state_pair in interior_state_pairs]

    t_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].temperature_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=boundary_states[btag],
            time=time)
            for btag in boundary_states)

        # Interior boundaries
        + sum(gradient_flux_interior(tpair) for tpair in t_interior_pairs)
    )

    # Fluxes in-hand, compute the gradient of temperature and mpi exchange it
    grad_t = grad_operator(discr, dd_vol, dd_faces,
                           quadrature_state.temperature, t_flux_bnd)

    grad_t_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, grad_t)
    ]

    # inviscid flux divergence-specific flux function for interior faces
    def finv_divergence_flux_interior(state_pair):
        return inviscid_facial_flux(
            discr, gas_model=gas_model, state_pair=state_pair,
            numerical_flux_func=inviscid_numerical_flux_func)

    # inviscid part of bcs applied here
    def finv_divergence_flux_boundary(btag, boundary_state):
        return boundaries[btag].inviscid_divergence_flux(
            discr,
            # Make sure we fields on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=boundary_state,
            time=time,
            numerical_flux_func=inviscid_numerical_flux_func
        )

    # viscous fluxes across interior faces (including partition and periodic bnd)
    def fvisc_divergence_flux_interior(state_pair, grad_cv_pair, grad_t_pair):
        return viscous_facial_flux(discr=discr, gas_model=gas_model,
                                   state_pair=state_pair, grad_cv_pair=grad_cv_pair,
                                   grad_t_pair=grad_t_pair,
                                   numerical_flux_func=viscous_numerical_flux_func)

    # viscous part of bcs applied here
    def fvisc_divergence_flux_boundary(btag, boundary_state):
        # Make sure we fields on the quadrature grid
        # restricted to the tag *btag*
        dd_btag = as_dofdesc(btag).with_discr_tag(quadrature_tag)
        return boundaries[btag].viscous_divergence_flux(
            discr=discr,
            btag=dd_btag,
            gas_model=gas_model,
            state_minus=boundary_state,
            grad_cv_minus=op.project(discr, dd_base, dd_btag, grad_cv),
            grad_t_minus=op.project(discr, dd_base, dd_btag, grad_t),
            time=time,
            numerical_flux_func=viscous_numerical_flux_func
        )

    vol_term = (

        # Compute the volume contribution of the viscous flux terms
        # using field values on the quadrature grid
        viscous_flux(state=quadrature_state,
                     # Interpolate gradients to the quadrature grid
                     grad_cv=op.project(discr, dd_base, dd_vol, grad_cv),
                     grad_t=op.project(discr, dd_base, dd_vol, grad_t))

        # Compute the volume contribution of the inviscid flux terms
        # using field values on the quadrature grid
        - inviscid_flux(state=quadrature_state)
    )

    bnd_term = (

        # All surface contributions from the viscous fluxes
        (
            # Domain boundary contributions for the viscous terms
            sum(fvisc_divergence_flux_boundary(btag, boundary_states[btag])
                for btag in boundary_states)

            # Interior interface contributions for the viscous terms
            + sum(fvisc_divergence_flux_interior(state_pair,
                                                 grad_cv_pair,
                                                 grad_t_pair)
                  for state_pair, grad_cv_pair, grad_t_pair in zip(
                      interior_state_pairs, grad_cv_interior_pairs,
                      grad_t_interior_pairs))
        )

        # All surface contributions from the inviscid fluxes
        - (
            # Domain boundary contributions for the inviscid terms
            sum(finv_divergence_flux_boundary(btag, boundary_states[btag])
                for btag in boundary_states)

            # Interior interface contributions for the inviscid terms
            + sum(finv_divergence_flux_interior(tpair)
                  for tpair in interior_state_pairs)
        )
    )

    # NS RHS
    return div_operator(discr, dd_vol, dd_faces, vol_term, bnd_term)
