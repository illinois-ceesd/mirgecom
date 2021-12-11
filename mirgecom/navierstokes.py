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

import numpy as np  # noqa
from grudge.symbolic.primitives import TracePair
from grudge.trace_pair import interior_trace_pairs

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
from mirgecom.fluid import make_conserved
from mirgecom.operators import (
    div_operator, grad_operator
)
from arraycontext import thaw


def ns_operator(discr, gas_model, state, boundaries, time=0.0,
                inviscid_numerical_flux_func=inviscid_flux_rusanov,
                gradient_numerical_flux_func=gradient_flux_central,
                viscous_numerical_flux_func=viscous_flux_central):
    r"""Compute RHS of the Navier-Stokes equations.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)

    Parameters
    ----------
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid solution

    dv: :class:`~mirgecom.eos.EOSDependentVars`
        Fluid solution-dependent quantities (e.g. pressure, temperature)

    boundaries
        Dictionary of boundary functions, one for each valid btag

    time
        Time

    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
        Implementing the transport properties including heat conductivity,
        and species diffusivities type(mirgecom.transport.TransportModel).

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF arrays representing the RHS of the
        Navier-Stokes equations.
    """
    if not state.is_viscous:
        raise ValueError("Navier-Stokes operator expects viscous gas model.")

    actx = state.array_context

    from mirgecom.gas_model import project_fluid_state
    boundary_states = {btag:
                       project_fluid_state(discr, btag, state, gas_model)
                       for btag in boundaries}

    cv_interior_pairs = interior_trace_pairs(discr, state.cv)

    tseed_interior_pairs = \
        interior_trace_pairs(discr, state.cv) if state.is_mixture else None

    from mirgecom.gas_model import make_fluid_state_trace_pairs
    interior_state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs, gas_model,
                                                        tseed_interior_pairs)

    # Operator-independent boundary flux interface
    def _elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                    interior_trace_pairs, boundary_states):
        return (sum(compute_interior_flux(tpair)
                      for tpair in interior_trace_pairs)
                + sum(compute_boundary_flux(btag, boundary_states[btag])
                      for btag in boundary_states))

    # Data-independent grad flux for faces
    def _grad_flux_interior(q_pair):
        normal = thaw(discr.normal(q_pair.dd), actx)
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = gradient_numerical_flux_func(q_pair, normal)
        return discr.project(q_pair.dd, "all_faces", flux_weak)

    # CV-specific boundary flux for grad operator
    def _cv_grad_flux_bnd(btag, boundary_state):
        return boundaries[btag].cv_gradient_flux(
            discr, btag=btag, gas_model=gas_model, state_minus=boundary_state,
            time=time, numerical_flux_func=gradient_numerical_flux_func
        )

    cv_flux_bnd = _elbnd_flux(discr, _grad_flux_interior, _cv_grad_flux_bnd,
                              cv_interior_pairs, boundary_states)

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = make_conserved(state.dim, q=grad_operator(discr, state.cv.join(),
                                                        cv_flux_bnd.join()))

    grad_cv_interior_pairs = interior_trace_pairs(discr, grad_cv)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # Capture the temperature for the interior faces for grad(T) calc
    # Note this is *all interior faces*, including partition boundaries
    # due to the use of *interior_state_pairs*.
    t_interior_pairs = [TracePair(state_pair.dd,
                                  interior=state_pair.int.temperature,
                                  exterior=state_pair.ext.temperature)
                        for state_pair in interior_state_pairs]

    def _t_grad_flux_bnd(btag, boundary_state):
        return boundaries[btag].temperature_gradient_flux(
            discr, btag=btag, gas_model=gas_model,
            state_minus=boundary_state, time=time)

    t_flux_bnd = _elbnd_flux(discr, _grad_flux_interior, _t_grad_flux_bnd,
                             t_interior_pairs, boundary_states)

    # Fluxes in-hand, compute the gradient of temperature and mpi exchange it
    grad_t = grad_operator(discr, state.temperature, t_flux_bnd)
    grad_t_interior_pairs = interior_trace_pairs(discr, grad_t)

    # inviscid flux divergence-specific flux function for interior faces
    def finv_divergence_flux_interior(state_pair):
        return inviscid_facial_flux(
            discr, gas_model=gas_model, state_pair=state_pair,
            numerical_flux_func=inviscid_numerical_flux_func)

    # inviscid part of bcs applied here
    def finv_divergence_flux_boundary(btag, boundary_state):
        return boundaries[btag].inviscid_divergence_flux(
            discr, btag=btag, gas_model=gas_model, state_minus=boundary_state,
            time=time, numerical_flux_func=inviscid_numerical_flux_func)

    # glob the inputs together in a tuple to use the _elbnd_flux wrapper
    viscous_states_int_bnd = [
        (state_pair, grad_cv_pair, grad_t_pair)
        for state_pair, grad_cv_pair, grad_t_pair in
        zip(interior_state_pairs, grad_cv_interior_pairs, grad_t_interior_pairs)]

    # viscous fluxes across interior faces (including partition and periodic bnd)
    def fvisc_divergence_flux_interior(tpair_tuple):
        state_pair = tpair_tuple[0]
        grad_cv_pair = tpair_tuple[1]
        grad_t_pair = tpair_tuple[2]
        return viscous_facial_flux(discr=discr, gas_model=gas_model,
                                   state_pair=state_pair, grad_cv_pair=grad_cv_pair,
                                   grad_t_pair=grad_t_pair,
                                   numerical_flux_func=viscous_numerical_flux_func)

    # viscous part of bcs applied here
    def fvisc_divergence_flux_boundary(btag, boundary_state):
        grad_cv_minus = discr.project("vol", btag, grad_cv)
        grad_t_minus = discr.project("vol", btag, grad_t)
        return boundaries[btag].viscous_divergence_flux(
            discr=discr, btag=btag, gas_model=gas_model,
            state_minus=boundary_state,
            grad_cv_minus=grad_cv_minus, grad_t_minus=grad_t_minus, time=time,
            numerical_flux_func=viscous_numerical_flux_func)

    vol_term = (
        viscous_flux(state=state, grad_cv=grad_cv, grad_t=grad_t)
        - inviscid_flux(state=state)
    )

    bnd_term = (
        _elbnd_flux(
            discr, fvisc_divergence_flux_interior, fvisc_divergence_flux_boundary,
            viscous_states_int_bnd, boundary_states)
        - _elbnd_flux(
            discr, finv_divergence_flux_interior, finv_divergence_flux_boundary,
            interior_state_pairs, boundary_states)
    )

    # NS RHS
    return div_operator(discr, vol_term, bnd_term)
