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
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_divergence_flux
)
from mirgecom.viscous import (
    viscous_flux,
    viscous_facial_flux
)
from mirgecom.flux import (
    gradient_flux_central
)
from mirgecom.fluid import make_conserved
from mirgecom.operators import (
    div_operator, grad_operator
)
from meshmode.dof_array import thaw


def ns_operator(discr, eos, boundaries, cv, time=0.0, dv=None):
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
    dim = discr.dim
    actx = cv.array_context
    if dv is None:
        dv = eos.dependent_vars(cv)

    # These locally-defined utilities use global discr, eos
    def _make_consistent_dv_pairs(cv_pairs, temperature_pairs=None):
        if temperature_pairs is not None:
            return [TracePair(
                    cv_pair.dd,
                    interior=eos.dependent_vars(cv_pair.int, tseed_pair.int),
                    exterior=eos.dependent_vars(cv_pair.ext, tseed_pair.ext))
                    for cv_pair, tseed_pair in zip(cv_pairs, temperature_pairs)]
        else:
            return [TracePair(cv_pair.dd,
                              interior=eos.dependent_vars(cv_pair.int),
                              exterior=eos.dependent_vars(cv_pair.ext))
                    for cv_pair in cv_pairs]

    def _make_thermally_consistent_cv_pairs(cv_pairs, temperature_pairs=None):
        return cv_pairs, _make_consistent_dv_pairs(eos, cv_pairs, temperature_pairs)

    def _make_thermally_consistent_cv_on_subdiscr(btag, cv, dv):
        cv_sub = discr.project("vol", btag, cv)
        tseed_sub = None
        if cv.nspecies > 0:
            tseed_sub = discr.project("vol", btag, dv.temperature)
        return cv_sub, eos.dependent_vars(cv_sub, tseed_sub)

    def _make_persistent_boundary_cv(boundaries, cv):
        dv = eos.dependent_vars(cv)
        return {btag: _make_thermally_consistent_cv_on_subdiscr(btag, cv, dv)[0]
                for btag in boundaries}

    cv_int_pair = interior_trace_pair(discr, cv)
    cv_interior_pairs = [cv_int_pair]
    q_comm_pairs = cross_rank_trace_pairs(discr, cv.join())
    num_partition_interfaces = len(q_comm_pairs)
    cv_part_pairs = [
        TracePair(q_pair.dd,
                  interior=make_conserved(dim, q=q_pair.int),
                  exterior=make_conserved(dim, q=q_pair.ext))
        for q_pair in q_comm_pairs]
    cv_interior_pairs.extend(cv_part_pairs)
    cv_bnds = _make_persistent_boundary_cv(boundaries, cv)

    tseed_interior_pairs = None
    tseed_int_pair = interior_trace_pair(discr, dv.temperature)
    tseed_part_pairs = cross_rank_trace_pairs(discr, dv.temperature)
    tseed_interior_pairs = [tseed_int_pair]
    tseed_interior_pairs.extend(tseed_part_pairs)

    cv_interior_pairs, dv_interior_pairs = \
        _make_thermally_consistent_cv_pairs(eos, cv_interior_pairs,
                                            tseed_interior_pairs)

    # Operator-independent boundary flux interface
    def _elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                    interior_trace_pairs, boundary_states):
        return (sum(compute_interior_flux(tpair)
                      for tpair in interior_trace_pairs)
                + sum(compute_boundary_flux(btag, boundary_states[btag])
                      for btag in boundary_states))

    # Data-independent grad flux for faces
    def _grad_flux_interior(int_tpair):
        normal = thaw(actx, discr.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = gradient_flux_central(int_tpair, normal)
        return discr.project(int_tpair.dd, "all_faces", flux_weak)

    # CV-specific boundary flux for grad operator
    def _cv_grad_flux_bnd(btag, boundary_cv):
        return boundaries[btag].cv_gradient_flux(
            discr, btag=btag, eos=eos, cv_minus=boundary_cv, time=time
        )

    cv_flux_bnd = _elbnd_flux(discr, _grad_flux_interior, _cv_grad_flux_bnd,
                              cv_interior_pairs, cv_bnds)

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = make_conserved(dim, q=grad_operator(discr, cv.join(),
                                                  cv_flux_bnd.join()))

    s_int_pair = interior_trace_pair(discr, grad_cv)
    s_part_pairs = [TracePair(xrank_tpair.dd,
                             interior=make_conserved(dim, q=xrank_tpair.int),
                             exterior=make_conserved(dim, q=xrank_tpair.ext))
                    for xrank_tpair in cross_rank_trace_pairs(discr, grad_cv.join())]
    grad_cv_interior_pairs = [s_int_pair]
    grad_cv_interior_pairs.append(s_part_pairs)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)

    # Capture the temperature for the interior faces for grad(T) calc
    t_interior_pairs = [TracePair("int_faces",
                                  interior=dv_pair.int.temperature,
                                  exterior=dv_pair.ext.temperature)
                        for dv_pair in dv_interior_pairs]

    t_flux_bnd = (sum(_grad_flux_interior(tpair) for tpair in t_interior_pairs)
                  + sum(boundaries[btag].temperature_gradient_flux(
                      discr, btag=btag, eos=eos, cv_minus=cv_bnds[btag], time=time)
                      for btag in boundaries))

    # Fluxes in-hand, compute the gradient of temperature
    grad_t = grad_operator(discr, dv.temperature, t_flux_bnd)
    delt_int_pair = interior_trace_pair(discr, grad_t)
    delt_part_pairs = cross_rank_trace_pairs(discr, grad_t)
    grad_t_interior_pairs = [delt_int_pair]
    grad_t_interior_pairs.append(delt_part_pairs)

    # inviscid flux divergence-specific flux function for interior faces
    def finv_divergence_flux_interior(cv_tpair):
        return inviscid_facial_divergence_flux(discr, eos, cv_tpair=cv_tpair)

    # inviscid part of bcs applied here
    def finv_divergence_flux_boundary(btag, boundary_cv):
        return boundaries[btag].inviscid_divergence_flux(
            discr, btag=btag, eos=eos, cv_minus=boundary_cv, time=time
        )

    # glob the inputs together in a tuple to use the _elbnd_flux wrapper
    viscous_states_int_bnd = [
        (cv_pair, grad_cv_pair, grad_t_pair) 
        for cv_pair, grad_cv_pair, grad_t_pair in
        zip(cv_interior_pairs, grad_cv_interior_pairs, grad_t_interior_pairs)]

    # viscous fluxes across interior faces (including partition and periodic bnd)
    def fvisc_divergence_flux_interior(tpair_tuple):
        cv_pair_int = tpair_tuple[0]
        s_pair_int = tpair_tuple[1]
        dt_pair_int = tpair_tuple[2]
        return viscous_facial_flux(discr, eos, cv_pair_int, s_pair_int, dt_pair_int)

    # viscous part of bcs applied here
    def fvisc_divergence_flux_boundary(btag, boundary_state):
        return boundaries[btag].viscous_divergence_flux(discr, btag, eos=eos,
                                                        cv_minus=cv, grad_cv=grad_cv,
                                                        grad_t=grad_t, time=time)

    vol_term = (
        viscous_flux(discr, eos=eos, cv=cv, grad_cv=grad_cv, grad_t=grad_t)
        - inviscid_flux(discr, eos=eos, cv=cv)
    ).join()

    bnd_term = (
        _elbnd_flux(
            discr, fvisc_divergence_flux_interior, fvisc_divergence_flux_boundary,
            viscous_states_int_bnd, boundaries)
        - _elbnd_flux(
            discr, finv_divergence_flux_interior, finv_divergence_flux_boundary,
            cv_int_tpair, cv_part_pairs, boundaries)
    ).join()

    # NS RHS
    return make_conserved(dim, q=div_operator(discr, vol_term, bnd_term))
