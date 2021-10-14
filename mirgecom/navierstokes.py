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

from typing import Tuple
import numpy as np  # noqa

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs,
    interior_trace_pair,
    cross_rank_trace_pairs
)
from grudge.dof_desc import DD_VOLUME, DOFDesc, DISCR_TAG_BASE

from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux
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


def ns_operator(discr, eos, boundaries, cv, t=0.0, quad_tag=None):
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

    boundaries
        Dictionary of boundary functions, one for each valid btag

    t
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
    if quad_tag is None:
        dd = DD_VOLUME
    else:
        dd = DOFDesc("vol", quad_tag)

    dd_allfaces = dd.with_dtag("all_faces")

    def to_quad(a, from_dd):
        return discr.project(from_dd, dd.with_dtag(from_dd.domain_tag), a)

    dim = discr.dim
    actx = cv.array_context
    cv_quad = discr.project("vol", dd, cv)

    def _elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                    int_tpair, xrank_pairs, boundaries):
        # cfoo = compute_interior_flux(int_tpair)
        # pfoo = sum(compute_interior_flux(part_tpair)
        #         for part_tpair in xrank_pairs)
        # bfoo = sum(compute_boundary_flux(btag) for btag in boundaries)
        # import ipdb; ipdb.set_trace()
        return (
            compute_interior_flux(int_tpair)
            + sum(compute_interior_flux(part_tpair)
                for part_tpair in xrank_pairs)
            + sum(compute_boundary_flux(btag) for btag in boundaries)
        )

    def scalar_flux_interior(int_tpair):
        # FIXME: interpolated `TracePair` objects don't get
        # their *dd* updated
        dd_tgt = int_tpair.dd.with_discr_tag(quad_tag)
        int_tpair = TracePair(
            dd_tgt,
            interior=discr.project(int_tpair.dd, dd_tgt, int_tpair.int),
            exterior=discr.project(int_tpair.dd, dd_tgt, int_tpair.ext)
        )
        normal = thaw(actx, discr.normal(dd_tgt))

        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = gradient_flux_central(int_tpair, normal)
        return discr.project(dd_tgt, dd_allfaces, flux_weak)

    def get_q_flux_bnd(btag):
        return boundaries[btag].q_boundary_flux(
            discr, btag=btag, cv=cv, eos=eos, time=t, quad_tag=quad_tag
        )

    cv_int_tpair = interior_trace_pair(discr, cv)
    cv_part_pairs = [
        TracePair(part_tpair.dd,
                  interior=make_conserved(dim, q=part_tpair.int),
                  exterior=make_conserved(dim, q=part_tpair.ext))
        for part_tpair in cross_rank_trace_pairs(discr, cv.join())]
    cv_flux_bnd = _elbnd_flux(discr, scalar_flux_interior, get_q_flux_bnd,
                              cv_int_tpair, cv_part_pairs, boundaries)

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = make_conserved(dim, q=grad_operator(discr,
                                                  cv_quad.join(),
                                                  cv_flux_bnd.join(),
                                                  quad_tag=quad_tag))
    grad_cv_quad = discr.project("vol", dd, grad_cv)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # - now computed, *not* communicated
    def get_t_flux_bnd(btag):
        return boundaries[btag].t_boundary_flux(discr, btag=btag, cv=cv, eos=eos,
                                                time=t, quad_tag=quad_tag)

    gas_t = discr.project("vol", dd, eos.temperature(cv))
    t_int_tpair = TracePair("int_faces",
                            interior=eos.temperature(cv_int_tpair.int),
                            exterior=eos.temperature(cv_int_tpair.ext))
    t_part_pairs = [
        TracePair(part_tpair.dd,
                  interior=eos.temperature(part_tpair.int),
                  exterior=eos.temperature(part_tpair.ext))
        for part_tpair in cv_part_pairs]
    t_flux_bnd = _elbnd_flux(discr, scalar_flux_interior, get_t_flux_bnd,
                             t_int_tpair, t_part_pairs, boundaries)
    grad_t = grad_operator(discr, gas_t, t_flux_bnd, quad_tag=quad_tag)
    grad_t_quad = discr.project("vol", dd, grad_t)

    # inviscid parts
    def finv_interior_face(cv_tpair):
        dd_tgt = cv_tpair.dd.with_discr_tag(quad_tag)
        cv_tpair = TracePair(
            dd_tgt,
            interior=discr.project(cv_tpair.dd, dd_tgt, cv_tpair.int),
            exterior=discr.project(cv_tpair.dd, dd_tgt, cv_tpair.ext)
        )
        return inviscid_facial_flux(discr, eos=eos, cv_tpair=cv_tpair)

    # inviscid part of bcs applied here
    def finv_domain_boundary(btag):
        return boundaries[btag].inviscid_boundary_flux(
            discr, btag=btag, eos=eos, cv=cv, time=t, quad_tag=quad_tag
        )

    # viscous parts
    s_int_pair = interior_trace_pair(discr, grad_cv)
    s_part_pairs = [TracePair(xrank_tpair.dd,
                             interior=make_conserved(dim, q=xrank_tpair.int),
                             exterior=make_conserved(dim, q=xrank_tpair.ext))
                    for xrank_tpair in cross_rank_trace_pairs(discr, grad_cv.join())]
    delt_int_pair = interior_trace_pair(discr, grad_t)
    delt_part_pairs = cross_rank_trace_pairs(discr, grad_t)
    num_partition_interfaces = len(cv_part_pairs)

    # glob the inputs together in a tuple to use the _elbnd_flux wrapper
    visc_part_inputs = [
        (cv_part_pairs[bnd_index], s_part_pairs[bnd_index],
         delt_part_pairs[bnd_index])
        for bnd_index in range(num_partition_interfaces)]

    # viscous fluxes across interior faces (including partition and periodic bnd)
    def fvisc_interior_face(tpair_tuple):
        cv_pair_int = tpair_tuple[0]
        s_pair_int = tpair_tuple[1]
        dt_pair_int = tpair_tuple[2]

        dd_tgt = cv_pair_int.dd.with_discr_tag(quad_tag)
        cv_pair_int = TracePair(
            dd_tgt,
            interior=discr.project(cv_pair_int.dd, dd_tgt, cv_pair_int.int),
            exterior=discr.project(cv_pair_int.dd, dd_tgt, cv_pair_int.ext)
        )
        s_pair_int = TracePair(
            dd_tgt,
            interior=discr.project(s_pair_int.dd, dd_tgt, s_pair_int.int),
            exterior=discr.project(s_pair_int.dd, dd_tgt, s_pair_int.ext)
        )
        dt_pair_int = TracePair(
            dd_tgt,
            interior=discr.project(dt_pair_int.dd, dd_tgt, dt_pair_int.int),
            exterior=discr.project(dt_pair_int.dd, dd_tgt, dt_pair_int.ext)
        )
        return viscous_facial_flux(discr, eos, cv_pair_int, s_pair_int, dt_pair_int)

    # viscous part of bcs applied here
    def visc_bnd_flux(btag):
        return boundaries[btag].viscous_boundary_flux(discr, btag, eos=eos,
                                                      cv=cv, grad_cv=grad_cv,
                                                      grad_t=grad_t, time=t,
                                                      quad_tag=quad_tag)

    vol_term = (
        viscous_flux(discr, eos=eos,
                     cv=cv_quad,
                     grad_cv=grad_cv_quad,
                     grad_t=grad_t_quad)
        - inviscid_flux(discr, eos=eos, cv=cv_quad)
    ).join()

    bnd_term = (
        _elbnd_flux(
            discr, fvisc_interior_face, visc_bnd_flux,
            (cv_int_tpair, s_int_pair, delt_int_pair),
            visc_part_inputs, boundaries)
        - _elbnd_flux(
            discr, finv_interior_face, finv_domain_boundary, cv_int_tpair,
            cv_part_pairs, boundaries)
    ).join()

    # NS RHS
    return make_conserved(
        dim, q=div_operator(discr, vol_term, bnd_term, quad_tag=quad_tag)
    )
