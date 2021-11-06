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
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_divergence_flux
)
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
from grudge.trace_pair import TracePair
from mirgecom.fluid import make_conserved
from mirgecom.operators import div_operator


def _inviscid_facial_divergence_flux(discr, eos, cv_pairs, temperature_pairs):
    return sum(
        inviscid_facial_divergence_flux(
            discr, cv_tpair=cv_pair,
            dv_tpair=TracePair(
                cv_pair.dd,
                interior=eos.dependent_vars(cv_pair.int, temp_pair.int),
                exterior=eos.dependent_vars(cv_pair.ext, temp_pair.ext)))
        for cv_pair, temp_pair in zip(cv_pairs, temperature_pairs))


def _get_dv_pair(discr, eos, cv_pair, temperature_pair=None):
    if temperature_pair is not None:
        return TracePair(
            cv_pair.dd,
            interior=eos.dependent_vars(cv_pair.int, temperature_pair.int),
            exterior=eos.dependent_vars(cv_pair.ext, temperature_pair.ext)
        )
    return TracePair(cv_pair.dd,
                     interior=eos.dependent_vars(cv_pair.int),
                     exterior=eos.depdndent_vars(cv_pair.ext))


def euler_operator(discr, eos, boundaries, cv, time=0.0,
                   dv=None):
    r"""Compute RHS of the Euler flow equations.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Euler flow equations:

        .. math::

            \dot{\mathbf{q}} = - \nabla\cdot\mathbf{F} +
                (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

    Parameters
    ----------
    cv: :class:`mirgecom.fluid.ConservedVars`
        Fluid conserved state object with the conserved variables.

    boundaries
        Dictionary of boundary functions, one for each valid btag

    time
        Time

    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF arrays representing the RHS of the Euler
        flow equations.
    """
    dim = discr.dim
    if dv is None:
        dv = eos.dependent_vars(cv)

    inviscid_flux_vol = inviscid_flux(discr, dv.pressure, cv)

    cv_comm_pairs = cross_rank_trace_pairs(discr, cv.join())
    cv_part_pairs = [
        TracePair(q_pair.dd,
                  interior=make_conserved(dim, q=q_pair.int),
                  exterior=make_conserved(dim, q=q_pair.ext))
        for q_pair in cv_comm_pairs]
    cv_int_pair = interior_trace_pair(discr, cv)

    flux_pb = 0
    flux_ib = 0
    flux_db = 0
    if cv.nspecies > 0:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the inviscid flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        temp_part_pairs = cross_rank_trace_pairs(discr, dv.temperature)
        flux_pb = _inviscid_facial_divergence_flux(discr, eos, cv_part_pairs,
                                                   temp_part_pairs)

        temp_int_pair = interior_trace_pair(discr, dv.temperature)
        dv_int_pair = _get_dv_pair(discr, eos, cv_int_pair, temp_int_pair)
        flux_ib = inviscid_facial_divergence_flux(discr, cv_int_pair, dv_int_pair)
        # Domain boundaries
        for btag in boundaries:
            bnd = boundaries[btag]
            cv_minus = discr.project("vol", btag, cv)
            temp_seed = discr.project("vol", btag, dv.temperature)
            dv_minus = eos.dependent_vars(cv_minus, temp_seed)
            flux_db = (
                flux_db + bnd.inviscid_divergence_flux(
                    discr, btag, eos, cv_minus, dv_minus, time=time)
            )
    else:
        for cv_pair in cv_part_pairs:
            dv_pair = _get_dv_pair(discr, eos, cv_pair)
            flux_pb = (flux_pb
                       + inviscid_facial_divergence_flux(discr, cv_tpair=cv_pair,
                                                         dv_tpair=dv_pair))

        dv_pair = _get_dv_pair(discr, eos, cv_int_pair)
        flux_ib = inviscid_facial_divergence_flux(discr, cv_tpair=cv_int_pair,
                                                  dv_tpair=dv_pair)
        flux_db = sum(
            boundaries[btag].inviscid_divergence_flux(
                discr, btag, eos, cv_minus=discr.project("vol", btag, cv),
                dv_minus=discr.project("vol", btag, dv), time=time)
            for btag in boundaries
        )

    inviscid_flux_bnd = flux_ib + flux_db + flux_pb
    q = -div_operator(discr, inviscid_flux_vol.join(), inviscid_flux_bnd.join())
    return make_conserved(discr.dim, q=q)


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
