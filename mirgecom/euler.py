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

from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux
)
from mirgecom.fluid import make_conserved
from mirgecom.operators import div_operator

from grudge.trace_pair import (
    TracePair,
    local_interior_trace_pair,
    cross_rank_trace_pairs
)
from grudge.dof_desc import DOFDesc, as_dofdesc

import grudge.op as op


def euler_operator(
        discr, eos, boundaries, cv, time=0.0, dv=None, quadrature_tag=None):
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

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`
        Fluid conserved state object containing the RHS evaluation of the Euler
        flow equations.
    """
    dd_vol = DOFDesc("vol", quadrature_tag)
    dd_faces = DOFDesc("all_faces", quadrature_tag)

    def interp_to_vol_quad(u):
        return op.project(discr, "vol", dd_vol, u)


    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(discr, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(discr, local_dd, local_dd_quad, utpair.ext)
        )

    # Interpolate conserved state to the volume quadrature grid
    cv_quad = interp_to_vol_quad(cv)

    if dv is None:
        dv = eos.dependent_vars(cv_quad)

    # Volume flux calculation
    inviscid_flux_vol = inviscid_flux(discr, dv.pressure, cv_quad)

    # Surface flux calculations
    inviscid_flux_bnd = (
        # Rank-local contributions
        inviscid_facial_flux(
            discr,
            eos=eos,
            cv_tpair=interp_to_surf_quad(local_interior_trace_pair(discr, cv))
        )
        # Cross-rank (across parallel partitions) contributions
        + sum(
            inviscid_facial_flux(
                discr,
                eos=eos,
                cv_tpair=interp_to_surf_quad(tpair)
            ) for tpair in cross_rank_trace_pairs(discr, cv)
        )
        # Contributions from boundary conditions
        + sum(
            boundaries[btag].inviscid_divergence_flux(
                discr,
                btag=as_dofdesc(btag).with_discr_tag(quadrature_tag),
                cv=cv,
                eos=eos,
                time=time
            ) for btag in boundaries
        )
    )
    return -div_operator(
        discr,
        dd_vol,
        dd_faces,
        inviscid_flux_vol,
        inviscid_flux_bnd
    )


def inviscid_operator(discr, eos, boundaries, q, t=0.0):
    """Interface :function:`euler_operator` with backwards-compatible API."""
    from warnings import warn
    warn("Do not call inviscid_operator; it is now called euler_operator. This"
         "function will disappear August 1, 2021", DeprecationWarning, stacklevel=2)
    return euler_operator(discr, eos, boundaries, make_conserved(discr.dim, q=q), t)


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
