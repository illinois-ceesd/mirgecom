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
    inviscid_facial_flux,
    flux_chandrashekar,
    entropy_stable_facial_flux
)
from mirgecom.fluid import (
    make_conserved,
    conservative_to_entropy_vars,
    entropy_to_conservative_vars
)
from mirgecom.operators import div_operator

from functools import partial

from grudge.trace_pair import (
    TracePair,
    interior_trace_pair,
    cross_rank_trace_pairs
)
from grudge.dof_desc import DOFDesc
from grudge.projection import volume_quadrature_project
from grudge.interpolation import \
    volume_and_surface_quadrature_interpolation
from grudge.flux_differencing import volume_flux_differencing

import grudge.op as op


def euler_operator(discr, eos, boundaries, cv, time=0.0):
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
    inviscid_flux_vol = inviscid_flux(discr, eos, cv)
    inviscid_flux_bnd = (
        inviscid_facial_flux(discr, eos=eos, cv_tpair=interior_trace_pair(discr, cv))
        + sum(inviscid_facial_flux(
            discr, eos=eos, cv_tpair=TracePair(
                part_tpair.dd, interior=make_conserved(discr.dim, q=part_tpair.int),
                exterior=make_conserved(discr.dim, q=part_tpair.ext)))
              for part_tpair in cross_rank_trace_pairs(discr, cv.join()))
        + sum(boundaries[btag].inviscid_boundary_flux(discr, btag=btag, cv=cv,
                                                      eos=eos, time=time)
              for btag in boundaries)
    )
    q = -div_operator(discr, inviscid_flux_vol.join(), inviscid_flux_bnd.join())
    return make_conserved(discr.dim, q=q)


def inviscid_operator(discr, eos, boundaries, q, t=0.0):
    """Interface :function:`euler_operator` with backwards-compatible API."""
    from warnings import warn
    warn("Do not call inviscid_operator; it is now called euler_operator. This"
         "function will disappear August 1, 2021", DeprecationWarning, stacklevel=2)
    return euler_operator(discr, eos, boundaries, make_conserved(discr.dim, q=q), t)


def entropy_stable_euler_operator(dcoll, eos, boundaries, cv, time=0.0, qtag=None):
    """todo.
    """
    dq = DOFDesc("vol", qtag)
    df = DOFDesc("all_faces", qtag)

    # Convert to projected entropy variables: ev_q = V_h P_q v(cv_q)
    proj_entropy_vars = volume_quadrature_project(
        dcoll,
        dq,
        # Map to entropy variables: v(u_q)
        conservative_to_entropy_vars(
            eos,
            # Interpolate state to vol quad grid: cv_q = V_q cv
            op.project(dcoll, "vol", dq, cv)
        )
    )


    def modified_conservedvars(proj_ev):
        """Converts the projected entropy variables into
        conserved variables on the quadrature (vol + surf) grid.
        """
        return entropy_to_conservative_vars(
            eos,
            # Interpolate projected entropy variables to
            # volume + surface quadrature grids
            volume_and_surface_quadrature_interpolation(
                dcoll, dq, df, proj_ev
            )
        )


    # Compute volume derivatives using flux differencing
    inviscid_flux_vol = volume_flux_differencing(
        dcoll,
        partial(flux_chandrashekar, dcoll, eos),
        dq, df,
        modified_conservedvars(proj_entropy_vars)
    )


    def modified_conservedvars_tpair(tpair):
        """Takes a trace pair containing the projected entropy variables
        and converts them into conserved variables on the quadrature
        grid.
        """
        dd_intfaces = tpair.dd
        dd_intfaces_quad = dd_intfaces.with_discr_tag(qtag)
        # Interpolate entropy variables to the surface quadrature grid
        vtilde_tpair = op.project(dcoll, dd_intfaces, dd_intfaces_quad, tpair)
        return TracePair(
            dd_intfaces_quad,
            # Convert interior and exterior states to conserved variables
            interior=entropy_to_conservative_vars(eos, vtilde_tpair.int),
            exterior=entropy_to_conservative_vars(eos, vtilde_tpair.ext)
        )

    # Compute interface terms
    inviscid_flux_bnd = (
        entropy_stable_facial_flux(
            dcoll,
            eos=eos,
            cv_tpair=modified_conservedvars_tpair(
                interior_trace_pair(dcoll, proj_entropy_vars)
            )
        )
        + sum(
            entropy_stable_facial_flux(
                dcoll,
                eos=eos,
                cv_tpair=modified_conservedvars_tpair(part_tpair)
            ) for part_tpair in cross_rank_trace_pairs(dcoll, proj_entropy_vars)
        )
        + sum(
            boundaries[btag].inviscid_boundary_flux(
                dcoll,
                btag=btag,
                cv=cv,
                eos=eos,
                time=time,
                quad_tag=qtag
            ) for btag in boundaries
        )
    )

    return op.inverse_mass(
        dcoll,
        -inviscid_flux_vol - op.face_mass(dcoll, df, inviscid_flux_bnd)
    )


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
