r""":mod:`mirgecom.artificial viscosity` Artificial viscocity for Euler.

Euler Equations with artificial viscosity term:

.. math::

    \partial_t \mathbf{Q} = \nabla\cdot\mathbf{F}^I +
    \nabla\cdot{\varepsilon\nabla\mathbf{Q}}

where:

-  fluid state: $\mathbf{Q} = [\rho, \rho{E}, \rho\mathbf{V}, \rho\mathbf{Y}]$
-  inviscid fluxes: $\mathbf{F}^I
-  artifical viscosity coefficient: $\varepsilon$

To evalutate the second order derivative the problem is recast as a set of first
 order problems:

.. math::

    \partial_t{\mathbf{Q}} = \nabla\cdot\mathbf{F}^I + \nabla\cdot\mathbf{R}
    \mathbf{R} = \varepsilon\nabla\mathbf{Q}

where $\mathbf{R}$ is an intermediate variable.

Evalutes the smoothness indicator of Persson:

.. math::

    S_e = \frac{\langle u_{N_p} - u_{N_{p-1}}, u_{N_p} -
        u_{N_{p-1}}\rangle_e}{\langle u_{N_p}, u_{N_p} \rangle_e}

where:
- $S_e$ is the smoothness indicator
- $u_{N_p}$ is the modal representation of the solution at the current polynomial
  order
- $u_{N_{p-1}}$ is the truncated modal represention to the polynomial order $p-1$
- The $L_2$ inner product on an element is denoted $\langle \cdot,\cdot \rangle_e$

The elementwise viscoisty is then calculated:

.. math::

    \varepsilon_e =
        \begin{cases}
            0, & s_e < s_0 - \kappa \\
            \frac{1}{2}\left( 1 + \sin \frac{\pi(s_e - s_0)}{2 \kappa} \right ),
            & s_0-\kappa \le s_e \le s_0+\kappa \\
            1, & s_e > s_0+\kappa
        \end{cases}

where:
- $\varepsilon_e$ is the element viscosity
- $s_e = \log_{10}{S_e} \sim 1/p^4$ is the smoothness indicator
- $s_0$ is a reference smoothness value
- $\kappa$ controls the width of the transition between 0 to 1

Smoothness Indicator Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: smoothness_indicator

AV RHS Evaluation
^^^^^^^^^^^^^^^^^

.. autofunction:: artificial_viscosity
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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

import numpy as np
import loopy as lp
from modepy import vandermonde
from pytools.obj_array import obj_array_vectorize
from meshmode.dof_array import thaw, DOFArray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair
from mirgecom.fluid import (
    split_conserved,
    join_conserved_vectors
)


# def _facial_flux_r(discr, q_tpair):
def _facial_flux_q(discr, q_tpair):

    q_int = q_tpair.int
    actx = q_int[0].array_context

    flux_dis = q_tpair.avg

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_out = flux_dis * normal

    return discr.project(q_tpair.dd, "all_faces", flux_out)


# def _facial_flux_q(discr, q_tpair):
def _facial_flux_r(discr, r_tpair):
    r_int = r_tpair.int
    actx = r_int[0].array_context

    normal = thaw(actx, discr.normal(r_tpair.dd))

    # flux_out = np.dot(r_tpair.avg, normal)
    flux_out = r_tpair.avg @ normal

    return discr.project(r_tpair.dd, "all_faces", flux_out)


def artificial_viscosity(discr, t, eos, boundaries, q, alpha, **kwargs):
    r"""Compute artifical viscosity for the euler equations.

    Calculates
    ----------
    numpy.ndarray
        The right-hand-side term for artificial viscosity.

        .. math::

            \dot{\nabla\cdot{\varepsilon\nabla\mathbf{Q}}}

    Parameters
    ----------
    q
        State array which expects the quantity to be limited on to be listed
        first in the array. For the Euler equations this could be the canonical
        conserved variables (mass, energy, mometum) for the fluid along with a
        vector of species masses for multi-component fluids.

    boundaries
        Dicitionary of boundary functions, one for each valid boundary tag

    t
        Time

    alpha
       The maximum artifical viscosity coeffiecent to be applied

    eos: mirgecom.eos.GasEOS
       Only used as a pass through to the boundary conditions.

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF Arrays representing the RHS associated
        with the artificial viscosity application.
    """
    dim = discr.dim
    cv = split_conserved(dim, q)

    # Get smoothness indicator based on fluid mass density
    indicator = smoothness_indicator(discr, cv.mass, **kwargs)

    grad_q_vol = join_conserved_vectors(dim, obj_array_vectorize(discr.weak_grad, q))

    q_flux_int = _facial_flux_q(discr, q_tpair=interior_trace_pair(discr, q))
    q_flux_pb = sum(_facial_flux_q(discr, q_tpair=pb_tpair)
                    for pb_tpair in cross_rank_trace_pairs(discr, q))
    q_flux_db = 0
    for btag in boundaries:
        q_tpair = TracePair(
            btag,
            interior=discr.project("vol", btag, q),
            exterior=boundaries[btag].exterior_soln(discr, btag=btag,
                                                    t=t, q=q, eos=eos))
        q_flux_db = q_flux_db + _facial_flux_q(discr, q_tpair=q_tpair)
    q_bnd_flux = q_flux_int + q_flux_pb + q_flux_db

    # Compute R
    r = discr.inverse_mass(
        -alpha * indicator * (grad_q_vol - discr.face_mass(q_bnd_flux))
    )

    div_r_vol = discr.weak_div(r)
    r_flux_int = _facial_flux_r(discr, r_tpair=interior_trace_pair(discr, r))
    r_flux_pb = sum(_facial_flux_r(discr, r_tpair=pb_tpair)
                    for pb_tpair in cross_rank_trace_pairs(discr, r))
    r_flux_db = 0
    for btag in boundaries:
        r_tpair = TracePair(
            btag,
            interior=discr.project("vol", btag, r),
            exterior=boundaries[btag].exterior_grad_q(discr, btag, t=t,
                                                      grad_q=r, eos=eos))
        r_flux_db = r_flux_db + _facial_flux_r(discr, r_tpair=r_tpair)
    r_flux_bnd = r_flux_int + r_flux_pb + r_flux_db

    # Return the rhs contribution
    return discr.inverse_mass(-div_r_vol + discr.face_mass(r_flux_bnd))


def linear_operator_kernel():
    """Apply linear operator to all elements."""
    from meshmode.array_context import make_loopy_program

    knl = make_loopy_program(
        """{[iel,idof,j]:
        0<=iel<nelements and
        0<=idof<ndiscr_nodes_out and
        0<=j<ndiscr_nodes_in}""",
        "result[iel,idof] = sum(j, mat[idof, j] * vec[iel, j])",
        name="modal_decomp",
    )
    knl = lp.tag_array_axes(knl, "mat", "stride:auto,stride:auto")
    return knl


def compute_smoothness_indicator():
    """Compute the smoothness indicator for all elements."""
    from meshmode.array_context import make_loopy_program

    knl = make_loopy_program(
        """{[iel,idof,j,k]:
        0<=iel<nelements and
        0<=idof<ndiscr_nodes_out and
        0<=j<ndiscr_nodes_in and
        0<=k<ndiscr_nodes_in}""",
        "result[iel,idof] = "
        "sum(k,vec[iel,k]*vec[iel,k]*modes[k])/sum(j,"
        " vec[iel,j]*vec[iel,j]+1.0e-12/ndiscr_nodes_in)",
        name="smooth_comp",
    )
    return knl


def smoothness_indicator(discr, u, kappa=1.0, s0=-6.0):
    """Calculate the smoothness indicator.

    Parameters
    ----------
    u
        A DOF Array of the field that is used to calculate the
        smoothness indicator.

    kappa
        A optional argument that sets the controls the width of the
        transition between 0 to 1.
    s0
        A optional argument that sets the smoothness level to limit
        on. Logical values are [0,-infinity) where -infinity results in
        all cells being tagged and 0 results in none.

    Returns
    -------
    meshmode.dof_array.DOFArray
        A DOF Array containing elementwise constant values between 0 and 1
        which indicate the smoothness of a given element.
    """
    assert isinstance(u, DOFArray)

    def get_kernel():
        return linear_operator_kernel()

    def get_indicator():
        return compute_smoothness_indicator()

    # Convert to modal solution representation
    actx = u.array_context
    uhat = discr.empty(actx, dtype=u.entry_dtype)
    for group in discr.discr_from_dd("vol").groups:
        vander = vandermonde(group.basis(), group.unit_nodes)
        vanderm1 = np.linalg.inv(vander)
        actx.call_loopy(
            get_kernel(),
            mat=actx.from_numpy(vanderm1),
            result=uhat[group.index],
            vec=u[group.index],
        )

    # Compute smoothness indicator value
    indicator = discr.empty(actx, dtype=u.entry_dtype)
    for group in discr.discr_from_dd("vol").groups:
        mode_ids = group.mode_ids()
        modes = len(mode_ids)
        order = group.order
        highest_mode = np.ones(modes)
        for mode_index, mode_id in enumerate(mode_ids):
            highest_mode[mode_index] = highest_mode[mode_index] * (
                sum(mode_id) == order
            )

        actx.call_loopy(
            get_indicator(),
            result=indicator[group.index],
            vec=uhat[group.index],
            modes=actx.from_numpy(highest_mode),
        )
    indicator = actx.np.log10(indicator + 1.0e-12)

    # Compute artificial viscosity percentage based on indicator and set parameters
    yesnol = indicator > (s0 - kappa)
    yesnou = indicator > (s0 + kappa)
    sin_indicator = actx.np.where(
        yesnol,
        0.5 * (1.0 + actx.np.sin(np.pi * (indicator - s0) / (2.0 * kappa))),
        0.0 * indicator,
    )
    indicator = actx.np.where(yesnou, 1.0 + 0.0 * indicator, sin_indicator)

    return indicator
