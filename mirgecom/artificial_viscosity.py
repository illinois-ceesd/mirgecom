r""":mod:`mirgecom.artificial viscosity` Artificial viscocity for Euler.

Artificial viscosity for the Euler Equations:

.. math::

    \partial_t \mathbf{R} = \nabla\cdot{\varepsilon\nabla\mathbf{R}}

where:

-  state $\mathbf{R} = [\rho, \rho{E}, \rho\vec{V}, \rho{Y}_\alpha]$
-  artifical viscosity coefficient $\varepsilon$

To evalutate the second order derivative the problem is recast as a set of first
 order problems:

.. math::

    \partial_t \mathbf{R} = \nabla\cdot{\mathbf{Q}}
    \mathbf{Q} = \varepsilon\nabla\mathbf{R}

where $\mathbf{Q}$ is an intermediate variable.

RHS Evaluation
^^^^^^^^^^^^^^

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
from pytools.obj_array import (
    obj_array_vectorize,
    obj_array_vectorize_n_args,
)
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair
from mirgecom.tag_cells import smoothness_indicator


def _facial_flux_r(discr, q_tpair):

    actx = q_tpair.int.array_context
    flux_dis = q_tpair.avg
    normal = thaw(actx, discr.normal(q_tpair.dd))
    flux_out = flux_dis * normal

    return discr.project(q_tpair.dd, "all_faces", flux_out)
  

def _facial_flux_q(discr, q_tpair):
    actx = q_tpair[0].int.array_context
    normal = thaw(actx, discr.normal(q_tpair.dd))
    flux_out = np.dot(q_tpair.avg, normal)

    return discr.project(q_tpair.dd, "all_faces", flux_out)


def artificial_viscosity(discr, t, eos, boundaries, r, alpha, **kwargs):
    r"""Compute artifical viscosity for the euler equations.

    Calculates
    ----------
    numpy.ndarray
        The right-hand-side for artificial viscosity for the euler equations.

        .. math::

            \dot{\nabla\cdot{\varepsilon\nabla\mathbf{R}}}

    Parameters
    ----------
    r
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
    # Get smoothness indicator
    indicator = smoothness_indicator(discr, r[0], **kwargs)

    dflux_r = obj_array_vectorize(discr.weak_grad, r)

    def my_facialflux_r_interior(q):
        qin = interior_trace_pair(discr, q)
        return _facial_flux_r(discr, q_tpair=qin)

    iff_r = obj_array_vectorize(my_facialflux_r_interior, r)

    def my_facialflux_r_partition(q):
        qin = cross_rank_trace_pairs(discr, q)
        return sum(_facial_flux_r(discr, q_tpair=part_pair) for part_pair in qin)

    pbf_r = obj_array_vectorize(my_facialflux_r_partition, r)

    dbf_r = np.zeros_like(iff_r)
    for btag in boundaries:

        def my_facialflux_r_boundary(sol_ext, sol_int):
            q_tpair = TracePair(
                btag,
                interior=sol_int,
                exterior=sol_ext,
            )
            return _facial_flux_r(discr, q_tpair=q_tpair)

        r_ext = boundaries[btag].exterior_sol(discr, btag=btag, t=t, q=r, eos=eos)
        r_int = discr.project("vol", btag, r)
        dbf_r = dbf_r + obj_array_vectorize_n_args(
            my_facialflux_r_boundary, r_ext, r_int
        )

    # Compute q, half way done!
    q = discr.inverse_mass(
        -alpha * indicator * (dflux_r - discr.face_mass(iff_r + pbf_r + dbf_r))
    )
    dflux_q = obj_array_vectorize(discr.weak_div, q)

    def my_facialflux_q_interior(q):
        qin = interior_trace_pair(discr, q)
        iff_q = _facial_flux_q(discr, q_tpair=qin)
        return iff_q

    iff_q = obj_array_vectorize(my_facialflux_q_interior, q)

    def my_facialflux_q_partition(q):
        qin = cross_rank_trace_pairs(discr, q)
        return sum(_facial_flux_q(discr, q_tpair=part_pair) for part_pair in qin)

    pbf_q = obj_array_vectorize(my_facialflux_q_partition, q)

    dbf_q = np.zeros_like(iff_q)
    for btag in boundaries:

        def my_facialflux_q_boundary(sol_ext, sol_int):
            q_tpair = TracePair(btag, interior=sol_int, exterior=sol_ext)
            return _facial_flux_q(discr, q_tpair=q_tpair)

        q_ext = boundaries[btag].av(discr, btag=btag, t=t, q=q, eos=eos)
        q_int = discr.project("vol", btag, q)
        dbf_q = dbf_q + obj_array_vectorize_n_args(
            my_facialflux_q_boundary, q_ext, q_int
        )

    # Return the rhs contribution
    return discr.inverse_mass(-dflux_q + discr.face_mass(iff_q + pbf_q + dbf_q))
