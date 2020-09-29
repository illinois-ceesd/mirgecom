r""":mod:`mirgecom.heat` computes the rhs of the heat equation.

Heat equation:

.. math::

    \partial_t \mathbf{u} = \alpha\nabla^2\mathbf{u}

.. autofunction:: heat_operator
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

import math
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from grudge.symbolic.primitives import TracePair
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs


def _q_flux(discr, alpha, u_tpair):
    actx = u_tpair.int[0].array_context

    normal = thaw(actx, discr.normal(u_tpair.dd))

    flux_weak = math.sqrt(alpha)*u_tpair.avg*normal

    return discr.project(u_tpair.dd, "all_faces", flux_weak)


def _u_flux(discr, alpha, q_tpair):
    actx = q_tpair.int[0].array_context

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_weak = math.sqrt(alpha)*make_obj_array([np.dot(q_tpair.avg, normal)])

    return discr.project(q_tpair.dd, "all_faces", flux_weak)


def heat_operator(discr, alpha, w):
    """Compute the RHS of the heat equation.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    alpha: float
        the (constant) diffusivity
    w: numpy.ndarray
        an object array containing the DOF array state vector

    Returns
    -------
    numpy.ndarray
        an object array containing the DOF array RHS vector
    """
    u = w

    dir_u = discr.project("vol", BTAG_ALL, u)

    q = discr.inverse_mass(
        -math.sqrt(alpha)*discr.weak_grad(u[0])
        +  # noqa: W504
        discr.face_mass(
            _q_flux(discr, alpha=alpha, u_tpair=interior_trace_pair(discr, u))
            + _q_flux(discr, alpha=alpha,
                u_tpair=TracePair(BTAG_ALL, interior=dir_u, exterior=-dir_u))
            + sum(
                _q_flux(discr, alpha=alpha, u_tpair=tpair)
                for tpair in cross_rank_trace_pairs(discr, u)
            )
        ))

    dir_q = discr.project("vol", BTAG_ALL, q)

    return (
        discr.inverse_mass(
            make_obj_array([-math.sqrt(alpha)*discr.weak_div(q)])
            +  # noqa: W504
            discr.face_mass(
                _u_flux(discr, alpha=alpha, q_tpair=interior_trace_pair(discr, q))
                + _u_flux(discr, alpha=alpha,
                    q_tpair=TracePair(BTAG_ALL, interior=dir_q, exterior=dir_q))
                + sum(
                    _u_flux(discr, alpha=alpha, q_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, q))
                )
            )
        )
