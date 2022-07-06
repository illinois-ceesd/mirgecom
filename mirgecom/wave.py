""":mod:`mirgecom.wave` computes the rhs of the wave equation.

.. autofunction:: wave_operator
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees"
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
import numpy.linalg as la  # noqa
from pytools.obj_array import flat_obj_array
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.trace_pair import TracePair, interior_trace_pairs
import grudge.op as op


def _flux(discr, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    actx = w_tpair.int[0].array_context
    normal = actx.thaw(discr.normal(w_tpair.dd))

    flux_weak = flat_obj_array(
        np.dot(v.avg, normal),
        normal*u.avg,
        )

    # upwind
    flux_weak += flat_obj_array(
        0.5*(u.ext-u.int),
        0.5*normal*np.dot(normal, v.ext-v.int),
        )

    return op.project(discr, w_tpair.dd, "all_faces", c*flux_weak)


class _WaveTag:
    pass


def wave_operator(discr, c, w):
    """Compute the RHS of the wave equation.

    Parameters
    ----------
    discr: grudge.discretization.DiscretizationCollection
        the discretization to use
    c: float
        the (constant) wave speed
    w: numpy.ndarray
        an object array of DOF arrays, representing the state vector

    Returns
    -------
    numpy.ndarray
        an object array of DOF arrays, representing the ODE RHS
    """
    u = w[0]
    v = w[1:]

    dir_u = op.project(discr, "vol", BTAG_ALL, u)
    dir_v = op.project(discr, "vol", BTAG_ALL, v)
    dir_bval = flat_obj_array(dir_u, dir_v)
    dir_bc = flat_obj_array(-dir_u, dir_v)

    return (
        op.inverse_mass(discr,
            flat_obj_array(
                -c*op.weak_local_div(discr, "vol", v),
                -c*op.weak_local_grad(discr, "vol", u)
                )
            +  # noqa: W504
            op.face_mass(discr,
                _flux(discr, c=c,
                      w_tpair=TracePair(BTAG_ALL, interior=dir_bval,
                                        exterior=dir_bc))
                + sum(
                    _flux(discr, c=c, w_tpair=tpair)
                    for tpair in interior_trace_pairs(discr, w, comm_tag=_WaveTag))
                )
            )
        )
