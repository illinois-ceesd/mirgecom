__copyright__ = "Copyright (C) 2020 CEESD Developers"

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
from pytools.obj_array import (
    flat_obj_array, make_obj_array)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from grudge.symbolic.primitives import TracePair
from grudge.eager import interior_trace_pair


__doc__ = """
.. autofunction:: wave_operator
"""


def _flux(discr, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    actx = w_tpair.int[0].array_context

    normal = thaw(actx, discr.normal(w_tpair.dd))

    def normal_times(scalar):
        # workaround for object array behavior
        return make_obj_array([ni*scalar for ni in normal])

    flux_weak = flat_obj_array(
        np.dot(v.avg, normal),
        normal_times(u.avg),
        )

    # upwind
    v_jump = np.dot(normal, v.int-v.ext)
    flux_weak += flat_obj_array(
        0.5*(u.int-u.ext),
        0.5*normal_times(v_jump),
        )

    return discr.project(w_tpair.dd, "all_faces", c*flux_weak)


def wave_operator(discr, c, w):
    """
    Args:
        discr (grudge.eager.EagerDGDiscretization): the discretization to use
        c (float): the (constant) wave speed)
        w (np.ndarray): an object array of DOF arrays, representing the state vector

    Returns:
        np.ndarray: an object array of DOF arrays, representing the ODE RHS
    """
    actx = w[0].array_context

    u = w[0]
    v = w[1:]

    dir_u = discr.project("vol", BTAG_ALL, u)
    dir_v = discr.project("vol", BTAG_ALL, v)
    dir_bval = flat_obj_array(dir_u, dir_v)
    dir_bc = flat_obj_array(-dir_u, dir_v)

    return (
        discr.inverse_mass(
            flat_obj_array(
                -c*discr.weak_div(v),
                -c*discr.weak_grad(u)
                )
            +  # noqa: W504
            discr.face_mass(
                _flux(discr, c=c, w_tpair=interior_trace_pair(discr, w))
                + _flux(discr, c=c, w_tpair=TracePair(BTAG_ALL, dir_bval,
                dir_bc))
                ))
        )
