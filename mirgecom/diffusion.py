r""":mod:`mirgecom.diffusion` computes the diffusion operator.

.. autofunction:: diffusion_operator
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
from pytools.obj_array import make_obj_array, obj_array_vectorize
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw, DOFArray
from grudge.symbolic.primitives import TracePair, DOFDesc, QTAG_NONE
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs


def _scalar(s):
    """Create an object array for a scalar."""
    return make_obj_array([s])


def _q_flux_const_diff(discr, sqrt_alpha, u_tpair):
    normal = thaw(u_tpair.int[0].array_context, discr.normal(u_tpair.dd))
    flux_weak = sqrt_alpha*u_tpair.avg*normal
    return discr.project(u_tpair.dd, "all_faces", flux_weak)


def _u_flux_const_diff(discr, sqrt_alpha, q_tpair):
    normal = thaw(q_tpair.int[0].array_context, discr.normal(q_tpair.dd))
    flux_weak = sqrt_alpha*_scalar(np.dot(q_tpair.avg, normal))
    return discr.project(q_tpair.dd, "all_faces", flux_weak)


def _q_flux_var_diff(discr, var_diff_quad_tag, alpha, u_tpair):
    actx = u_tpair.int[0].array_context

    dd = u_tpair.dd
    dd_quad = dd.with_qtag(var_diff_quad_tag)

    normal = thaw(actx, discr.normal(dd))

    flux_weak = u_tpair.avg*normal

    dd_allfaces_quad = dd_quad.with_dtag("all_faces")
    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = actx.np.sqrt(alpha_quad)
    flux_quad = discr.project(dd, dd_quad, flux_weak)

    return discr.project(dd_quad, dd_allfaces_quad, _scalar(sqrt_alpha_quad)
                * flux_quad)


def _u_flux_var_diff(discr, var_diff_quad_tag, alpha, q_tpair):
    actx = q_tpair.int[0].array_context

    dd = q_tpair.dd
    dd_quad = dd.with_qtag(var_diff_quad_tag)

    normal = thaw(actx, discr.normal(dd))

    flux_weak = _scalar(np.dot(q_tpair.avg, normal))

    dd_allfaces_quad = dd_quad.with_dtag("all_faces")
    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = actx.np.sqrt(alpha_quad)
    flux_quad = discr.project(dd, dd_quad, flux_weak)

    return discr.project(dd_quad, dd_allfaces_quad, _scalar(sqrt_alpha_quad)
                * flux_quad)


def _operator_const_diff(discr, alpha, w):
    u = make_obj_array([w])

    dir_u = discr.project("vol", BTAG_ALL, u)

    sqrt_alpha = math.sqrt(alpha)

    q = discr.inverse_mass(
        -sqrt_alpha*discr.weak_grad(u[0])
        +  # noqa: W504
        discr.face_mass(
            _q_flux_const_diff(discr, sqrt_alpha=sqrt_alpha,
                        u_tpair=interior_trace_pair(discr, u))
            + _q_flux_const_diff(discr, sqrt_alpha=sqrt_alpha,
                u_tpair=TracePair(BTAG_ALL, interior=dir_u, exterior=-dir_u))
            + sum(
                _q_flux_const_diff(discr, sqrt_alpha=sqrt_alpha, u_tpair=tpair)
                for tpair in cross_rank_trace_pairs(discr, u)
            )
        ))

    dir_q = discr.project("vol", BTAG_ALL, q)

    return (
        discr.inverse_mass(
            -sqrt_alpha*_scalar(discr.weak_div(q))
            +  # noqa: W504
            discr.face_mass(
                _u_flux_const_diff(discr, sqrt_alpha=sqrt_alpha,
                            q_tpair=interior_trace_pair(discr, q))
                + _u_flux_const_diff(discr, sqrt_alpha=sqrt_alpha,
                    q_tpair=TracePair(BTAG_ALL, interior=dir_q, exterior=dir_q))
                + sum(
                    _u_flux_const_diff(discr, sqrt_alpha=sqrt_alpha,
                        q_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, q))
                )
            )[0]
        )


def _operator_var_diff(discr, alpha, w, var_diff_quad_tag):
    if var_diff_quad_tag is None:
        raise RuntimeError("Must specify var_diff_quad_tag when using variable"
            " diffusivity.")

    actx = w.array_context

    u = make_obj_array([w])

    dir_u = discr.project("vol", BTAG_ALL, u)

    dd_quad = DOFDesc("vol", var_diff_quad_tag)
    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = actx.np.sqrt(alpha_quad)
    u_quad = discr.project("vol", dd_quad, u[0])

    dd_allfaces_quad = DOFDesc("all_faces", var_diff_quad_tag)

    q = (
        -discr.grad(actx.np.sqrt(alpha))*u  # not sure how to do overintegration here
        +  # noqa: W504
        discr.inverse_mass(
            -discr.weak_grad(dd_quad, sqrt_alpha_quad*u_quad)
            +  # noqa: W504
            discr.face_mass(
                dd_allfaces_quad,
                _q_flux_var_diff(discr, var_diff_quad_tag, alpha,
                    u_tpair=interior_trace_pair(discr, u))
                + _q_flux_var_diff(discr, var_diff_quad_tag, alpha,
                    u_tpair=TracePair(BTAG_ALL, interior=dir_u, exterior=-dir_u))
                + sum(
                    _q_flux_var_diff(discr, var_diff_quad_tag, alpha,
                        u_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, u)
                )
            ))
        )

    dir_q = discr.project("vol", BTAG_ALL, q)

    q_quad = discr.project("vol", dd_quad, q)

    return (
        discr.inverse_mass(
            -_scalar(discr.weak_div(dd_quad, _scalar(sqrt_alpha_quad)*q_quad))
            +  # noqa: W504
            discr.face_mass(
                dd_allfaces_quad,
                _u_flux_var_diff(discr, var_diff_quad_tag, alpha,
                    q_tpair=interior_trace_pair(discr, q))
                + _u_flux_var_diff(discr, var_diff_quad_tag, alpha,
                    q_tpair=TracePair(BTAG_ALL, interior=dir_q, exterior=dir_q))
                + sum(
                    _u_flux_var_diff(discr, var_diff_quad_tag, alpha,
                        q_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, q))
                )
            )[0]
        )


def _operator(discr, alpha, w, var_diff_quad_tag):
    if isinstance(alpha, np.ndarray):
        return _operator_var_diff(discr, alpha, w, var_diff_quad_tag)
    else:
        return _operator_const_diff(discr, alpha, w)


def diffusion_operator(discr, alpha, w, var_diff_quad_tag=None):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    :math:`\nabla\cdot(\alpha\nabla w)`, where :math:`\alpha` is the diffusivity and
    :math:`w` is a scalar field.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    alpha: float
        the (constant) diffusivity
    w: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array or object array of DOF arrays to which the operator should be
        applied
    var_diff_quad_tag: string or QTAG_NONE
        quadrature tag indicating which discretization method in *discr* to use for
        variable diffusivity

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *w*
    """
    if (isinstance(w, np.ndarray)
            and w.dtype.char == "O"
            and not isinstance(w, DOFArray)):
        return obj_array_vectorize(lambda u: _operator(discr, alpha, u,
                    var_diff_quad_tag), w)
    else:
        return _operator(discr, alpha, w, var_diff_quad_tag)
