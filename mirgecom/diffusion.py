r""":mod:`mirgecom.diffusion` computes the diffusion operator.

.. autofunction:: diffusion_operator
.. autoclass:: DiffusionBoundary
.. autoclass:: DirichletDiffusionBoundary
.. autoclass:: NeumannDiffusionBoundary
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

import abc
import math
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array, obj_array_vectorize_n_args
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw, DOFArray
from grudge.symbolic.primitives import DOFDesc
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair, as_dofdesc


def _sqrt(actx, x):
    if isinstance(x, DOFArray):
        return actx.np.sqrt(x)
    else:
        return math.sqrt(x)


def _grad(discr, x):
    if isinstance(x, DOFArray):
        return discr.grad(x)
    else:
        return 0.


def _q_flux(discr, quad_tag, alpha, u_tpair):
    actx = u_tpair.int.array_context

    dd = u_tpair.dd
    dd_quad = dd.with_qtag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = _sqrt(actx, alpha_quad)

    u_avg_quad = discr.project(dd, dd_quad, u_tpair.avg)

    return discr.project(dd_quad, dd_allfaces_quad,
        -sqrt_alpha_quad * u_avg_quad * normal_quad)


def _u_flux(discr, quad_tag, alpha, q_tpair):
    actx = q_tpair.int[0].array_context

    dd = q_tpair.dd
    dd_quad = dd.with_qtag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = _sqrt(actx, alpha_quad)

    q_avg_quad = discr.project(dd, dd_quad, q_tpair.avg)

    return discr.project(dd_quad, dd_allfaces_quad,
        -sqrt_alpha_quad * np.dot(q_avg_quad, normal_quad))


class DiffusionBoundary(metaclass=abc.ABCMeta):
    """
    Diffusion boundary base class.

    .. automethod:: get_q_flux
    .. automethod:: get_u_flux
    """

    @abc.abstractmethod
    def get_q_flux(self, discr, quad_tag, alpha, dd, u):
        """Compute the flux for *q* on the boundary corresponding to *dd*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_u_flux(self, discr, quad_tag, alpha, dd, q):
        """Compute the flux for *u* on the boundary corresponding to *dd*."""
        raise NotImplementedError


class DirichletDiffusionBoundary(DiffusionBoundary):
    r"""
    Dirichlet boundary condition for the diffusion operator.

    For boundary condition $u|_\Gamma = f$, uses external data

    .. math::

                 u^+ &= 2 f - u^-

        \mathbf{q}^+ &= \mathbf{q}^-

    to compute boundary fluxes as shown in [Hesthaven_2008]_, Section 7.1.

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) along the boundary
        """
        self.value = value

    # Observe: Dirichlet BC enforced on q, not u
    def get_q_flux(self, discr, quad_tag, alpha, dd, u):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=2.*self.value-u_int)
        return _q_flux(discr, quad_tag, alpha, u_tpair)

    def get_u_flux(self, discr, quad_tag, alpha, dd, q):  # noqa: D102
        q_int = discr.project("vol", dd, q)
        q_tpair = TracePair(dd, interior=q_int, exterior=q_int)
        return _u_flux(discr, quad_tag, alpha, q_tpair)


class NeumannDiffusionBoundary(DiffusionBoundary):
    r"""
    Neumann boundary condition for the diffusion operator.

    For boundary condition $\frac{\partial u}{\partial \mathbf{n}}|_\Gamma = g$, uses
    external data

    .. math::

        u^+ = u^-

    to compute boundary fluxes for $\mathbf{q} = \sqrt{\alpha}\nabla u$, and computes
    boundary fluxes for $u_t = \nabla \cdot (\sqrt{\alpha} \mathbf{q})$ using

    .. math::

        \mathbf{F}\cdot\mathbf{\hat{n}} &= \alpha\nabla u\cdot\mathbf{\hat{n}}

                                        &= \alpha\frac{\partial u}{\partial
                                                \mathbf{n}}

                                        &= \alpha g

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) along the boundary
        """
        self.value = value

    def get_q_flux(self, discr, quad_tag, alpha, dd, u):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=u_int)
        return _q_flux(discr, quad_tag, alpha, u_tpair)

    def get_u_flux(self, discr, quad_tag, alpha, dd, q):  # noqa: D102
        dd_quad = dd.with_qtag(quad_tag)
        dd_allfaces_quad = dd_quad.with_dtag("all_faces")
        # Compute the flux directly instead of constructing an external q value
        # (and the associated TracePair); this approach is simpler in the
        # spatially-varying alpha case (the other approach would result in a
        # q_tpair that lives in the quadrature discretization, as it involves
        # computing sqrt(alpha); _u_flux would need to be modified to accept such
        # values).
        alpha_int_quad = discr.project("vol", dd_quad, alpha)
        if isinstance(self.value, DOFArray):
            value_quad = discr.project(dd, dd_quad, self.value)
            flux_quad = -alpha_int_quad*value_quad
        else:
            flux_quad = -alpha_int_quad*self.value
        return discr.project(dd_quad, dd_allfaces_quad, flux_quad)


def diffusion_operator(discr, quad_tag, alpha, boundaries, u):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    $\nabla\cdot(\alpha\nabla u)$, where $\alpha$ is the diffusivity and
    $u$ is a scalar field.

    Uses unstabilized central numerical fluxes.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    quad_tag:
        quadrature tag indicating which discretization in *discr* to use for
        overintegration
    alpha: Union[numbers.Number, meshmode.dof_array.DOFArray]
        the diffusivity value(s)
    boundaries:
        dictionary (or list of dictionaries) mapping boundary tags to
        :class:`DiffusionBoundary` instances
    u: Union[meshmode.dof_array.DOFArray, numpy.ndarray]
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *u*
    """
    if isinstance(u, np.ndarray):
        if not isinstance(boundaries, list):
            raise TypeError("boundaries must be a list if u is an object array")
        if len(boundaries) != len(u):
            raise TypeError("boundaries must be the same length as u")
        return obj_array_vectorize_n_args(lambda boundaries_i, u_i:
            diffusion_operator(discr, quad_tag, alpha, boundaries_i, u_i),
            make_obj_array(boundaries), u)

    for btag, bdry in boundaries.items():
        if not isinstance(bdry, DiffusionBoundary):
            raise TypeError(f"Unrecognized boundary type for tag {btag}. "
                "Must be an instance of DiffusionBoundary.")

    actx = u.array_context

    dd_quad = DOFDesc("vol", quad_tag)
    dd_allfaces_quad = DOFDesc("all_faces", quad_tag)

    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = _sqrt(actx, alpha_quad)
    grad_alpha_quad = discr.project("vol", dd_quad, _grad(discr, alpha))

    u_quad = discr.project("vol", dd_quad, u)

    q = discr.inverse_mass(
        # Decompose phi_i*grad(sqrt(alpha)*phi_j) term via the product rule in
        # order to avoid having to define a new operator
        discr.mass(dd_quad, -0.5/sqrt_alpha_quad * grad_alpha_quad * u_quad)
        +  # noqa: W504
        discr.weak_grad(dd_quad, -sqrt_alpha_quad * u_quad)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            _q_flux(discr, quad_tag, alpha, interior_trace_pair(discr, u))
            + sum(
                bdry.get_q_flux(discr, quad_tag, alpha, as_dofdesc(btag),
                    u)
                for btag, bdry in boundaries.items()
            )
            + sum(
                _q_flux(discr, quad_tag, alpha, tpair)
                for tpair in cross_rank_trace_pairs(discr, u)
            )
        ))

    q_quad = discr.project("vol", dd_quad, q)

    return (
        discr.inverse_mass(
            discr.weak_div(dd_quad, -sqrt_alpha_quad*q_quad)
            -  # noqa: W504
            discr.face_mass(
                dd_allfaces_quad,
                _u_flux(discr, quad_tag, alpha, interior_trace_pair(discr, q))
                + sum(
                    bdry.get_u_flux(discr, quad_tag, alpha, as_dofdesc(btag),
                        q)
                    for btag, bdry in boundaries.items()
                )
                + sum(
                    _u_flux(discr, quad_tag, alpha, tpair)
                    for tpair in cross_rank_trace_pairs(discr, q))
                )
            )
        )
