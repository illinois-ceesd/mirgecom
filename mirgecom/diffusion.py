r""":mod:`mirgecom.diffusion` computes the diffusion operator.

.. autofunction:: gradient_flux
.. autofunction:: diffusion_flux
.. autofunction:: diffusion_operator
.. autoclass:: DiffusionBoundary
.. autoclass:: DirichletDiffusionBoundary
.. autoclass:: NeumannDiffusionBoundary
.. autoclass:: AggregateDiffusionBoundary
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
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from grudge.dof_desc import DOFDesc, as_dofdesc
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair


def gradient_flux(discr, quad_tag, u_tpair):
    r"""Compute the numerical flux for $\nabla u$."""
    if isinstance(u_tpair.int, np.ndarray):
        actx = u_tpair.int[0].array_context
    else:
        actx = u_tpair.int.array_context

    dd = u_tpair.dd
    dd_quad = dd.with_discr_tag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(u, normal):
        if isinstance(u, np.ndarray):
            return -np.outer(u, normal)
        else:
            return -u * normal

    return discr.project(dd_quad, dd_allfaces_quad, flux(
        to_quad(u_tpair.avg), normal_quad))


def diffusion_flux(discr, quad_tag, alpha_tpair, grad_u_tpair):
    r"""Compute the numerical flux for $\nabla \cdot (\alpha \nabla u)$."""
    if isinstance(grad_u_tpair.int[0], np.ndarray):
        actx = grad_u_tpair.int[0][0].array_context
    else:
        actx = grad_u_tpair.int[0].array_context

    dd = grad_u_tpair.dd
    dd_quad = dd.with_discr_tag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(alpha, grad_u, normal):
        if isinstance(grad_u[0], np.ndarray):
            return -alpha * np.stack(grad_u, axis=0) @ normal
        else:
            return -alpha * np.dot(grad_u, normal)

    flux_tpair = TracePair(dd_quad,
        interior=flux(
            to_quad(alpha_tpair.int), to_quad(grad_u_tpair.int), normal_quad),
        exterior=flux(
            to_quad(alpha_tpair.ext), to_quad(grad_u_tpair.ext), normal_quad)
        )

    return discr.project(dd_quad, dd_allfaces_quad, flux_tpair.avg)


class DiffusionBoundary(metaclass=abc.ABCMeta):
    """
    Diffusion boundary base class.

    .. automethod:: get_gradient_flux
    .. automethod:: get_diffusion_flux
    """

    @abc.abstractmethod
    def get_gradient_flux(self, discr, quad_tag, dd, alpha, u):
        """Compute the flux for grad(u) on the boundary corresponding to *dd*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u):
        """Compute the flux for diff(u) on the boundary corresponding to *dd*."""
        raise NotImplementedError


class DirichletDiffusionBoundary(DiffusionBoundary):
    r"""
    Dirichlet boundary condition for the diffusion operator.

    For the boundary condition $u|_\Gamma = f$, uses external data

    .. math::

                 u^+ &= 2 f - u^-

        (\nabla u)^+ &= (\nabla u)^-

    to compute boundary fluxes as shown in [Hesthaven_2008]_, Section 7.1.

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) of $f$ along the boundary
        """
        self.value = value

    def get_gradient_flux(self, discr, quad_tag, dd, alpha, u):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=2*self.value-u_int)
        return gradient_flux(discr, quad_tag, u_tpair)

    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u):  # noqa: D102
        alpha_int = discr.project("vol", dd, alpha)
        alpha_tpair = TracePair(dd, interior=alpha_int, exterior=alpha_int)
        grad_u_int = discr.project("vol", dd, grad_u)
        grad_u_tpair = TracePair(dd, interior=grad_u_int, exterior=grad_u_int)
        return diffusion_flux(discr, quad_tag, alpha_tpair, grad_u_tpair)


class NeumannDiffusionBoundary(DiffusionBoundary):
    r"""
    Neumann boundary condition for the diffusion operator.

    For the boundary condition $(\nabla u \cdot \mathbf{\hat{n}})|_\Gamma = g$, uses
    external data

    .. math::

        u^+ = u^-

    when computing the boundary fluxes for $\nabla u$, and uses

    .. math::

        (-\alpha \nabla u\cdot\mathbf{\hat{n}})|_\Gamma &=
            -\alpha^- (\nabla u\cdot\mathbf{\hat{n}})|_\Gamma

                                                        &= -\alpha^- g

    when computing the boundary fluxes for $\nabla \cdot (\alpha \nabla u)$.

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) of $g$ along the boundary
        """
        self.value = value

    def get_gradient_flux(self, discr, quad_tag, dd, alpha, u):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=u_int)
        return gradient_flux(discr, quad_tag, u_tpair)

    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u):  # noqa: D102
        dd_quad = dd.with_discr_tag(quad_tag)
        dd_allfaces_quad = dd_quad.with_dtag("all_faces")
        # Compute the flux directly instead of constructing an external grad_u value
        # (and the associated TracePair); this approach is simpler in the
        # spatially-varying alpha case (the other approach would result in a
        # grad_u_tpair that lives in the quadrature discretization; diffusion_flux
        # would need to be modified to accept such values).
        alpha_int_quad = discr.project("vol", dd_quad, alpha)
        value_quad = discr.project(dd, dd_quad, self.value)
        flux_quad = -alpha_int_quad*value_quad
        return discr.project(dd_quad, dd_allfaces_quad, flux_quad)


class AggregateDiffusionBoundary(DiffusionBoundary):
    """
    Combined boundary condition for non-scalar diffusion operator.

    Aggregates diffusion BCs for multiple components into a single BC.

    .. automethod:: __init__
    """

    def __init__(self, boundaries):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        boundaries:
            a list or object array of :class:`DiffusionBoundary` instances
        """
        self.boundaries = boundaries.copy()

    def get_gradient_flux(self, discr, quad_tag, dd, alpha, u):  # noqa: D102
        component_fluxes = make_obj_array([
            bdry.get_gradient_flux(discr, quad_tag, dd, alpha, u[i])
            for i, bdry in enumerate(self.boundaries)
            ])
        return np.stack(component_fluxes, axis=0)

    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u):  # noqa: D102
        component_fluxes = make_obj_array([
            bdry.get_diffusion_flux(discr, quad_tag, dd, alpha, grad_u[i])
            for i, bdry in enumerate(self.boundaries)
            ])
        return component_fluxes


def diffusion_operator(discr, quad_tag, alpha, boundaries, u, boundary_kwargs=None,
        return_grad_u=False):
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
    alpha: numbers.Number or meshmode.dof_array.DOFArray
        the diffusivity value(s)
    boundaries:
        dictionary mapping boundary tags to :class:`DiffusionBoundary` instances or
        fluid boundary class instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    boundary_kwargs: :class:`dict`
        dictionary of extra arguments to pass through to the boundary conditions
    return_grad_u: bool
        an optional flag indicating whether $\nabla u$ should also be returned

    Returns
    -------
    diff_u: meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *u*
    grad_u: numpy.ndarray
        the gradient of *u*; only returned if *return_grad_u* is True
    """
    if boundary_kwargs is None:
        boundary_kwargs = dict()

    dd_quad = DOFDesc("vol", quad_tag)
    dd_allfaces_quad = DOFDesc("all_faces", quad_tag)

    grad_u = discr.inverse_mass(
        discr.weak_grad(-u)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            gradient_flux(discr, quad_tag, interior_trace_pair(discr, u))
            + sum(
                bdry.get_gradient_flux(discr, quad_tag, as_dofdesc(btag), alpha, u)
                for btag, bdry in boundaries.items()
                if isinstance(bdry, DiffusionBoundary))
            + sum(
                gradient_flux(discr, quad_tag,
                    TracePair(as_dofdesc(btag),
                        interior=discr.project("vol", as_dofdesc(btag), u),
                        exterior=bdry.exterior_q(discr, u, as_dofdesc(btag),
                            **boundary_kwargs)))
                for btag, bdry in boundaries.items()
                if not isinstance(bdry, DiffusionBoundary))
            + sum(
                gradient_flux(discr, quad_tag, u_tpair)
                for u_tpair in cross_rank_trace_pairs(discr, u))
            )
        )

    alpha_quad = discr.project("vol", dd_quad, alpha)
    grad_u_quad = discr.project("vol", dd_quad, grad_u)

    diff_u = discr.inverse_mass(
        discr.weak_div(dd_quad, -alpha_quad*grad_u_quad)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            diffusion_flux(discr, quad_tag, interior_trace_pair(discr, alpha),
                interior_trace_pair(discr, grad_u))
            + sum(
                bdry.get_diffusion_flux(discr, quad_tag, as_dofdesc(btag), alpha,
                    grad_u)
                for btag, bdry in boundaries.items()
                if isinstance(bdry, DiffusionBoundary))
            + sum(
                diffusion_flux(discr, quad_tag,
                    # Dumb hack to avoid projecting alpha twice
                    (lambda alpha_int:
                        TracePair(as_dofdesc(btag),
                            interior=alpha_int,
                            exterior=alpha_int)
                     )(discr.project("vol", as_dofdesc(btag), alpha)),
                    TracePair(as_dofdesc(btag),
                        interior=discr.project("vol", as_dofdesc(btag), grad_u),
                        exterior=bdry.exterior_grad_q(discr, grad_u,
                            as_dofdesc(btag), **boundary_kwargs)))
                for btag, bdry in boundaries.items()
                if not isinstance(bdry, DiffusionBoundary))
            + sum(
                diffusion_flux(discr, quad_tag, alpha_tpair, grad_u_tpair)
                for alpha_tpair, grad_u_tpair in zip(
                    cross_rank_trace_pairs(discr, alpha),
                    cross_rank_trace_pairs(discr, grad_u)))
            )
        )

    if return_grad_u:
        return diff_u, grad_u
    else:
        return diff_u
