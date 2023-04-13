r""":mod:`mirgecom.diffusion` computes the diffusion operator.

.. autofunction:: grad_facial_flux
.. autofunction:: diffusion_facial_flux
.. autofunction:: grad_operator
.. autofunction:: diffusion_operator
.. autoclass:: DiffusionBoundary
.. autoclass:: DirichletDiffusionBoundary
.. autoclass:: NeumannDiffusionBoundary
.. autoclass:: PrescribedFluxDiffusionBoundary
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
from functools import partial
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array, obj_array_vectorize_n_args
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.discretization.connection import FACE_RESTR_ALL  # noqa
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    VolumeDomainTag,
    DISCR_TAG_BASE,
)
from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs,
    tracepair_with_discr_tag,
)
import grudge.op as op
from mirgecom.utils import normalize_boundaries


def grad_facial_flux(u_tpair, normal):
    r"""Compute the numerical flux for $\nabla u$."""
    return -u_tpair.avg * normal


def diffusion_facial_flux(kappa_tpair, grad_u_tpair, normal):
    r"""Compute the numerical flux for $\nabla \cdot (\kappa \nabla u)$."""
    flux_tpair = TracePair(grad_u_tpair.dd,
        interior=-kappa_tpair.int * np.dot(grad_u_tpair.int, normal),
        exterior=-kappa_tpair.ext * np.dot(grad_u_tpair.ext, normal))
    return flux_tpair.avg


class DiffusionBoundary(metaclass=abc.ABCMeta):
    """
    Diffusion boundary base class.

    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    @abc.abstractmethod
    def get_grad_flux(self, dcoll, dd_bdry, u_minus):
        """Compute the flux for grad(u) on the boundary *dd_bdry*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_diffusion_flux(self, dcoll, dd_bdry, kappa_minus, grad_u_minus):
        """Compute the flux for diff(u) on the boundary *dd_bdry*."""
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
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(self, function):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        function
            user-defined function prescribing the flux along the boundary
        """
        self.function = function

    def get_grad_flux(self, dcoll, dd_bdry, u_minus, **kwargs):
        """Flux for gradient evaluation."""
        actx = u_minus.array_context
        ext_value = self.function(u_minus, **kwargs)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=2*ext_value-u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return grad_facial_flux(u_tpair, normal)

    def get_diffusion_flux(self, dcoll, dd_bdry, u_minus, kappa_minus,
                           grad_u_minus, **kwargs):
        """Flux for Laplacian evaluation."""
        actx = grad_u_minus[0].array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        grad_u_tpair = TracePair(dd_bdry,
            interior=grad_u_minus,
            exterior=grad_u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return diffusion_facial_flux(kappa_tpair, grad_u_tpair, normal)


class NeumannDiffusionBoundary(DiffusionBoundary):
    r"""
    Neumann boundary condition for the diffusion operator.

    For the boundary condition $(\nabla u \cdot \mathbf{\hat{n}})|_\Gamma = g$, uses
    external data

    .. math::

        u^+ = u^-

    when computing the boundary fluxes for $\nabla u$, and uses

    .. math::

        (-\kappa \nabla u\cdot\mathbf{\hat{n}})|_\Gamma &=
            -\kappa^- (\nabla u\cdot\mathbf{\hat{n}})|_\Gamma

                                                        &= -\kappa^- g

    when computing the boundary fluxes for $\nabla \cdot (\kappa \nabla u)$.

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(self, function):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        function
            user-defined function to prescribe the gradient along the boundary
        """
        self.function = function

    def get_grad_flux(self, dcoll, dd_bdry, u_minus, **kwargs):
        """Flux for gradient evaluation."""
        actx = u_minus.array_context
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return grad_facial_flux(u_tpair, normal)

    def get_diffusion_flux(self, dcoll, dd_bdry, u_minus, kappa_minus,
                           grad_u_minus, **kwargs):
        """Flux for Laplacian evaluation."""
        actx = grad_u_minus[0].array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        grad_u_tpair = TracePair(dd_bdry,
            interior=grad_u_minus,
            exterior=(
                grad_u_minus
                + 2 * (self.function(u_minus, grad_u_minus, **kwargs)
                       - np.dot(grad_u_minus, normal)) * normal))
        return diffusion_facial_flux(kappa_tpair, grad_u_tpair, normal)


class PrescribedFluxDiffusionBoundary(DiffusionBoundary):
    r"""Prescribed flux boundary condition for the diffusion operator.

    For the boundary condition $(\nabla u \cdot \mathbf{\hat{n}})|_\Gamma$, uses
    external data

    .. math::

        u^+ = u^-

    when computing the boundary fluxes for $\nabla u$, and and uses the
    prescribed flux to evaluate the numerical flux $f^*$ using

    .. math::

        f^*|_\Gamma = \frac{1}{2} (F_{presc} + F^-)

    when computing the boundary fluxes for $\nabla \cdot (\kappa \nabla u)$.

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(self, function):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        function
            function prescribing the external flux
        """
        self.function = function

    def get_grad_flux(self, dcoll, dd_bdry, u_minus, **kwargs):
        """Flux for gradient evaluation."""
        actx = u_minus.array_context
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return grad_facial_flux(u_tpair, normal)

    def get_diffusion_flux(self, dcoll, dd_bdry, u_minus, kappa_minus,
                           grad_u_minus, **kwargs):
        """Flux for Laplacian evaluation."""
        actx = grad_u_minus[0].array_context
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        grad_u_tpair = TracePair(dd_bdry,
            interior=grad_u_minus,
            exterior=grad_u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))

        # average between the prescribed value and the internal value
        # FIXME maybe the minus in the prescribed function is due to the normal?
        return 0.5*(-kappa_tpair.int * np.dot(grad_u_tpair.int, normal)
            - self.function(u_tpair, kappa_tpair, grad_u_tpair, normal, **kwargs))


class _DiffusionStateTag:
    pass


class _DiffusionKappaTag:
    pass


class _DiffusionGradTag:
    pass


def grad_operator(
        dcoll, boundaries, u, *, quadrature_tag=DISCR_TAG_BASE, dd=DD_VOLUME_ALL,
        comm_tag=None, **kwargs):
    r"""
    Compute the gradient of *u*.

    Uses unstabilized central numerical fluxes.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    boundaries:
        dictionary (or list of dictionaries) mapping boundary tags to
        :class:`DiffusionBoundary` instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    quadrature_tag:
        quadrature tag indicating which discretization in *dcoll* to use for
        overintegration
    dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *u* lives. Must be a volume
        on the base discretization.
    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    grad_u: numpy.ndarray
        the gradient of *u*
    """
    if isinstance(u, np.ndarray):
        if not isinstance(boundaries, list):
            raise TypeError("boundaries must be a list if u is an object array")
        if len(boundaries) != len(u):
            raise TypeError("boundaries must be the same length as u")
        return obj_array_vectorize_n_args(
            lambda boundaries_i, u_i: grad_operator(
                dcoll, boundaries_i, u_i, quadrature_tag=quadrature_tag,
                dd=dd),
            make_obj_array(boundaries), u)

    actx = u.array_context

    boundaries = normalize_boundaries(boundaries)

    for bdtag, bdry in boundaries.items():
        if not isinstance(bdry, DiffusionBoundary):
            raise TypeError(f"Unrecognized boundary type for tag {bdtag}. "
                "Must be an instance of DiffusionBoundary.")

    if not isinstance(dd.domain_tag, VolumeDomainTag):
        raise TypeError("dd must represent a volume")
    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    def interior_flux(u_tpair):
        dd_trace_quad = u_tpair.dd.with_discr_tag(quadrature_tag)
        u_tpair_quad = interp_to_surf_quad(u_tpair)
        normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
        return op.project(
            dcoll, dd_trace_quad, dd_allfaces_quad,
            grad_facial_flux(u_tpair_quad, normal_quad))

    def boundary_flux(bdtag, bdry):
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
        u_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, u)
        return op.project(
            dcoll, dd_bdry_quad, dd_allfaces_quad,
            bdry.get_grad_flux(dcoll, dd_bdry_quad, u_minus_quad, **kwargs))

    return op.inverse_mass(
        dcoll, dd_vol,
        op.weak_local_grad(dcoll, dd_vol, -u)
        -  # noqa: W504
        op.face_mass(
            dcoll, dd_allfaces_quad,
            sum(
                interior_flux(u_tpair)
                for u_tpair in interior_trace_pairs(
                    dcoll, u, volume_dd=dd_vol,
                    comm_tag=(_DiffusionStateTag, comm_tag)))
            + sum(
                boundary_flux(bdtag, bdry)
                for bdtag, bdry in boundaries.items())
            )
        )


def diffusion_operator(
        dcoll, kappa, boundaries, u, *, return_grad_u=False,
        quadrature_tag=DISCR_TAG_BASE, dd=DD_VOLUME_ALL, comm_tag=None,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        grad_u=None, **kwargs):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    $\nabla\cdot(\kappa\nabla u)$, where $\kappa$ is the conductivity and
    $u$ is a scalar field.

    Uses unstabilized central numerical fluxes.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    kappa: numbers.Number or meshmode.dof_array.DOFArray
        the conductivity value(s)
    boundaries:
        dictionary (or list of dictionaries) mapping boundary domain tags to
        :class:`DiffusionBoundary` instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    return_grad_u: bool
        an optional flag indicating whether $\nabla u$ should also be returned
    quadrature_tag:
        quadrature tag indicating which discretization in *dcoll* to use for
        overintegration
    dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *u* lives. Must be a volume
        on the base discretization.
    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    diff_u: meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *u*
    grad_u: numpy.ndarray
        the gradient of *u*; only returned if *return_grad_u* is True
    """
    if isinstance(u, np.ndarray):
        if not isinstance(boundaries, list):
            raise TypeError("boundaries must be a list if u is an object array")
        if len(boundaries) != len(u):
            raise TypeError("boundaries must be the same length as u")
        return obj_array_vectorize_n_args(
            lambda boundaries_i, u_i: diffusion_operator(
                dcoll, kappa, boundaries_i, u_i, return_grad_u=return_grad_u,
                quadrature_tag=quadrature_tag, dd=dd),
            make_obj_array(boundaries), u)

    actx = u.array_context

    boundaries = normalize_boundaries(boundaries)

    for bdtag, bdry in boundaries.items():
        if not isinstance(bdry, DiffusionBoundary):
            raise TypeError(f"Unrecognized boundary type for tag {bdtag}. "
                "Must be an instance of DiffusionBoundary.")

    if not isinstance(dd.domain_tag, VolumeDomainTag):
        raise TypeError("dd must represent a volume")
    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    if grad_u is None:
        grad_u = grad_operator(
            dcoll, boundaries, u, quadrature_tag=quadrature_tag, dd=dd_vol,
            comm_tag=comm_tag, **kwargs)

    kappa_quad = op.project(dcoll, dd_vol, dd_vol_quad, kappa)
    grad_u_quad = op.project(dcoll, dd_vol, dd_vol_quad, grad_u)

    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    def interior_flux(kappa_tpair, grad_u_tpair):
        dd_trace_quad = grad_u_tpair.dd.with_discr_tag(quadrature_tag)
        kappa_tpair_quad = interp_to_surf_quad(kappa_tpair)
        grad_u_tpair_quad = interp_to_surf_quad(grad_u_tpair)
        normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
        return op.project(
            dcoll, dd_trace_quad, dd_allfaces_quad,
            diffusion_facial_flux(kappa_tpair_quad, grad_u_tpair_quad, normal_quad))

    def boundary_flux(bdtag, bdry):
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
        u_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, u)
        kappa_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, kappa)
        grad_u_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, grad_u)
        return op.project(
            dcoll, dd_bdry_quad, dd_allfaces_quad,
            bdry.get_diffusion_flux(
                dcoll, dd_bdry_quad, u_minus_quad, kappa_minus_quad,
                grad_u_minus_quad, **kwargs)
        )

    diff_u = op.inverse_mass(
        dcoll, dd_vol,
        op.weak_local_div(dcoll, dd_vol_quad, -kappa_quad*grad_u_quad)
        -  # noqa: W504
        op.face_mass(
            dcoll, dd_allfaces_quad,
            sum(
                interior_flux(kappa_tpair, grad_u_tpair)
                for kappa_tpair, grad_u_tpair in zip(
                    interior_trace_pairs(
                        dcoll, kappa, volume_dd=dd_vol,
                        comm_tag=(_DiffusionKappaTag, comm_tag)),
                    interior_trace_pairs(
                        dcoll, grad_u, volume_dd=dd_vol,
                        comm_tag=(_DiffusionGradTag, comm_tag)))
            )
            + sum(
                boundary_flux(bdtag, bdry)
                for bdtag, bdry in boundaries.items())
            )
        )

    if return_grad_u:
        return diff_u, grad_u
    else:
        return diff_u
