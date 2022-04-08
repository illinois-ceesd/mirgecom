r""":mod:`mirgecom.diffusion` computes the diffusion operator.

.. autofunction:: grad_flux
.. autofunction:: diffusion_flux
.. autofunction:: grad_operator
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
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array, obj_array_vectorize_n_args
from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DOFDesc, as_dofdesc, DISCR_TAG_BASE
from grudge.trace_pair import TracePair, interior_trace_pairs


def grad_flux(discr, u_tpair, *, quadrature_tag=DISCR_TAG_BASE):
    r"""Compute the numerical flux for $\nabla u$."""
    actx = u_tpair.int.array_context

    dd = u_tpair.dd
    dd_quad = dd.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(discr.normal(dd_quad), actx)

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(u, normal):
        return -u * normal

    return discr.project(dd_quad, dd_allfaces_quad, flux(
        to_quad(u_tpair.avg), normal_quad))


def diffusion_flux(
        discr, kappa_tpair, grad_u_tpair, *, quadrature_tag=DISCR_TAG_BASE):
    r"""Compute the numerical flux for $\nabla \cdot (\kappa \nabla u)$."""
    actx = grad_u_tpair.int[0].array_context

    dd = grad_u_tpair.dd
    dd_quad = dd.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(discr.normal(dd_quad), actx)

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(kappa, grad_u, normal):
        return -kappa * np.dot(grad_u, normal)

    flux_tpair = TracePair(dd_quad,
        interior=flux(
            to_quad(kappa_tpair.int), to_quad(grad_u_tpair.int), normal_quad),
        exterior=flux(
            to_quad(kappa_tpair.ext), to_quad(grad_u_tpair.ext), normal_quad)
        )

    return discr.project(dd_quad, dd_allfaces_quad, flux_tpair.avg)


class DiffusionBoundary(metaclass=abc.ABCMeta):
    """
    Diffusion boundary base class.

    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    @abc.abstractmethod
    def get_grad_flux(
            self, discr, dd, u, *, quadrature_tag=DISCR_TAG_BASE):
        """Compute the flux for grad(u) on the boundary corresponding to *dd*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_diffusion_flux(
            self, discr, dd, kappa, grad_u, *, quadrature_tag=DISCR_TAG_BASE):
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

    def get_grad_flux(
            self, discr, dd, u, *, quadrature_tag=DISCR_TAG_BASE):  # noqa: D102
        """Get gradient flux."""
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=2*self.value-u_int)
        return grad_flux(discr, u_tpair, quadrature_tag=quadrature_tag)

    def get_diffusion_flux(
            self, discr, dd, kappa, grad_u, *,
            quadrature_tag=DISCR_TAG_BASE):  # noqa: D102
        """Get diffusion flux."""
        kappa_int = discr.project("vol", dd, kappa)
        kappa_tpair = TracePair(dd, interior=kappa_int, exterior=kappa_int)
        grad_u_int = discr.project("vol", dd, grad_u)
        grad_u_tpair = TracePair(dd, interior=grad_u_int, exterior=grad_u_int)
        return diffusion_flux(
            discr, kappa_tpair, grad_u_tpair, quadrature_tag=quadrature_tag)


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

    def get_grad_flux(
            self, discr, dd, u, *, quadrature_tag=DISCR_TAG_BASE):  # noqa: D102
        """Get gradient flux."""
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=u_int)
        return grad_flux(discr, u_tpair, quadrature_tag=quadrature_tag)

    def get_diffusion_flux(
            self, discr, dd, kappa, grad_u, *,
            quadrature_tag=DISCR_TAG_BASE):  # noqa: D102
        """Get diffusion flux."""
        dd_quad = dd.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_quad.with_dtag("all_faces")
        # Compute the flux directly instead of constructing an external grad_u value
        # (and the associated TracePair); this approach is simpler in the
        # spatially-varying kappa case (the other approach would result in a
        # grad_u_tpair that lives in the quadrature discretization; diffusion_flux
        # would need to be modified to accept such values).
        kappa_int_quad = discr.project("vol", dd_quad, kappa)
        value_quad = discr.project(dd, dd_quad, self.value)
        flux_quad = -kappa_int_quad*value_quad
        return discr.project(dd_quad, dd_allfaces_quad, flux_quad)


class _DiffusionStateTag:
    pass


class _DiffusionKappaTag:
    pass


class _DiffusionGradTag:
    pass


def grad_operator(discr, boundaries, u, quadrature_tag=DISCR_TAG_BASE):
    r"""
    Compute the gradient of *u*.

    Uses unstabilized central numerical fluxes.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    boundaries:
        dictionary (or list of dictionaries) mapping boundary tags to
        :class:`DiffusionBoundary` instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    quadrature_tag:
        quadrature tag indicating which discretization in *discr* to use for
        overintegration

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
                discr, boundaries_i, u_i, quadrature_tag=quadrature_tag),
            make_obj_array(boundaries), u)

    for btag, bdry in boundaries.items():
        if not isinstance(bdry, DiffusionBoundary):
            raise TypeError(f"Unrecognized boundary type for tag {btag}. "
                "Must be an instance of DiffusionBoundary.")

    dd_allfaces_quad = DOFDesc("all_faces", quadrature_tag)

    return discr.inverse_mass(
        discr.weak_grad(-u)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            sum(
                grad_flux(discr, u_tpair, quadrature_tag=quadrature_tag)
                for u_tpair in interior_trace_pairs(
                    discr, u, comm_tag=_DiffusionStateTag))
            + sum(
                bdry.get_grad_flux(
                    discr, as_dofdesc(btag), u, quadrature_tag=quadrature_tag)
                for btag, bdry in boundaries.items())
            )
        )


# Yuck
def _normalize_arguments(*args, **kwargs):
    if len(args) >= 2 and not isinstance(args[1], (dict, list)):
        # Old deprecated positional argument list
        pos_arg_names = ["kappa", "quad_tag", "boundaries", "u"]
    else:
        pos_arg_names = ["kappa", "boundaries", "u"]

    arg_dict = {
        arg_name: arg
        for arg_name, arg in zip(pos_arg_names[:len(args)], args)}
    arg_dict.update(kwargs)

    from warnings import warn

    if "alpha" in arg_dict:
        warn(
            "alpha argument is deprecated and will disappear in Q3 2022. "
            "Use kappa instead.", DeprecationWarning, stacklevel=3)
        kappa = arg_dict["alpha"]
    else:
        kappa = arg_dict["kappa"]

    boundaries = arg_dict["boundaries"]
    u = arg_dict["u"]

    if "quad_tag" in arg_dict:
        warn(
            "quad_tag argument is deprecated and will disappear in Q3 2022. "
            "Use quadrature_tag instead.", DeprecationWarning, stacklevel=3)
        quadrature_tag = arg_dict["quad_tag"]
    elif "quadrature_tag" in arg_dict:
        quadrature_tag = arg_dict["quadrature_tag"]
    else:
        # quadrature_tag is optional
        quadrature_tag = DISCR_TAG_BASE

    return kappa, boundaries, u, quadrature_tag


def diffusion_operator(discr, *args, return_grad_u=False, **kwargs):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    $\nabla\cdot(\kappa\nabla u)$, where $\kappa$ is the conductivity and
    $u$ is a scalar field.

    Uses unstabilized central numerical fluxes.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    kappa: numbers.Number or meshmode.dof_array.DOFArray
        the conductivity value(s)
    boundaries:
        dictionary (or list of dictionaries) mapping boundary tags to
        :class:`DiffusionBoundary` instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    return_grad_u: bool
        an optional flag indicating whether $\nabla u$ should also be returned
    quadrature_tag:
        quadrature tag indicating which discretization in *discr* to use for
        overintegration

    Returns
    -------
    diff_u: meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *u*
    grad_u: numpy.ndarray
        the gradient of *u*; only returned if *return_grad_u* is True
    """
    kappa, boundaries, u, quadrature_tag = _normalize_arguments(*args, **kwargs)

    if isinstance(u, np.ndarray):
        if not isinstance(boundaries, list):
            raise TypeError("boundaries must be a list if u is an object array")
        if len(boundaries) != len(u):
            raise TypeError("boundaries must be the same length as u")
        return obj_array_vectorize_n_args(
            lambda boundaries_i, u_i: diffusion_operator(
                discr, kappa, boundaries_i, u_i, return_grad_u=return_grad_u,
                quadrature_tag=quadrature_tag),
            make_obj_array(boundaries), u)

    for btag, bdry in boundaries.items():
        if not isinstance(bdry, DiffusionBoundary):
            raise TypeError(f"Unrecognized boundary type for tag {btag}. "
                "Must be an instance of DiffusionBoundary.")

    dd_quad = DOFDesc("vol", quadrature_tag)
    dd_allfaces_quad = DOFDesc("all_faces", quadrature_tag)

    grad_u = grad_operator(discr, boundaries, u, quadrature_tag=quadrature_tag)

    kappa_quad = discr.project("vol", dd_quad, kappa)
    grad_u_quad = discr.project("vol", dd_quad, grad_u)

    diff_u = discr.inverse_mass(
        discr.weak_div(dd_quad, -kappa_quad*grad_u_quad)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            sum(
                diffusion_flux(
                    discr, kappa_tpair, grad_u_tpair, quadrature_tag=quadrature_tag)
                for kappa_tpair, grad_u_tpair in zip(
                    interior_trace_pairs(discr, kappa, comm_tag=_DiffusionKappaTag),
                    interior_trace_pairs(discr, grad_u, comm_tag=_DiffusionGradTag)))
            + sum(
                bdry.get_diffusion_flux(
                    discr, as_dofdesc(btag), kappa, grad_u,
                    quadrature_tag=quadrature_tag)
                for btag, bdry in boundaries.items())
            )
        )

    if return_grad_u:
        return diff_u, grad_u
    else:
        return diff_u
