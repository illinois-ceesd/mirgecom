r""":mod:`mirgecom.diffusion` computes the diffusion operator.

Diffusion equation:

.. math::

    \frac{\partial u}{\partial t} = \nabla \cdot (\boldsymbol{\kappa} \nabla u)

where:

- conserved variable $u$
- scalar (isotropic) or diagonal-tensor (orthotropic) $\boldsymbol{\kappa}$

In the orthotropic case, the product $\boldsymbol{\kappa} \nabla u$ is treated
as a Hadamard product of arrays rather than a matrix-array multiplication.
Fully anisotropic materials are not currently handled.

Due to the possibility of big differences in the magnitude of diffusivity
coefficients, such as thermal conductivity in air and a solid, an harmonic
average can be used to increase robustness and avoid numerical instabilities.
In this case, to ensure flux continuity,

.. math::

    F = \kappa^- \nabla T^- = \kappa^+ \nabla T^+
        = \bar{\kappa} \left( \frac{\nabla T^- + \nabla T^+}{2} \right)

where

.. math::
    \bar{\kappa} = \frac{2 \kappa^-_{ii} \kappa^+_{ii}}
        {\kappa^-_{ii} + \kappa^+_{ii}}

with $\kappa_{ii}$ being either the individual components of the diffusivity
array (orthotropic material) or a single scalar (isotropic).

Flux functions
^^^^^^^^^^^^^^

.. autofunction:: diffusion_flux
.. autofunction:: grad_facial_flux_central
.. autofunction:: grad_facial_flux_weighted
.. autofunction:: diffusion_facial_flux_central
.. autofunction:: diffusion_facial_flux_harmonic

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: grad_operator
.. autofunction:: diffusion_operator

Boundary conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: DiffusionBoundary
.. autoclass:: DirichletDiffusionBoundary
.. autoclass:: NeumannDiffusionBoundary
.. autoclass:: RobinDiffusionBoundary
.. autoclass:: PrescribedFluxDiffusionBoundary
.. autoclass:: DummyDiffusionBoundary
"""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
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
from pytools.obj_array import make_obj_array, obj_array_vectorize_n_args
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
from grudge import op
from mirgecom.math import harmonic_mean
from mirgecom.utils import normalize_boundaries


def grad_facial_flux_central(kappa_tpair, u_tpair, normal):
    r"""Compute the numerical flux for $\nabla u$.

    Uses a simple average of the two sides' values:

    .. math::

        F = -\frac{u^- + u^+}{2} \hat{n}.
    """
    return -u_tpair.avg * normal


def grad_facial_flux_weighted(kappa_tpair, u_tpair, normal):
    r"""Compute the numerical flux for $\nabla u$ using a weighted average.

    Weights each side's value by the corresponding thermal conductivity $\kappa$:

    .. math::

        F = -\frac{\kappa^- u^- + \kappa^+ u^+}{\kappa^- + \kappa^+} \hat{n}

    For an orthotropic material, uses an averaging wrt to the normal component
    according to

    .. math::

        \kappa = n^T \cdot \kappa \cdot n
    """
    actx = u_tpair.int.array_context

    # If any of the coefficients are orthotropic, weight by the normal.
    if isinstance(kappa_tpair.int, np.ndarray):
        kappa_int = np.dot(normal, kappa_tpair.int*normal)
    else:
        kappa_int = kappa_tpair.int

    if isinstance(kappa_tpair.ext, np.ndarray):
        kappa_ext = np.dot(normal, kappa_tpair.ext*normal)
    else:
        kappa_ext = kappa_tpair.ext

    kappa_sum = actx.np.where(
        actx.np.greater(kappa_int + kappa_ext, 0*kappa_int),
        kappa_int + kappa_ext,
        0*kappa_int + 1)

    return -(u_tpair.int*kappa_int + u_tpair.ext*kappa_ext)/kappa_sum * normal


def diffusion_flux(kappa, grad_u):
    r"""
    Compute the diffusive flux $-\kappa \nabla u$.

    Parameters
    ----------
    kappa: float, numpy.ndarray or :class:`meshmode.dof_array.DOFArray`

        The thermal conductivity.

    grad_u: numpy.ndarray

        Gradient of the state variable *u*.

    Returns
    -------
    meshmode.dof_array.DOFArray

        The diffusive flux.
    """
    return -kappa * grad_u


def diffusion_facial_flux_central(
        kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal, *,
        penalty_amount=None):
    r"""Compute the numerical flux for $\nabla \cdot (\kappa \nabla u)$.

    Uses a simple average of the two sides' values:

    .. math::

        F = -\frac{\kappa^- \nabla u^- + \kappa^+ \nabla u^+}{2} \cdot \hat{n}
                - \tau (u^+ - u^-).

    The amount of penalization $\tau$ is given by

    .. math::

        \tau = \frac{\alpha \bar{\kappa}_{avg}}{l},

    where $\alpha$ is a user-definied value, $l$ is the element characteristic
    lengthscale and $\bar{\kappa}_{avg}$ is the averaged value, considering
    both isotropic (scalar) or orthotropic (array) cases. In the latter, the
    normal value is used for the penalization (see [Ern_2008]_).
    """
    if penalty_amount is None:
        # FIXME: After verifying the form of the penalty term, figure out what value
        # makes sense to use as a default here
        penalty_amount = 0.05

    flux_tpair = TracePair(grad_u_tpair.dd,
        interior=diffusion_flux(kappa_tpair.int, grad_u_tpair.int),
        exterior=diffusion_flux(kappa_tpair.ext, grad_u_tpair.ext))

    flux_without_penalty = np.dot(flux_tpair.avg, normal)

    # TODO: Verify that this is the correct form for the penalty term
    if isinstance(kappa_tpair.avg, np.ndarray):
        kappa_avg_normal = np.dot(normal, kappa_tpair.avg.int*normal)
        tau = penalty_amount*kappa_avg_normal/lengthscales_tpair.avg
    else:
        tau = penalty_amount*kappa_tpair.avg/lengthscales_tpair.avg

    return flux_without_penalty - tau*(u_tpair.ext - u_tpair.int)


def diffusion_facial_flux_harmonic(
        kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal, *,
        penalty_amount=None):
    r"""Compute the numerical flux for $\nabla \cdot (\kappa \nabla u)$.

    Uses a modified average of the two sides' values that replaces $\kappa^-$
    and $\kappa^+$ with their harmonic mean, plus a penalization term

    .. math::

        F = -\frac{2 \kappa_{ii}^- \kappa_{ii}^+}{\kappa_{ii}^- + \kappa_{ii}^+}
                \frac{\nabla u^- + \nabla u^+}{2} \cdot \hat{n} - \tau (u^+ - u^-).

    The amount of penalization $\tau$ is given by

    .. math::

        \tau = \frac{\alpha \bar{\kappa}_{harm}}{l},

    where $\alpha$ is a user-defined value, $l$ is the element characteristic
    lengthscale and $\bar{\kappa}_{harm}$ is the harmonic mean, considering
    both isotropic (scalar) or orthotropic (array) cases. In the latter, the
    normal value is used for the penalization (see [Ern_2008]_).
    """
    if penalty_amount is None:
        # FIXME: After verifying the form of the penalty term, figure out what value
        # makes sense to use as a default here
        penalty_amount = 0.05

    kappa_harmonic_mean = harmonic_mean(kappa_tpair.int, kappa_tpair.ext)

    flux_tpair = TracePair(grad_u_tpair.dd,
        interior=diffusion_flux(kappa_harmonic_mean, grad_u_tpair.int),
        exterior=diffusion_flux(kappa_harmonic_mean, grad_u_tpair.ext))

    flux_without_penalty = np.dot(flux_tpair.avg, normal)

    # TODO: Verify that this is the correct form for the penalty term
    if isinstance(kappa_harmonic_mean, np.ndarray):
        # if orthotropic, weight by the normal
        kappa_mean_normal = np.dot(normal, kappa_harmonic_mean*normal)
        tau = penalty_amount*kappa_mean_normal/lengthscales_tpair.avg
    else:
        tau = penalty_amount*kappa_harmonic_mean/lengthscales_tpair.avg

    return flux_without_penalty - tau*(u_tpair.ext - u_tpair.int)


class DiffusionBoundary(metaclass=abc.ABCMeta):
    """
    Diffusion boundary base class.

    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    @abc.abstractmethod
    def get_grad_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):
        """Compute the flux for grad(u) on the boundary *dd_bdry*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):
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
            self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=2*self.value-u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=2*self.value-u_minus)
        grad_u_tpair = TracePair(dd_bdry,
            interior=grad_u_minus,
            exterior=grad_u_minus)
        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(
            kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal,
            penalty_amount=penalty_amount)


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
            self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        grad_u_tpair = TracePair(dd_bdry,
            interior=grad_u_minus,
            exterior=(
                grad_u_minus
                + 2 * (self.value - np.dot(grad_u_minus, normal)) * normal))
        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)
        return numerical_flux_func(
            kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal,
            penalty_amount=penalty_amount)


class RobinDiffusionBoundary(DiffusionBoundary):
    r"""
    Robin boundary condition for the diffusion operator.

    The non-homogeneous Robin boundary condition is a linear combination of
    $u$ and its gradient $\nabla u$, given by

    .. math::

        (\alpha u - \kappa \nabla u \cdot \mathbf{\hat{n}})|_\Gamma =
            \alpha u_{ref}.

    where $u_{ref}$ is the reference value of $u$ for $x \to \infty$ and
    $\alpha$ is the weight for $u$. The gradient weight $\kappa$ is the
    conductivity (thermal) or the diffusivity (species).

    The current implementation uses external data

    .. math::

        u^+ = u^-

    when computing the boundary fluxes for $\nabla u$, and

    .. math::

        \nabla u\cdot\mathbf{\hat{n}} |_\Gamma =
            \frac{\alpha}{\kappa}(u_{ref} - u^-)

    when computing the boundary fluxes for $\nabla \cdot (\kappa \nabla u)$.

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(self, u_ref, alpha):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        u_ref: float or meshmode.dof_array.DOFArray
            the reference value(s) of $u$ along the boundary
        alpha: float or meshmode.dof_array.DOFArray
            the weight for the variable $u$ at the boundary
        """
        self.u_ref = u_ref
        self.alpha = alpha

    def get_grad_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        dudn_bc = self.alpha * (self.u_ref - u_minus)/kappa_minus
        grad_u_tpair = TracePair(dd_bdry,
            interior=grad_u_minus,
            exterior=(
                grad_u_minus
                + 2 * (dudn_bc - np.dot(grad_u_minus, normal)) * normal))
        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)
        return numerical_flux_func(
            kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal,
            penalty_amount=penalty_amount)


class PrescribedFluxDiffusionBoundary(DiffusionBoundary):
    r"""
    Prescribed flux boundary condition for the diffusion operator.

    For the boundary condition $(\nabla u \cdot \mathbf{\hat{n}})|_\Gamma$, uses
    external data

    .. math::

        u^+ = u^-

    when computing the boundary fluxes for $\nabla u$, and applies directly
    the prescribed flux $g$ when computing $\nabla \cdot (\kappa \nabla u)$:

    .. math::

        f_{presc} \cdot \hat{n}

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
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

    def get_grad_flux(self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_minus)
        u_tpair = TracePair(
            dd_bdry, interior=u_minus, exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):  # noqa: D102
        actx = u_minus.array_context

        # returns the product "flux @ normal"
        return actx.np.zeros_like(u_minus) + self.value


class DummyDiffusionBoundary(DiffusionBoundary):
    """Dummy boundary condition that duplicates the internal values."""

    def get_grad_flux(self, dcoll, dd_bdry, kappa_minus, u_minus, *,
                      numerical_flux_func):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(kappa_tpair, u_tpair, normal)

    def get_diffusion_flux(self, dcoll, dd_bdry, kappa_minus, u_minus,
                           grad_u_minus, lengthscales_minus, *,
                           numerical_flux_func=diffusion_facial_flux_harmonic,
                           penalty_amount=None):  # noqa: D102
        actx = u_minus.array_context
        kappa_tpair = TracePair(dd_bdry,
            interior=kappa_minus,
            exterior=kappa_minus)
        u_tpair = TracePair(dd_bdry,
            interior=u_minus,
            exterior=u_minus)
        grad_u_tpair = TracePair(dd_bdry,
            interior=grad_u_minus,
            exterior=grad_u_minus)
        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(
            kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal,
            penalty_amount=penalty_amount)


class _DiffusionKappaTag:
    pass


class _DiffusionStateTag:
    pass


class _DiffusionGradTag:
    pass


class _DiffusionLengthscalesTag:
    pass


def grad_operator(
        dcoll, kappa, boundaries, u, *, quadrature_tag=DISCR_TAG_BASE,
        dd=DD_VOLUME_ALL, comm_tag=None,
        numerical_flux_func=grad_facial_flux_weighted,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        kappa_tpairs=None,
        u_tpairs=None):
    r"""
    Compute the gradient of *u*.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    kappa: numbers.Number or meshmode.dof_array.DOFArray
        the conductivity value(s)
    boundaries:
        dictionary (or list of dictionaries) mapping boundary tags to
        :class:`DiffusionBoundary` instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    numerical_flux_func:
        function that computes the numerical gradient flux
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
                dcoll, kappa, boundaries_i, u_i, quadrature_tag=quadrature_tag,
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

    if kappa_tpairs is None:
        kappa_tpairs = interior_trace_pairs(
            dcoll, kappa, volume_dd=dd_vol, comm_tag=(_DiffusionKappaTag, comm_tag))

    if u_tpairs is None:
        u_tpairs = interior_trace_pairs(
            dcoll, u, volume_dd=dd_vol, comm_tag=(_DiffusionStateTag, comm_tag))

    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    def interior_flux(kappa_tpair, u_tpair):
        dd_trace_quad = kappa_tpair.dd.with_discr_tag(quadrature_tag)
        kappa_tpair_quad = interp_to_surf_quad(kappa_tpair)
        u_tpair_quad = interp_to_surf_quad(u_tpair)
        normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
        return op.project(
            dcoll, dd_trace_quad, dd_allfaces_quad,
            numerical_flux_func(kappa_tpair_quad, u_tpair_quad, normal_quad))

    def boundary_flux(bdtag, bdry):
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
        kappa_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, kappa)
        u_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, u)
        return op.project(
            dcoll, dd_bdry_quad, dd_allfaces_quad,
            bdry.get_grad_flux(
                dcoll, dd_bdry_quad, kappa_minus_quad, u_minus_quad,
                numerical_flux_func=numerical_flux_func))

    return op.inverse_mass(
        dcoll, dd_vol_quad,
        op.weak_local_grad(dcoll, dd_vol_quad, -u)
        -  # noqa: W504
        op.face_mass(
            dcoll, dd_allfaces_quad,
            sum(
                interior_flux(kappa_tpair, u_tpair)
                for kappa_tpair, u_tpair in zip(kappa_tpairs, u_tpairs))
            + sum(
                boundary_flux(bdtag, bdry)
                for bdtag, bdry in boundaries.items())
            )
        )


def diffusion_operator(
        dcoll, kappa, boundaries, u, *, return_grad_u=False, penalty_amount=None,
        gradient_numerical_flux_func=grad_facial_flux_weighted,
        diffusion_numerical_flux_func=diffusion_facial_flux_harmonic,
        quadrature_tag=DISCR_TAG_BASE, dd=DD_VOLUME_ALL, comm_tag=None,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        grad_u=None):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    $\nabla\cdot(\kappa\nabla u)$, where $\kappa$ is the conductivity and
    $u$ is a scalar field.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    kappa: numbers.Number or meshmode.dof_array.DOFArray or numpy.ndarray
        the conductivity value(s)
    boundaries:
        dictionary (or list of dictionaries) mapping boundary domain tags to
        :class:`DiffusionBoundary` instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    return_grad_u: bool
        an optional flag indicating whether $\nabla u$ should also be returned
    penalty_amount: float
        strength parameter for the diffusion flux interior penalty (temporary?);
        the default value is 0.05
    gradient_numerical_flux_func:
        function that computes the numerical gradient flux
    diffusion_numerical_flux_func:
        function that computes the numerical diffusive flux
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
        if isinstance(kappa, np.ndarray):
            if len(kappa) != len(u):
                raise TypeError("kappa must be the same length as u")
        else:
            kappa = make_obj_array([kappa for _ in range(len(u))])
        return obj_array_vectorize_n_args(
            lambda kappa_i, boundaries_i, u_i: diffusion_operator(
                dcoll, kappa_i, boundaries_i, u_i, return_grad_u=return_grad_u,
                penalty_amount=penalty_amount,
                gradient_numerical_flux_func=gradient_numerical_flux_func,
                diffusion_numerical_flux_func=diffusion_numerical_flux_func,
                quadrature_tag=quadrature_tag, dd=dd),
            kappa, make_obj_array(boundaries), u)

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

    kappa_tpairs = interior_trace_pairs(
        dcoll, kappa, volume_dd=dd_vol, comm_tag=(_DiffusionKappaTag, comm_tag))

    u_tpairs = interior_trace_pairs(
        dcoll, u, volume_dd=dd_vol, comm_tag=(_DiffusionStateTag, comm_tag))

    if grad_u is None:
        grad_u = grad_operator(
            dcoll, kappa, boundaries, u,
            numerical_flux_func=gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag, dd=dd_vol, comm_tag=comm_tag,
            kappa_tpairs=kappa_tpairs, u_tpairs=u_tpairs)

    grad_u_tpairs = interior_trace_pairs(
        dcoll, grad_u, volume_dd=dd_vol,
        comm_tag=(_DiffusionGradTag, comm_tag))

    kappa_quad = op.project(dcoll, dd_vol, dd_vol_quad, kappa)
    grad_u_quad = op.project(dcoll, dd_vol, dd_vol_quad, grad_u)

    from grudge.dt_utils import characteristic_lengthscales
    lengthscales = characteristic_lengthscales(actx, dcoll, dd=dd_vol)*(0*u+1)

    lengthscales_tpairs = interior_trace_pairs(
        dcoll, lengthscales, volume_dd=dd_vol,
        comm_tag=(_DiffusionLengthscalesTag, comm_tag))

    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    def interior_flux(kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair):
        dd_trace_quad = u_tpair.dd.with_discr_tag(quadrature_tag)
        u_tpair_quad = interp_to_surf_quad(u_tpair)
        kappa_tpair_quad = interp_to_surf_quad(kappa_tpair)
        grad_u_tpair_quad = interp_to_surf_quad(grad_u_tpair)
        lengthscales_tpair_quad = interp_to_surf_quad(lengthscales_tpair)
        normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
        return op.project(
            dcoll, dd_trace_quad, dd_allfaces_quad,
            diffusion_numerical_flux_func(
                kappa_tpair_quad, u_tpair_quad, grad_u_tpair_quad,
                lengthscales_tpair_quad, normal_quad,
                penalty_amount=penalty_amount))

    def boundary_flux(bdtag, bdry):
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
        u_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, u)
        kappa_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, kappa)
        grad_u_minus_quad = op.project(dcoll, dd_vol, dd_bdry_quad, grad_u)
        lengthscales_minus_quad = op.project(
            dcoll, dd_vol, dd_bdry_quad, lengthscales)
        return op.project(
            dcoll, dd_bdry_quad, dd_allfaces_quad,
            bdry.get_diffusion_flux(
                dcoll, dd_bdry_quad, kappa_minus_quad, u_minus_quad,
                grad_u_minus_quad, lengthscales_minus_quad,
                penalty_amount=penalty_amount,
                numerical_flux_func=diffusion_numerical_flux_func))

    diff_u = op.inverse_mass(
        dcoll, dd_vol_quad,
        op.weak_local_div(dcoll, dd_vol_quad, -kappa_quad*grad_u_quad)
        -  # noqa: W504
        op.face_mass(
            dcoll, dd_allfaces_quad,
            sum(
                interior_flux(kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair)
                for kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair in zip(
                    kappa_tpairs, u_tpairs, grad_u_tpairs, lengthscales_tpairs))
            + sum(
                boundary_flux(bdtag, bdry)
                for bdtag, bdry in boundaries.items())
            )
        )

    if return_grad_u:
        return diff_u, grad_u
    else:
        return diff_u
