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
from meshmode.dof_array import thaw
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair


def _q_flux(discr, alpha, u_tpair):
    normal = thaw(u_tpair.int.array_context, discr.normal(u_tpair.dd))
    flux_weak = math.sqrt(alpha)*u_tpair.avg*normal
    return discr.project(u_tpair.dd, "all_faces", flux_weak)


def _u_flux(discr, alpha, q_tpair):
    normal = thaw(q_tpair.int[0].array_context, discr.normal(q_tpair.dd))
    flux_weak = math.sqrt(alpha)*np.dot(q_tpair.avg, normal)
    return discr.project(q_tpair.dd, "all_faces", flux_weak)


class DiffusionBoundary(metaclass=abc.ABCMeta):
    """
    Diffusion boundary base class.

    .. automethod:: get_q_flux
    .. automethod:: get_u_flux
    """

    @abc.abstractmethod
    def get_q_flux(self, discr, alpha, dd, u):
        """Compute the flux for *q* on the boundary corresponding to *dd*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_u_flux(self, discr, alpha, dd, q):
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
    def get_q_flux(self, discr, alpha, dd, u):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=2.*self.value-u_int)
        return _q_flux(discr, alpha, u_tpair)

    def get_u_flux(self, discr, alpha, dd, q):  # noqa: D102
        q_int = discr.project("vol", dd, q)
        q_tpair = TracePair(dd, interior=q_int, exterior=q_int)
        return _u_flux(discr, alpha, q_tpair)


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

    def get_q_flux(self, discr, alpha, dd, u):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=u_int)
        return _q_flux(discr, alpha, u_tpair)

    def get_u_flux(self, discr, alpha, dd, q):  # noqa: D102
        ones = discr.zeros(q[0].array_context) + 1.
        bdry_ones = discr.project("vol", dd, ones)
        # Computing the flux directly instead of constructing an external q value
        # avoids some potential nastiness with sqrt(alpha) and overintegration in
        # the variable alpha case
        flux_weak = alpha*self.value*bdry_ones
        return discr.project(dd, "all_faces", flux_weak)


def diffusion_operator(discr, alpha, boundaries, u):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    $\nabla\cdot(\alpha\nabla u)$, where $\alpha$ is the diffusivity and
    $u$ is a scalar field.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    alpha: float
        the (constant) diffusivity
    boundaries:
        dictionary (or list of dictionaries) mapping boundary tags to
        :class:`DiffusionBoundary` instances
    u: meshmode.dof_array.DOFArray or numpy.ndarray
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
            diffusion_operator(discr, alpha, boundaries_i, u_i),
            make_obj_array(boundaries), u)

    for btag, bdry in boundaries.items():
        if not isinstance(bdry, DiffusionBoundary):
            raise TypeError(f"Unrecognized boundary type for tag {btag}. "
                "Must be an instance of DiffusionBoundary.")

    q = discr.inverse_mass(
        -math.sqrt(alpha)*discr.weak_grad(u)
        +  # noqa: W504
        discr.face_mass(
            _q_flux(discr, alpha=alpha, u_tpair=interior_trace_pair(discr, u))
            + sum(
                bdry.get_q_flux(discr, alpha=alpha, dd=btag, u=u)
                for btag, bdry in boundaries.items()
            )
            + sum(
                _q_flux(discr, alpha=alpha, u_tpair=tpair)
                for tpair in cross_rank_trace_pairs(discr, u)
            )
        ))

    return (
        discr.inverse_mass(
            -math.sqrt(alpha)*discr.weak_div(q)
            +  # noqa: W504
            discr.face_mass(
                _u_flux(discr, alpha=alpha, q_tpair=interior_trace_pair(discr, q))
                + sum(
                    bdry.get_u_flux(discr, alpha=alpha, dd=btag, q=q)
                    for btag, bdry in boundaries.items()
                )
                + sum(
                    _u_flux(discr, alpha=alpha, q_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, q))
                )
            )
        )
