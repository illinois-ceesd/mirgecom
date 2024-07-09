r""":mod:`mirgecom.multiphysics.thermally_coupled_fluid_wall` for thermally-coupled
fluid and wall.

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
(:mod:`mirgecom.navierstokes`) with a wall subdomain governed by the heat
equation (:mod:`mirgecom.diffusion`) through temperature and heat flux. This
coupling can optionally include a sink term representing emitted radiation.
In the non-radiating case, coupling enforces continuity of temperature and heat flux

.. math::
    T_\text{wall} &= T_\text{fluid} \\
    -\kappa_\text{wall} \nabla T_\text{wall} \cdot \hat{n} &=
        -\kappa_\text{fluid} \nabla T_\text{fluid} \cdot \hat{n},

and in the radiating case, coupling enforces a similar condition but with an
additional radiation sink term in the heat flux

.. math::
    -\kappa_\text{wall} \nabla T_\text{wall} \cdot \hat{n} =
        -\kappa_\text{fluid} \nabla T_\text{fluid} \cdot \hat{n}
        + \epsilon \sigma (T^4 - T_\text{ambient}^4).

Boundary Setup Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: add_interface_boundaries_no_grad
.. autofunction:: add_interface_boundaries

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: InterfaceWallBoundary
.. autoclass:: InterfaceWallRadiationBoundary
"""

__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
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

from dataclasses import dataclass, replace
import numpy as np
# from abc import abstractmethod
# from functools import partial

from arraycontext import dataclass_array_container
from meshmode.dof_array import DOFArray
from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    # as_dofdesc,
)
# import grudge.op as op

# from mirgecom.flux import num_flux_central
from mirgecom.diffusion import (
    grad_facial_flux_weighted,
    # diffusion_flux,
    diffusion_facial_flux_harmonic,
    DiffusionBoundary,
    # grad_operator as wall_grad_t_operator,
    # diffusion_operator,
)
from mirgecom.multiphysics import make_interface_boundaries
from mirgecom.utils import project_from_base


class _ThermalDataNoGradInterVolTag:
    pass


class _ThermalDataInterVolTag:
    pass


class _WallGradTag:
    pass


class _WallOperatorTag:
    pass


def _replace_kappa(state, kappa):
    """Replace the thermal conductivity in fluid state *state* with *kappa*."""
    new_tv = replace(state.tv, thermal_conductivity=kappa)
    return replace(state, tv=new_tv)


# FIXME: Interior penalty should probably use an average of the lengthscales on
# both sides of the interface
class InterfaceWallBoundary(DiffusionBoundary):
    """
    Boundary for the wall side of the fluid-wall interface.

    .. automethod:: __init__
    .. automethod:: get_grad_flux
    .. automethod:: get_diffusion_flux
    """

    def __init__(self, kappa_plus, u_plus, grad_u_plus=None, gap_resistance=None):
        r"""
        Initialize InterfaceWallBoundary.

        Argument *grad_u_plus* is only required if the boundary will be used to
        compute the heat flux.

        Parameters
        ----------
        kappa_plus: float or :class:`meshmode.dof_array.DOFArray`

            Thermal conductivity from the fluid side.

        u_plus: :class:`meshmode.dof_array.DOFArray`

            Temperature from the fluid side.

        grad_u_plus: :class:`meshmode.dof_array.DOFArray` or None

            Temperature gradient from the fluid side.
        """
        self.kappa_plus = kappa_plus
        self.u_plus = u_plus
        self.grad_u_plus = grad_u_plus
        self._gap_resistance = gap_resistance
        if gap_resistance < 1e-10:
            print("YABADABADOO")
            sys.exit()

    def get_grad_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, *,
            numerical_flux_func=grad_facial_flux_weighted):  # noqa: D102
        actx = u_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        if self._gap_resistance is None:
            kappa_plus = project_from_base(dcoll, dd_bdry, self.kappa_plus)

            kappa_tpair = TracePair(
                dd_bdry, interior=kappa_minus, exterior=kappa_plus)

            u_plus = project_from_base(dcoll, dd_bdry, self.u_plus)
            u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)

            return numerical_flux_func(kappa_tpair, u_tpair, normal)
        else:
            return -u_minus*normal

    def get_diffusion_flux(
            self, dcoll, dd_bdry, kappa_minus, u_minus, grad_u_minus,
            lengthscales_minus, *, penalty_amount=None,
            numerical_flux_func=diffusion_facial_flux_harmonic):  # noqa: D102
        if self.grad_u_plus is None:
            raise TypeError(
                "Boundary does not have external gradient data.")

        actx = u_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        kappa_plus = project_from_base(dcoll, dd_bdry, self.kappa_plus)

        kappa_tpair = TracePair(
            dd_bdry, interior=kappa_minus, exterior=kappa_plus)

        u_plus = project_from_base(dcoll, dd_bdry, self.u_plus)
        u_tpair = TracePair(dd_bdry, interior=u_minus, exterior=u_plus)

        if self._gap_resistance is None:
            grad_u_plus = project_from_base(dcoll, dd_bdry, self.grad_u_plus)
            grad_u_tpair = TracePair(
                dd_bdry, interior=grad_u_minus, exterior=grad_u_plus)

            lengthscales_tpair = TracePair(
                dd_bdry, interior=lengthscales_minus, exterior=lengthscales_minus)

            return numerical_flux_func(
                kappa_tpair, u_tpair, grad_u_tpair, lengthscales_tpair, normal,
                penalty_amount=penalty_amount)
        else:
            return -(u_plus - u_minus)/self._gap_resistance  # TODO check sign


@dataclass_array_container
@dataclass(frozen=True)
class _ThermalDataNoGrad:
    kappa: DOFArray
    temperature: DOFArray


@dataclass_array_container
@dataclass(frozen=True)
class _ThermalData(_ThermalDataNoGrad):
    grad_temperature: np.ndarray


def _make_thermal_data(kappa, temperature, grad_temperature=None):
    if not isinstance(kappa, DOFArray):
        kappa = kappa * (0*temperature + 1)

    if grad_temperature is not None:
        thermal_data = _ThermalData(kappa, temperature, grad_temperature)
    else:
        thermal_data = _ThermalDataNoGrad(kappa, temperature)

    return thermal_data


def _get_interface_trace_pairs_no_grad(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        comm_tag):
    pairwise_thermal_data = {
        (wall_1_dd, wall_2_dd): (
            _make_thermal_data(wall_1_kappa, wall_1_temperature),
            _make_thermal_data(wall_2_kappa, wall_2_temperature))}
    return inter_volume_trace_pairs(
        dcoll, pairwise_thermal_data,
        comm_tag=(_ThermalDataNoGradInterVolTag, comm_tag))


def _get_interface_trace_pairs(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        wall_1_grad_temperature, wall_2_grad_temperature,
        comm_tag):

    pairwise_thermal_data = {
        (wall_1_dd, wall_2_dd): (
            _make_thermal_data(
                wall_1_kappa,
                wall_1_temperature,
                wall_1_grad_temperature),
            _make_thermal_data(
                wall_2_kappa,
                wall_2_temperature,
                wall_2_grad_temperature))}

    return inter_volume_trace_pairs(
        dcoll, pairwise_thermal_data,
        comm_tag=(_ThermalDataInterVolTag, comm_tag))


def _get_interface_boundaries_no_grad(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        interface_resistance,
        comm_tag):

    interface_tpairs = _get_interface_trace_pairs_no_grad(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        comm_tag)

    def make_wall_boundary(interface_tpair):
        return InterfaceWallBoundary(
            interface_tpair.ext.kappa,
            interface_tpair.ext.temperature,
            gap_resistance=interface_resistance)

    bdry_factories = {
        (wall_2_dd, wall_1_dd): make_wall_boundary,
        (wall_1_dd, wall_2_dd): make_wall_boundary}

    interface_boundaries = make_interface_boundaries(
        bdry_factories, interface_tpairs)

    wall_1_interface_boundaries = interface_boundaries[wall_2_dd, wall_1_dd]
    wall_2_interface_boundaries = interface_boundaries[wall_1_dd, wall_2_dd]

    return wall_1_interface_boundaries, wall_2_interface_boundaries


def _get_interface_boundaries(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        wall_1_grad_temperature, wall_2_grad_temperature,
        # wall_1_boundaries, wall_2_boundaries,
        interface_resistance,
        # wall_penalty_amount,
        # quadrature_tag,
        comm_tag):

    interface_tpairs = _get_interface_trace_pairs(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        wall_1_grad_temperature, wall_2_grad_temperature,
        comm_tag)

    def make_wall_boundary(interface_tpair):
        return InterfaceWallBoundary(
            interface_tpair.ext.kappa,
            interface_tpair.ext.temperature,
            grad_u_plus=interface_tpair.ext.grad_temperature,
            gap_resistance=interface_resistance)

    bdry_factories = {
        (wall_2_dd, wall_1_dd): make_wall_boundary,
        (wall_1_dd, wall_2_dd): make_wall_boundary}

    interface_boundaries = make_interface_boundaries(
        bdry_factories, interface_tpairs)

    wall_1_interface_boundaries = interface_boundaries[wall_2_dd, wall_1_dd]
    wall_2_interface_boundaries = interface_boundaries[wall_1_dd, wall_2_dd]

    return wall_1_interface_boundaries, wall_2_interface_boundaries


def add_interface_boundaries_no_grad(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        wall_1_boundaries, wall_2_boundaries,
        *,
        interface_resistance=None,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    """
    Parameters
    ----------

    dcoll: class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    wall_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the wall volume.

    wall_kappa: float or :class:`meshmode.dof_array.DOFArray`

        Thermal conductivity for the wall volume.

    wall_temperature: :class:`meshmode.dof_array.DOFArray`

        Temperature for the wall volume.

    wall_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the wall subdomain.

    interface_resistance:

    quadrature_tag

        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    comm_tag: Hashable
        Tag for distributed communication
    """
    wall_1_interface_boundaries_no_grad, wall_2_interface_boundaries_no_grad = \
        _get_interface_boundaries_no_grad(
            dcoll,
            wall_1_dd, wall_2_dd,
            wall_1_kappa, wall_2_kappa,
            wall_1_temperature, wall_2_temperature,
            interface_resistance,
            # quadrature_tag,
            comm_tag)

    wall_1_all_boundaries_no_grad = {}
    wall_1_all_boundaries_no_grad.update(wall_1_boundaries)
    wall_1_all_boundaries_no_grad.update(wall_1_interface_boundaries_no_grad)

    wall_2_all_boundaries_no_grad = {}
    wall_2_all_boundaries_no_grad.update(wall_2_boundaries)
    wall_2_all_boundaries_no_grad.update(wall_2_interface_boundaries_no_grad)

    return wall_1_all_boundaries_no_grad, wall_2_all_boundaries_no_grad


def add_interface_boundaries(
        dcoll,
        wall_1_dd, wall_2_dd,
        wall_1_kappa, wall_2_kappa,
        wall_1_temperature, wall_2_temperature,
        wall_1_grad_temperature, wall_2_grad_temperature,
        wall_1_boundaries, wall_2_boundaries,
        *,
        interface_resistance=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        comm_tag=None):
    """
    Include the wall-wall interface boundaries.

    Parameters
    ----------

    dcoll: class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    wall_dd: :class:`grudge.dof_desc.DOFDesc`

        DOF descriptor for the wall volume.

    wall_kappa: float or :class:`meshmode.dof_array.DOFArray`

        Thermal conductivity for the wall volume.

    wall_temperature: :class:`meshmode.dof_array.DOFArray`

        Temperature for the wall volume.

    wall_grad_temperature: numpy.ndarray

        Temperature gradient for the wall volume.

    wall_boundaries

        Dictionary of boundary functions, one for each valid non-interface
        :class:`~grudge.dof_desc.BoundaryDomainTag` on the wall subdomain.

    interface_resistance:

    wall_penalty_amount: float

        Coefficient $c$ for the interior penalty on the heat flux. See
        :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall.InterfaceFluidBoundary`
        for details.

    quadrature_tag

        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    comm_tag: Hashable
        Tag for distributed communication
    """
    if wall_penalty_amount is None:
        # FIXME: After verifying the form of the penalty term, figure out what value
        # makes sense to use as a default here
        wall_penalty_amount = 0.05

    wall_1_interface_boundaries, wall_2_interface_boundaries = \
        _get_interface_boundaries(
            dcoll,
            wall_1_dd, wall_2_dd,
            wall_1_kappa, wall_2_kappa,
            wall_1_temperature, wall_2_temperature,
            wall_1_grad_temperature, wall_2_grad_temperature,
            # wall_1_boundaries, wall_2_boundaries,
            interface_resistance,
            # wall_penalty_amount,
            # quadrature_tag,
            comm_tag)

    wall_1_all_boundaries = {}
    wall_1_all_boundaries.update(wall_1_boundaries)
    wall_1_all_boundaries.update(wall_1_interface_boundaries)

    wall_2_all_boundaries = {}
    wall_2_all_boundaries.update(wall_2_boundaries)
    wall_2_all_boundaries.update(wall_2_interface_boundaries)

    return wall_1_all_boundaries, wall_2_all_boundaries
