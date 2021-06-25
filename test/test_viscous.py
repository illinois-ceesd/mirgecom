"""Test the viscous fluid helper functions."""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
import numpy.random
import numpy.linalg as la  # noqa
import pyopencl.clmath  # noqa
import logging
import pytest  # noqa

from pytools.obj_array import make_obj_array
from meshmode.dof_array import thaw
import grudge.op as op
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from mirgecom.fluid import make_conserved
from mirgecom.transport import SimpleTransport
from mirgecom.eos import IdealSingleGas

logger = logging.getLogger(__name__)


def test_viscous_stress_tensor(actx_factory):
    """Test tau data structure and values against exact."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    # assemble velocities for simple, unique grad components
    velocity_x = nodes[0] + 2*nodes[1] + 3*nodes[2]
    velocity_y = 4*nodes[0] + 5*nodes[1] + 6*nodes[2]
    velocity_z = 7*nodes[0] + 8*nodes[1] + 9*nodes[2]
    velocity = make_obj_array([velocity_x, velocity_y, velocity_z])

    mass = 2*ones
    energy = zeros + 2.5
    mom = mass * velocity

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    grad_cv = make_conserved(dim, q=op.local_grad(discr, cv.join()))

    mu_b = 1.0
    mu = 0.5

    tv_model = SimpleTransport(bulk_viscosity=mu_b, viscosity=mu)

    eos = IdealSingleGas(transport_model=tv_model)

    # Exact answer for tau
    exp_grad_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    exp_grad_v_t = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    exp_grad_v_div = 15
    exp_tau = (mu*(exp_grad_v + exp_grad_v_t)
               + (mu_b - 2*mu/3)*exp_grad_v_div*np.eye(3))

    from mirgecom.viscous import viscous_stress_tensor
    tau = viscous_stress_tensor(discr, eos, cv, grad_cv)

    # The errors come from grad_v
    assert discr.norm(tau - exp_tau, np.inf) < 1e-12


def test_species_diffusive_flux(actx_factory):
    """Test species diffusive flux and values against exact."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    # assemble velocities for simple, unique grad components
    velocity_x = nodes[0] + 2*nodes[1] + 3*nodes[2]
    velocity_y = 4*nodes[0] + 5*nodes[1] + 6*nodes[2]
    velocity_z = 7*nodes[0] + 8*nodes[1] + 9*nodes[2]
    velocity = make_obj_array([velocity_x, velocity_y, velocity_z])

    # assemble y so that each one has simple, but unique grad components
    nspecies = 2*dim
    y = make_obj_array([ones for _ in range(nspecies)])
    for idim in range(dim):
        ispec = 2*idim
        y[ispec] = (ispec+1)*(idim*dim+1)*sum([(iidim+1)*nodes[iidim]
                                               for iidim in range(dim)])
        y[ispec+1] = -y[ispec]

    massval = 2
    mass = massval*ones
    energy = zeros + 2.5
    mom = mass * velocity
    species_mass = mass*y

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)

    grad_cv = make_conserved(dim, q=op.local_grad(discr, cv.join()))

    mu_b = 1.0
    mu = 0.5
    kappa = 5.0
    # assemble d_alpha so that every species has a unique j
    d_alpha = np.array([(ispec+1) for ispec in range(nspecies)])

    tv_model = SimpleTransport(bulk_viscosity=mu_b, viscosity=mu,
                               thermal_conductivity=kappa,
                               species_diffusivity=d_alpha)

    eos = IdealSingleGas(transport_model=tv_model)

    from mirgecom.viscous import diffusive_flux
    j = diffusive_flux(discr, eos, cv, grad_cv)

    tol = 1e-10
    for idim in range(dim):
        ispec = 2*idim
        exact_dy = np.array([((ispec+1)*(idim*dim+1))*(iidim+1)
                             for iidim in range(dim)])
        exact_j = -massval * d_alpha[ispec] * exact_dy
        assert discr.norm(j[ispec] - exact_j, np.inf) < tol
        exact_j = massval * d_alpha[ispec+1] * exact_dy
        assert discr.norm(j[ispec+1] - exact_j, np.inf) < tol


def test_diffusive_heat_flux(actx_factory):
    """Test diffusive heat flux and values against exact."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    # assemble velocities for simple, unique grad components
    velocity_x = nodes[0] + 2*nodes[1] + 3*nodes[2]
    velocity_y = 4*nodes[0] + 5*nodes[1] + 6*nodes[2]
    velocity_z = 7*nodes[0] + 8*nodes[1] + 9*nodes[2]
    velocity = make_obj_array([velocity_x, velocity_y, velocity_z])

    # assemble y so that each one has simple, but unique grad components
    nspecies = 2*dim
    y = make_obj_array([ones for _ in range(nspecies)])
    for idim in range(dim):
        ispec = 2*idim
        y[ispec] = (ispec+1)*(idim*dim+1)*sum([(iidim+1)*nodes[iidim]
                                               for iidim in range(dim)])
        y[ispec+1] = -y[ispec]

    massval = 2
    mass = massval*ones
    energy = zeros + 2.5
    mom = mass * velocity
    species_mass = mass*y

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)
    grad_cv = make_conserved(dim, q=op.local_grad(discr, cv.join()))

    mu_b = 1.0
    mu = 0.5
    kappa = 5.0
    # assemble d_alpha so that every species has a unique j
    d_alpha = np.array([(ispec+1) for ispec in range(nspecies)])

    tv_model = SimpleTransport(bulk_viscosity=mu_b, viscosity=mu,
                               thermal_conductivity=kappa,
                               species_diffusivity=d_alpha)

    eos = IdealSingleGas(transport_model=tv_model)

    from mirgecom.viscous import diffusive_flux
    j = diffusive_flux(discr, eos, cv, grad_cv)

    tol = 1e-10
    for idim in range(dim):
        ispec = 2*idim
        exact_dy = np.array([((ispec+1)*(idim*dim+1))*(iidim+1)
                             for iidim in range(dim)])
        exact_j = -massval * d_alpha[ispec] * exact_dy
        assert discr.norm(j[ispec] - exact_j, np.inf) < tol
        exact_j = massval * d_alpha[ispec+1] * exact_dy
        assert discr.norm(j[ispec+1] - exact_j, np.inf) < tol


def test_viscous_timestep(actx_factory):
    """Test timestep size."""
    actx = actx_factory()
    dim = 3
    nel_1d = 5

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    # assemble velocities for simple, unique grad components
    velocity_x = nodes[0] + 2*nodes[1] + 3*nodes[2]
    velocity_y = 4*nodes[0] + 5*nodes[1] + 6*nodes[2]
    velocity_z = 7*nodes[0] + 8*nodes[1] + 9*nodes[2]
    velocity = make_obj_array([velocity_x, velocity_y, velocity_z])

    # assemble y so that each one has simple, but unique grad components
    nspecies = 2*dim
    y = make_obj_array([ones for _ in range(nspecies)])
    for idim in range(dim):
        ispec = 2*idim
        y[ispec] = (ispec+1)*(idim*dim+1)*sum([(iidim+1)*nodes[iidim]
                                               for iidim in range(dim)])
        y[ispec+1] = -y[ispec]

    massval = 2
    mass = massval*ones
    energy = zeros + 2.5
    mom = mass * velocity
    species_mass = mass*y

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)

    # grad_cv = make_conserved(dim, q=op.local_grad(discr, cv.join()))

    mu_b = 1.0
    mu = 0.5
    kappa = 5.0
    # assemble d_alpha so that every species has a unique j
    d_alpha = np.array([(ispec+1) for ispec in range(nspecies)])

    tv_model = SimpleTransport(bulk_viscosity=mu_b, viscosity=mu,
                               thermal_conductivity=kappa,
                               species_diffusivity=d_alpha)

    eos = IdealSingleGas(transport_model=tv_model)

    from mirgecom.viscous import get_viscous_timestep
    timestep = get_viscous_timestep(discr, eos, cv)

    tol = 1e-9
    # TODO: avoid using characteristic_lengthscales
    from grudge.dt_utils import characteristic_lengthscales

    speed_total = actx.np.sqrt(np.dot(velocity, velocity)) + eos.sound_speed(cv)
    mu_arr = eos.transport_model()._make_array(mu, cv)
    actual = characteristic_lengthscales(cv.array_context, discr) / (
        speed_total + (
            mu_arr / characteristic_lengthscales(cv.array_context, discr)
        )
    )

    for i in range(actual.shape[0]):
        assert discr.norm(actual[i] - timestep[i], np.inf) < tol
