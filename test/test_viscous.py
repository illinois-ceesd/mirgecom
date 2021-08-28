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
from meshmode.mesh import BTAG_ALL
import grudge.op as op
from grudge.eager import (
    EagerDGDiscretization,
    interior_trace_pair
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from mirgecom.fluid import make_conserved
from mirgecom.transport import (
    SimpleTransport,
    PowerLawTransport
)
from mirgecom.eos import IdealSingleGas

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("transport_model", [0, 1])
def test_viscous_stress_tensor(actx_factory, transport_model):
    """Test tau data structure and values against exact."""
    actx = actx_factory()
    dim = 3
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
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

    if transport_model:
        tv_model = SimpleTransport(bulk_viscosity=1.0, viscosity=0.5)
    else:
        tv_model = PowerLawTransport()

    eos = IdealSingleGas(transport_model=tv_model)
    mu = tv_model.viscosity(eos, cv)
    lam = tv_model.volume_viscosity(eos, cv)

    # Exact answer for tau
    exp_grad_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    exp_grad_v_t = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    exp_div_v = 15
    exp_tau = (mu*(exp_grad_v + exp_grad_v_t)
               + lam*exp_div_v*np.eye(3))

    from mirgecom.viscous import viscous_stress_tensor
    tau = viscous_stress_tensor(discr, eos, cv, grad_cv)

    # The errors come from grad_v
    assert discr.norm(tau - exp_tau, np.inf) < 1e-12


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None):
    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, npoints_per_axis=n, boundary_tag_to_face=bttf, mesh_type=t)


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 2.3])
def test_poiseuille_fluxes(actx_factory, order, kappa):
    """Test the viscous fluxes using a Poiseuille input state."""
    actx = actx_factory()
    dim = 2

    from pytools.convergence import EOCRecorder
    e_eoc_rec = EOCRecorder()
    p_eoc_rec = EOCRecorder()

    base_pressure = 100000.0
    pressure_ratio = 1.001
    mu = 42  # arbitrary
    left_boundary_location = 0
    right_boundary_location = 0.1
    ybottom = 0.
    ytop = .02
    nspecies = 0
    spec_diffusivity = 0 * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    xlen = right_boundary_location - left_boundary_location
    p_low = base_pressure
    p_hi = pressure_ratio*base_pressure
    dpdx = (p_low - p_hi) / xlen
    rho = 1.0

    eos = IdealSingleGas(transport_model=transport_model)

    from mirgecom.initializers import PlanarPoiseuille
    initializer = PlanarPoiseuille(density=rho, mu=mu)

    def _elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                    int_tpair, boundaries):
        return (compute_interior_flux(int_tpair)
                + sum(compute_boundary_flux(btag) for btag in boundaries))

    from mirgecom.flux import gradient_flux_central

    def cv_flux_interior(int_tpair):
        normal = thaw(actx, discr.normal(int_tpair.dd))
        flux_weak = gradient_flux_central(int_tpair, normal)
        return discr.project(int_tpair.dd, "all_faces", flux_weak)

    def cv_flux_boundary(btag):
        boundary_discr = discr.discr_from_dd(btag)
        bnd_nodes = thaw(actx, boundary_discr.nodes())
        cv_bnd = initializer(x_vec=bnd_nodes, eos=eos)
        bnd_nhat = thaw(actx, discr.normal(btag))
        from grudge.trace_pair import TracePair
        bnd_tpair = TracePair(btag, interior=cv_bnd, exterior=cv_bnd)
        flux_weak = gradient_flux_central(bnd_tpair, bnd_nhat)
        return discr.project(bnd_tpair.dd, "all_faces", flux_weak)

    for nfac in [1, 2, 4]:

        npts_axis = nfac*(11, 21)
        box_ll = (left_boundary_location, ybottom)
        box_ur = (right_boundary_location, ytop)
        mesh = _get_box_mesh(2, a=box_ll, b=box_ur, n=npts_axis)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(actx, discr.nodes())

        # compute max element size
        from grudge.dt_utils import h_max_from_volume
        h_max = h_max_from_volume(discr)

        # form exact cv
        cv = initializer(x_vec=nodes, eos=eos)
        cv_int_tpair = interior_trace_pair(discr, cv)
        boundaries = [BTAG_ALL]
        cv_flux_bnd = _elbnd_flux(discr, cv_flux_interior, cv_flux_boundary,
                                  cv_int_tpair, boundaries)
        from mirgecom.operators import grad_operator
        grad_cv = make_conserved(dim, q=grad_operator(discr, cv.join(),
                                                      cv_flux_bnd.join()))

        xp_grad_cv = initializer.exact_grad(x_vec=nodes, eos=eos, cv_exact=cv)
        xp_grad_v = 1/cv.mass * xp_grad_cv.momentum
        xp_tau = mu * (xp_grad_v + xp_grad_v.transpose())

        # sanity check the gradient:
        relerr_scale_e = 1.0 / discr.norm(xp_grad_cv.energy, np.inf)
        relerr_scale_p = 1.0 / discr.norm(xp_grad_cv.momentum, np.inf)
        graderr_e = discr.norm((grad_cv.energy - xp_grad_cv.energy), np.inf)
        graderr_p = discr.norm((grad_cv.momentum - xp_grad_cv.momentum), np.inf)
        graderr_e *= relerr_scale_e
        graderr_p *= relerr_scale_p
        assert graderr_e < 5e-7
        assert graderr_p < 5e-11

        zeros = discr.zeros(actx)
        ones = zeros + 1
        pressure = eos.pressure(cv)
        # grad of p should be dp/dx
        xp_grad_p = make_obj_array([dpdx*ones, zeros])
        grad_p = op.local_grad(discr, pressure)
        dpscal = 1.0/np.abs(dpdx)

        temperature = eos.temperature(cv)
        tscal = rho*eos.gas_const()*dpscal
        xp_grad_t = xp_grad_p/(cv.mass*eos.gas_const())
        grad_t = op.local_grad(discr, temperature)

        # sanity check
        assert discr.norm(grad_p - xp_grad_p, np.inf)*dpscal < 5e-9
        assert discr.norm(grad_t - xp_grad_t, np.inf)*tscal < 5e-9

        # verify heat flux
        from mirgecom.viscous import conductive_heat_flux
        heat_flux = conductive_heat_flux(discr, eos, cv, grad_t)
        xp_heat_flux = -kappa*xp_grad_t
        assert discr.norm(heat_flux - xp_heat_flux, np.inf) < 2e-8

        # verify diffusive mass flux is zilch (no scalar components)
        from mirgecom.viscous import diffusive_flux
        j = diffusive_flux(discr, eos, cv, grad_cv)
        assert len(j) == 0

        xp_e_flux = np.dot(xp_tau, cv.velocity) - xp_heat_flux
        xp_mom_flux = xp_tau
        from mirgecom.viscous import viscous_flux
        vflux = viscous_flux(discr, eos, cv, grad_cv, grad_t)

        efluxerr = (
            discr.norm(vflux.energy - xp_e_flux, np.inf)
            / discr.norm(xp_e_flux, np.inf)
        )
        momfluxerr = (
            discr.norm(vflux.momentum - xp_mom_flux, np.inf)
            / discr.norm(xp_mom_flux, np.inf)
        )

        assert discr.norm(vflux.mass, np.inf) == 0
        e_eoc_rec.add_data_point(h_max, efluxerr)
        p_eoc_rec.add_data_point(h_max, momfluxerr)

    assert (
        e_eoc_rec.order_estimate() >= order - 0.5
        or e_eoc_rec.max_error() < 3e-9
    )
    assert (
        p_eoc_rec.order_estimate() >= order - 0.5
        or p_eoc_rec.max_error() < 2e-12
    )


def test_species_diffusive_flux(actx_factory):
    """Test species diffusive flux and values against exact."""
    actx = actx_factory()
    dim = 3
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
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
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
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


@pytest.mark.parametrize("array_valued", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_local_max_species_diffusivity(actx_factory, dim, array_valued):
    """Test the local maximum species diffusivity."""
    actx = actx_factory()
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    zeros = discr.zeros(actx)
    ones = zeros + 1.0
    vel = .32

    velocity = make_obj_array([zeros+vel for _ in range(dim)])

    massval = 1
    mass = massval*ones

    energy = zeros + 1.0 / (1.4*.4)
    mom = mass * velocity
    species_mass = np.array([1., 2., 3.], dtype=object)

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)

    d_alpha_input = np.array([.1, .2, .3])
    if array_valued:
        f = 1 + 0.1*actx.np.sin(nodes[0])
        d_alpha_input *= f

    tv_model = SimpleTransport(species_diffusivity=d_alpha_input)
    eos = IdealSingleGas(transport_model=tv_model)
    d_alpha = tv_model.species_diffusivity(eos, cv)

    from mirgecom.viscous import get_local_max_species_diffusivity
    expected = .3*ones
    if array_valued:
        expected *= f
    calculated = get_local_max_species_diffusivity(actx, discr, d_alpha)

    assert discr.norm(calculated-expected, np.inf) == 0


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("mu", [-1, 0, 1, 2])
@pytest.mark.parametrize("vel", [0, 1])
def test_viscous_timestep(actx_factory, dim, mu, vel):
    """Test timestep size."""
    actx = actx_factory()
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 1

    discr = EagerDGDiscretization(actx, mesh, order=order)
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    velocity = make_obj_array([zeros+vel for _ in range(dim)])

    massval = 1
    mass = massval*ones

    # I *think* this energy should yield c=1.0
    energy = zeros + 1.0 / (1.4*.4)
    mom = mass * velocity
    species_mass = None

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)

    from grudge.dt_utils import characteristic_lengthscales
    chlen = characteristic_lengthscales(actx, discr)
    from grudge.op import nodal_min
    chlen_min = nodal_min(discr, "vol", chlen)

    mu = mu*chlen_min
    if mu < 0:
        mu = 0
        tv_model = None
    else:
        tv_model = SimpleTransport(viscosity=mu)

    eos = IdealSingleGas(transport_model=tv_model)

    from mirgecom.viscous import get_viscous_timestep
    dt_field = get_viscous_timestep(discr, eos, cv)

    speed_total = actx.np.sqrt(np.dot(velocity, velocity)) + eos.sound_speed(cv)
    dt_expected = chlen / (speed_total + (mu / chlen))

    error = (dt_expected - dt_field) / dt_expected
    assert discr.norm(error, np.inf) == 0
