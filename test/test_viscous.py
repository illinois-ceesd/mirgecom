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
from meshmode.discretization.connection import FACE_RESTR_ALL
from meshmode.mesh import BTAG_ALL
from grudge.dof_desc import as_dofdesc
import grudge.op as op
from grudge.trace_pair import interior_trace_pairs
from mirgecom.discretization import create_discretization_collection

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from mirgecom.fluid import make_conserved
from mirgecom.transport import (
    SimpleTransport,
    PowerLawTransport
)
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.simutil import get_box_mesh
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

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = dcoll.zeros(actx)
    ones = zeros + 1.0

    # assemble velocities for simple, unique grad components
    velocity_x = nodes[0] + 2*nodes[1] + 3*nodes[2]
    velocity_y = 4*nodes[0] + 5*nodes[1] + 6*nodes[2]
    velocity_z = 7*nodes[0] + 8*nodes[1] + 9*nodes[2]
    velocity = make_obj_array([velocity_x, velocity_y, velocity_z])

    mass = 2*ones
    energy = zeros + 2.5 + .5*mass*np.dot(velocity, velocity)
    mom = mass * velocity

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    grad_cv = op.local_grad(dcoll, cv)

    if transport_model:
        tv_model = SimpleTransport(bulk_viscosity=1.0, viscosity=0.5)
    else:
        tv_model = PowerLawTransport()

    eos = IdealSingleGas()
    gas_model = GasModel(eos=eos, transport=tv_model)
    fluid_state = make_fluid_state(cv, gas_model)

    mu = tv_model.viscosity(cv=cv, dv=fluid_state.dv)
    lam = tv_model.volume_viscosity(cv=cv, dv=fluid_state.dv)

    # Exact answer for tau
    exp_grad_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    exp_grad_v_t = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    exp_div_v = 15
    exp_tau = (mu*(exp_grad_v + exp_grad_v_t)
               + lam*exp_div_v*np.eye(3))

    from mirgecom.viscous import viscous_stress_tensor
    tau = viscous_stress_tensor(fluid_state, grad_cv)

    # The errors come from grad_v
    assert actx.to_numpy(op.norm(dcoll, tau - exp_tau, np.inf)) < 1e-12


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

    eos = IdealSingleGas()
    gas_model = GasModel(eos=eos, transport=transport_model)

    from mirgecom.initializers import PlanarPoiseuille
    initializer = PlanarPoiseuille(density=rho, mu=mu)

    def _elbnd_flux(dcoll, compute_interior_flux, compute_boundary_flux,
                    int_tpairs, boundaries):
        return (
            sum(compute_interior_flux(int_tpair) for int_tpair in int_tpairs)
            + sum(compute_boundary_flux(as_dofdesc(bdtag)) for bdtag in boundaries))

    from mirgecom.flux import num_flux_central

    def cv_flux_interior(int_tpair):
        normal = actx.thaw(dcoll.normal(int_tpair.dd))
        from arraycontext import outer
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        dd_allfaces = int_tpair.dd.with_boundary_tag(FACE_RESTR_ALL)
        return op.project(dcoll, int_tpair.dd, dd_allfaces, flux_weak)

    def cv_flux_boundary(dd_bdry):
        boundary_discr = dcoll.discr_from_dd(dd_bdry)
        bnd_nodes = actx.thaw(boundary_discr.nodes())
        cv_bnd = initializer(x_vec=bnd_nodes, eos=eos)
        bnd_nhat = actx.thaw(dcoll.normal(dd_bdry))
        from grudge.trace_pair import TracePair
        bnd_tpair = TracePair(dd_bdry, interior=cv_bnd, exterior=cv_bnd)
        from arraycontext import outer
        flux_weak = outer(num_flux_central(bnd_tpair.int, bnd_tpair.ext), bnd_nhat)
        dd_allfaces = dd_bdry.with_boundary_tag(FACE_RESTR_ALL)
        return op.project(dcoll, dd_bdry, dd_allfaces, flux_weak)

    for nfac in [1, 2, 4]:

        nels_axis = nfac*(10, 20)
        box_ll = (left_boundary_location, ybottom)
        box_ur = (right_boundary_location, ytop)
        mesh = get_box_mesh(2, a=box_ll, b=box_ur, n=nels_axis)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        dcoll = create_discretization_collection(actx, mesh, order=order)
        nodes = actx.thaw(dcoll.nodes())

        def inf_norm(x):
            return actx.to_numpy(op.norm(dcoll, x, np.inf))  # noqa

        # compute max element size
        from grudge.dt_utils import h_max_from_volume
        h_max = h_max_from_volume(dcoll)

        # form exact cv
        cv = initializer(x_vec=nodes, eos=eos)
        cv_int_tpairs = interior_trace_pairs(dcoll, cv)
        boundaries = [BTAG_ALL]
        cv_flux_bnd = _elbnd_flux(dcoll, cv_flux_interior, cv_flux_boundary,
                                  cv_int_tpairs, boundaries)
        from mirgecom.operators import grad_operator
        dd_vol = as_dofdesc("vol")
        dd_allfaces = as_dofdesc("all_faces")
        grad_cv = grad_operator(dcoll, dd_vol, dd_allfaces, cv, cv_flux_bnd)

        xp_grad_cv = initializer.exact_grad(x_vec=nodes, eos=eos, cv_exact=cv)
        xp_grad_v = 1/cv.mass * xp_grad_cv.momentum
        xp_tau = mu * (xp_grad_v + xp_grad_v.transpose())

        # sanity check the gradient:
        relerr_scale_e = 1.0 / inf_norm(xp_grad_cv.energy)
        relerr_scale_p = 1.0 / inf_norm(xp_grad_cv.momentum)
        graderr_e = inf_norm(grad_cv.energy - xp_grad_cv.energy)
        graderr_p = inf_norm(grad_cv.momentum - xp_grad_cv.momentum)
        graderr_e *= relerr_scale_e
        graderr_p *= relerr_scale_p
        assert graderr_e < 5e-7
        assert graderr_p < 5e-11

        zeros = dcoll.zeros(actx)
        ones = zeros + 1
        pressure = eos.pressure(cv)
        # grad of p should be dp/dx
        xp_grad_p = make_obj_array([dpdx*ones, zeros])
        grad_p = op.local_grad(dcoll, pressure)
        dpscal = 1.0/np.abs(dpdx)

        temperature = eos.temperature(cv)
        tscal = rho*eos.gas_const()*dpscal
        xp_grad_t = xp_grad_p/(cv.mass*eos.gas_const())
        grad_t = op.local_grad(dcoll, temperature)

        # sanity check
        assert inf_norm(grad_p - xp_grad_p)*dpscal < 5e-9
        assert inf_norm(grad_t - xp_grad_t)*tscal < 5e-9

        fluid_state = make_fluid_state(cv, gas_model)
        # verify heat flux
        from mirgecom.viscous import conductive_heat_flux
        heat_flux = conductive_heat_flux(fluid_state, grad_t)
        xp_heat_flux = -kappa*xp_grad_t
        assert inf_norm(heat_flux - xp_heat_flux) < 2e-8

        xp_e_flux = np.dot(xp_tau, cv.velocity) - xp_heat_flux
        xp_mom_flux = xp_tau
        from mirgecom.viscous import viscous_flux
        vflux = viscous_flux(fluid_state, grad_cv, grad_t)

        efluxerr = (
            inf_norm(vflux.energy - xp_e_flux)
            / inf_norm(xp_e_flux)
        )
        momfluxerr = (
            inf_norm(vflux.momentum - xp_mom_flux)
            / inf_norm(xp_mom_flux)
        )

        assert inf_norm(vflux.mass) == 0
        e_eoc_rec.add_data_point(actx.to_numpy(h_max), efluxerr)
        p_eoc_rec.add_data_point(actx.to_numpy(h_max), momfluxerr)

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

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = dcoll.zeros(actx)
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

    grad_cv = op.local_grad(dcoll, cv)

    mu_b = 1.0
    mu = 0.5
    kappa = 5.0
    # assemble d_alpha so that every species has a unique j
    d_alpha = np.array([(ispec+1) for ispec in range(nspecies)])

    tv_model = SimpleTransport(bulk_viscosity=mu_b, viscosity=mu,
                               thermal_conductivity=kappa,
                               species_diffusivity=d_alpha)

    eos = IdealSingleGas()
    gas_model = GasModel(eos=eos, transport=tv_model)
    fluid_state = make_fluid_state(cv, gas_model)

    from mirgecom.viscous import diffusive_flux
    j = diffusive_flux(fluid_state, grad_cv)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    tol = 1e-10
    for idim in range(dim):
        ispec = 2*idim
        exact_dy = np.array([((ispec+1)*(idim*dim+1))*(iidim+1)
                             for iidim in range(dim)])
        exact_j = -massval * d_alpha[ispec] * exact_dy
        assert inf_norm(j[ispec] - exact_j) < tol
        exact_j = massval * d_alpha[ispec+1] * exact_dy
        assert inf_norm(j[ispec+1] - exact_j) < tol


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

    eos = IdealSingleGas()

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = dcoll.zeros(actx)
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

    # create a gradient of temperature
    temperature = nodes[0] + 2*nodes[1] + 3*nodes[2]

    massval = 2
    mass = massval*ones
    energy = eos.get_internal_energy(temperature) \
             + 0.5*massval*np.dot(velocity, velocity)
    mom = mass * velocity
    species_mass = mass*y

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)
    grad_cv = op.local_grad(dcoll, cv)
    grad_t = op.local_grad(dcoll, temperature)

    mu_b = 1.0
    mu = 0.5
    # create orthotropic heat conduction
    zeros = nodes[0]*0.0
    kappa = make_obj_array([0.1 + zeros, 0.2 + zeros, 0.3 + zeros])
    # assemble d_alpha so that every species has a unique j
    d_alpha = np.array([(ispec+1) for ispec in range(nspecies)])

    tv_model = SimpleTransport(bulk_viscosity=mu_b, viscosity=mu,
                               thermal_conductivity=kappa,
                               species_diffusivity=d_alpha)

    gas_model = GasModel(eos=eos, transport=tv_model)
    fluid_state = make_fluid_state(cv, gas_model)

    from mirgecom.viscous import diffusive_flux, conductive_heat_flux
    q = conductive_heat_flux(fluid_state, grad_t)
    j = diffusive_flux(fluid_state, grad_cv)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    tol = 1e-10
    for idim in range(dim):
        exact_q = -kappa[idim]*grad_t[idim]
        assert inf_norm(q[idim] - exact_q) < tol

    for idim in range(dim):
        ispec = 2*idim
        exact_dy = np.array([((ispec+1)*(idim*dim+1))*(iidim+1)
                             for iidim in range(dim)])
        exact_j = -massval * d_alpha[ispec] * exact_dy
        assert inf_norm(j[ispec] - exact_j) < tol
        exact_j = massval * d_alpha[ispec+1] * exact_dy
        assert inf_norm(j[ispec+1] - exact_j) < tol


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

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = dcoll.zeros(actx)
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
    eos = IdealSingleGas()

    dv = eos.dependent_vars(cv)

    d_alpha = tv_model.species_diffusivity(eos=eos, cv=cv, dv=dv)

    from mirgecom.viscous import get_local_max_species_diffusivity
    expected = .3*ones
    if array_valued:
        expected *= f
    calculated = get_local_max_species_diffusivity(actx, d_alpha)

    assert actx.to_numpy(op.norm(dcoll, calculated-expected, np.inf)) == 0


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

    dcoll = create_discretization_collection(actx, mesh, order=order)
    zeros = dcoll.zeros(actx)
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
    chlen = characteristic_lengthscales(actx, dcoll)
    from grudge.op import nodal_min
    chlen_min = nodal_min(dcoll, "vol", chlen)

    mu = mu*chlen_min
    if mu < 0:
        mu = 0
        tv_model = None
    else:
        tv_model = SimpleTransport(viscosity=mu)

    eos = IdealSingleGas()
    gas_model = GasModel(eos=eos, transport=tv_model)
    fluid_state = make_fluid_state(cv, gas_model)

    from mirgecom.viscous import get_viscous_timestep
    dt_field = get_viscous_timestep(dcoll, fluid_state)

    speed_total = fluid_state.wavespeed
    dt_expected = chlen / (speed_total + (mu / chlen))

    error = (dt_expected - dt_field) / dt_expected
    assert actx.to_numpy(op.norm(dcoll, error, np.inf)) == 0
