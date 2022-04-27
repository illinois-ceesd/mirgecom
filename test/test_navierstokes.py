"""Test the Navier-Stokes gas dynamics module."""

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

import numpy as np
import numpy.random
import numpy.linalg as la  # noqa
import pyopencl.clmath  # noqa
import logging
import pytest

from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)

from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.navierstokes import ns_operator
from mirgecom.fluid import make_conserved
from grudge.dof_desc import DTAG_BOUNDARY

from mirgecom.boundary import (
    DummyBoundary,
    PrescribedFluidBoundary,
    AdiabaticNoslipMovingBoundary
)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("nspecies", [0, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_uniform_rhs(actx_factory, nspecies, dim, order):
    """Test the Navier-Stokes operator using a trivial constant/uniform state.

    This state should yield rhs = 0 to FP.  The test is performed for 1, 2,
    and 3 dimensions, with orders 1, 2, and 3, with and without passive species.
    """
    actx = actx_factory()

    tolerance = 1e-9

    from pytools.convergence import EOCRecorder
    eoc_rec0 = EOCRecorder()
    eoc_rec1 = EOCRecorder()
    # for nel_1d in [4, 8, 12]:
    for nel_1d in [4, 8]:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
        )

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
        from meshmode.discretization.poly_element import \
            default_simplex_group_factory, QuadratureSimplexGroupFactory

        discr = EagerDGDiscretization(
            actx, mesh,
            discr_tag_to_group_factory={
                DISCR_TAG_BASE: default_simplex_group_factory(
                    base_dim=mesh.dim, order=order),
                DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
            }
        )

        zeros = discr.zeros(actx)
        ones = zeros + 1.0

        mass_input = discr.zeros(actx) + 1
        energy_input = discr.zeros(actx) + 2.5

        mom_input = make_obj_array(
            [discr.zeros(actx) for i in range(discr.dim)]
        )

        mass_frac_input = flat_obj_array(
            [ones / ((i + 1) * 10) for i in range(nspecies)]
        )
        species_mass_input = mass_input * mass_frac_input
        num_equations = dim + 2 + len(species_mass_input)

        cv = make_conserved(
            dim, mass=mass_input, energy=energy_input, momentum=mom_input,
            species_mass=species_mass_input)

        expected_rhs = make_conserved(
            dim, q=make_obj_array([discr.zeros(actx)
                                   for i in range(num_equations)])
        )
        mu = 1.0
        kappa = 0.0
        spec_diffusivity = 0 * np.ones(nspecies)

        from mirgecom.gas_model import GasModel, make_fluid_state
        gas_model = GasModel(
            eos=IdealSingleGas(),
            transport=SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity))
        state = make_fluid_state(gas_model=gas_model, cv=cv)

        boundaries = {BTAG_ALL: DummyBoundary()}

        ns_rhs = ns_operator(discr, gas_model=gas_model, boundaries=boundaries,
                             state=state, time=0.0)

        rhs_resid = ns_rhs - expected_rhs
        rho_resid = rhs_resid.mass
        rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum
        rhoy_resid = rhs_resid.species_mass

        rho_rhs = ns_rhs.mass
        rhoe_rhs = ns_rhs.energy
        rhov_rhs = ns_rhs.momentum
        rhoy_rhs = ns_rhs.species_mass

        logger.info(
            f"rho_rhs  = {rho_rhs}\n"
            f"rhoe_rhs = {rhoe_rhs}\n"
            f"rhov_rhs = {rhov_rhs}\n"
            f"rhoy_rhs = {rhoy_rhs}\n"
        )

        assert actx.to_numpy(discr.norm(rho_resid, np.inf)) < tolerance
        assert actx.to_numpy(discr.norm(rhoe_resid, np.inf)) < tolerance
        for i in range(dim):
            assert actx.to_numpy(discr.norm(mom_resid[i], np.inf)) < tolerance
        for i in range(nspecies):
            assert actx.to_numpy(discr.norm(rhoy_resid[i], np.inf)) < tolerance

        err_max = actx.to_numpy(discr.norm(rho_resid, np.inf))
        eoc_rec0.add_data_point(1.0 / nel_1d, err_max)

        # set a non-zero, but uniform velocity component
        for i in range(len(mom_input)):
            mom_input[i] = discr.zeros(actx) + (-1.0) ** i

        cv = make_conserved(
            dim, mass=mass_input, energy=energy_input, momentum=mom_input,
            species_mass=species_mass_input)

        state = make_fluid_state(gas_model=gas_model, cv=cv)
        boundaries = {BTAG_ALL: DummyBoundary()}
        ns_rhs = ns_operator(discr, gas_model=gas_model, boundaries=boundaries,
                             state=state, time=0.0)

        rhs_resid = ns_rhs - expected_rhs

        rho_resid = rhs_resid.mass
        rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum
        rhoy_resid = rhs_resid.species_mass

        assert actx.to_numpy(discr.norm(rho_resid, np.inf)) < tolerance
        assert actx.to_numpy(discr.norm(rhoe_resid, np.inf)) < tolerance

        for i in range(dim):
            assert actx.to_numpy(discr.norm(mom_resid[i], np.inf)) < tolerance
        for i in range(nspecies):
            assert actx.to_numpy(discr.norm(rhoy_resid[i], np.inf)) < tolerance

        err_max = actx.to_numpy(discr.norm(rho_resid, np.inf))
        eoc_rec1.add_data_point(1.0 / nel_1d, err_max)

    logger.info(
        f"V == 0 Errors:\n{eoc_rec0}"
        f"V != 0 Errors:\n{eoc_rec1}"
    )

    assert (
        eoc_rec0.order_estimate() >= order - 0.5
        or eoc_rec0.max_error() < 1e-9
    )
    assert (
        eoc_rec1.order_estimate() >= order - 0.5
        or eoc_rec1.max_error() < 1e-9
    )


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None):
    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t)


@pytest.mark.parametrize("order", [2, 3])
def test_poiseuille_rhs(actx_factory, order):
    """Test the Navier-Stokes operator using a Poiseuille state.

    This state should yield rhs = 0 to FP.  The test is performed for 1, 2,
    and 3 dimensions, with orders 1, 2, and 3, with and without passive species.
    """
    actx = actx_factory()
    dim = 2
    tolerance = 1e-9

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    base_pressure = 100000.0
    pressure_ratio = 1.001
    mu = 1.0
    left_boundary_location = 0
    right_boundary_location = 0.1
    ybottom = 0.
    ytop = .02
    nspecies = 0
    mu = 1.0
    kappa = 0.0
    spec_diffusivity = 0 * np.ones(nspecies)
    from mirgecom.gas_model import GasModel, make_fluid_state
    gas_model = GasModel(
        eos=IdealSingleGas(),
        transport=SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                  species_diffusivity=spec_diffusivity))

    def poiseuille_2d(x_vec, eos, cv=None, **kwargs):
        y = x_vec[1]
        x = x_vec[0]
        x0 = left_boundary_location
        xmax = right_boundary_location
        xlen = xmax - x0
        p_low = base_pressure
        p_hi = pressure_ratio*base_pressure
        dp = p_hi - p_low
        dpdx = dp/xlen
        h = ytop - ybottom
        u_x = dpdx*y*(h - y)/(2*mu)
        p_x = p_hi - dpdx*x
        rho = 1.0
        mass = 0*x + rho
        u_y = 0*x
        velocity = make_obj_array([u_x, u_y])
        ke = .5*np.dot(velocity, velocity)*mass
        gamma = eos.gamma()
        if cv is not None:
            mass = cv.mass
            vel = cv.velocity
            ke = .5*np.dot(vel, vel)*mass

        rho_e = p_x/(gamma-1) + ke
        return make_conserved(2, mass=mass, energy=rho_e,
                              momentum=mass*velocity)

    initializer = poiseuille_2d

    # for nel_1d in [4, 8, 12]:
    for nfac in [1, 2, 4, 8]:

        npts_axis = nfac*(12, 20)
        box_ll = (left_boundary_location, ybottom)
        box_ur = (right_boundary_location, ytop)
        mesh = _get_box_mesh(2, a=box_ll, b=box_ur, n=npts_axis)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
        from meshmode.discretization.poly_element import \
            default_simplex_group_factory, QuadratureSimplexGroupFactory

        discr = EagerDGDiscretization(
            actx, mesh,
            discr_tag_to_group_factory={
                DISCR_TAG_BASE: default_simplex_group_factory(
                    base_dim=mesh.dim, order=order),
                DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
            }
        )
        nodes = thaw(discr.nodes(), actx)

        cv_input = initializer(x_vec=nodes, eos=gas_model.eos)
        num_eqns = dim + 2
        expected_rhs = make_conserved(
            dim, q=make_obj_array([discr.zeros(actx)
                                   for i in range(num_eqns)])
        )

        def boundary_func(discr, btag, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            return make_fluid_state(initializer(x_vec=nodes, eos=gas_model.eos,
                                                **kwargs), gas_model)

        boundaries = {
            DTAG_BOUNDARY("-1"):
            PrescribedFluidBoundary(boundary_state_func=boundary_func),
            DTAG_BOUNDARY("+1"):
            PrescribedFluidBoundary(boundary_state_func=boundary_func),
            DTAG_BOUNDARY("-2"): AdiabaticNoslipMovingBoundary(),
            DTAG_BOUNDARY("+2"): AdiabaticNoslipMovingBoundary()}

        state = make_fluid_state(gas_model=gas_model, cv=cv_input)
        ns_rhs = ns_operator(discr, gas_model=gas_model, boundaries=boundaries,
                             state=state, time=0.0)

        rhs_resid = ns_rhs - expected_rhs
        rho_resid = rhs_resid.mass
        # rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum

        rho_rhs = ns_rhs.mass
        # rhoe_rhs = ns_rhs.energy
        rhov_rhs = ns_rhs.momentum
        # rhoy_rhs = ns_rhs.species_mass

        print(
            f"rho_rhs  = {rho_rhs}\n"
            # f"rhoe_rhs = {rhoe_rhs}\n"
            f"rhov_rhs = {rhov_rhs}\n"
            # f"rhoy_rhs = {rhoy_rhs}\n"
        )

        tol_fudge = 2e-4
        assert actx.to_numpy(discr.norm(rho_resid, np.inf)) < tolerance
        # assert actx.to_numpy(discr.norm(rhoe_resid, np.inf)) < tolerance
        mom_err = [actx.to_numpy(discr.norm(mom_resid[i], np.inf))
                   for i in range(dim)]
        err_max = max(mom_err)
        for i in range(dim):
            assert mom_err[i] < tol_fudge

        # err_max = actx.to_numpy(discr.norm(rho_resid, np.inf)
        eoc_rec.add_data_point(1.0 / nfac, err_max)

    logger.info(
        f"V != 0 Errors:\n{eoc_rec}"
    )

    if order <= 1:
        assert eoc_rec.order_estimate() >= order - 0.5
    else:
        # Poiseuille is a quadratic profile, exactly represented by
        # quadratic and higher
        assert eoc_rec.max_error() < tol_fudge
