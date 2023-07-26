"""Test boundary condition and bc-related functions."""

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
import numpy.linalg as la  # noqa
import logging
import pytest
from pytools.obj_array import make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.discretization.connection import FACE_RESTR_ALL
from mirgecom.initializers import Lump
from mirgecom.eos import IdealSingleGas
from grudge.trace_pair import interior_trace_pair, interior_trace_pairs
from grudge.trace_pair import TracePair
from grudge.dof_desc import as_dofdesc
from mirgecom.fluid import make_conserved
from mirgecom.discretization import create_discretization_collection
from mirgecom.inviscid import (
    inviscid_facial_flux_rusanov,
    inviscid_facial_flux_hll
)
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    project_fluid_state,
    make_fluid_state_trace_pairs
)
import grudge.op as op
from mirgecom.simutil import get_box_mesh

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


# def _test_mengaldo_bc_interface(dcoll, dd_bdry, gas_model,
#                                state_minus, bc):
#    state_plus = bc.state_plus(dcoll, dd_bdry=dd_bry, gas_model=gas_model,
#                               state_minus=state_minus)
#    state_bc = bc.state_bc()
#    t_bc = bc.temperature_bc()
#    grad_cv_bc = bc.grad_cv_bc()
#    grad_t_bc = bc.grad_temperature_bc()
#    f_i = bc.inviscid_divergence_flux()
#    f_v = bc.viscous_divergence_flux()
#    f_dt = bc.temperature_gradient_flux()
#    f_dcv = bc.cv_gradient_flux()

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_normal_axes_utility(actx_factory, dim):
    """Check that we can reliably get an orthonormal set given a normal."""
    actx = actx_factory()

    from mirgecom.boundary import _get_normal_axes as gna
    order = 1
    nels_geom = 5
    a = -.01
    b = .01
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    normal_vectors = nodes / actx.np.sqrt(np.dot(nodes, nodes))
    normal_set = gna(normal_vectors)
    nset = len(normal_set)
    assert nset == dim

    def vec_norm(vec, p=2):
        return actx.to_numpy(op.norm(dcoll, vec, p=p)) # noqa

    # make sure the vecs all have mag=1
    for i in range(nset):
        assert vec_norm(normal_set[i]).all() == 1

    tol = 1e-12
    if dim > 1:
        for i in range(dim):
            for j in range(dim-1):
                next_index = (i + j + 1) % dim
                print(f"(i,j) = ({i}, {next_index})")
                norm_comp = np.dot(normal_set[i], normal_set[next_index])
                assert vec_norm(norm_comp, np.inf) < tol

        if dim == 2:
            rmat_exp = make_obj_array([normal_set[0][0], normal_set[0][1],
                                       normal_set[1][0], normal_set[1][1]])
        else:
            rmat_exp = make_obj_array(
                [normal_set[0][0], normal_set[0][1], normal_set[0][2],
                 normal_set[1][0], normal_set[1][1], normal_set[1][2],
                 normal_set[2][0], normal_set[2][1], normal_set[2][2]])
        rmat_exp = rmat_exp.reshape(dim, dim)
        print(f"{rmat_exp=}")

        from mirgecom.boundary import _get_rotation_matrix as grm
        rotation_matrix = grm(normal_vectors)
        print(f"{rotation_matrix.shape}")
        print(f"{rotation_matrix=}")
        print(f"{rmat_exp - rotation_matrix=}")

        resid = rmat_exp - rotation_matrix
        assert vec_norm(resid).all() == 0


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
def test_farfield_boundary(actx_factory, dim, flux_func):
    """Check FarfieldBoundary boundary treatment."""
    actx = actx_factory()
    order = 1
    from mirgecom.boundary import FarfieldBoundary

    gamma = 1.4
    kappa = 3.0
    sigma = 5.0
    ff_temp = 2.0
    ff_press = .5
    ff_vel = np.zeros(dim) + 1.0
    ff_dens = ff_press / ff_temp
    ff_ke = .5*ff_dens*np.dot(ff_vel, ff_vel)
    ff_energy = ff_press/(gamma-1) + ff_ke

    from mirgecom.initializers import Uniform
    ff_init = Uniform(dim=dim, rho=ff_dens, p=ff_press,
                      velocity=ff_vel)
    ff_momentum = ff_dens*ff_vel

    from mirgecom.transport import SimpleTransport

    gas_model = GasModel(eos=IdealSingleGas(gas_const=1.0),
                         transport=SimpleTransport(viscosity=sigma,
                                                   thermal_conductivity=kappa))
    bndry = FarfieldBoundary(numdim=dim,
                             free_stream_velocity=ff_vel,
                             free_stream_pressure=ff_press,
                             free_stream_temperature=ff_temp)

    nels_geom = 16
    a = -1.0
    b = 1.0
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    nhat = actx.thaw(dcoll.normal(BTAG_ALL))
    print(f"{nhat=}")

    ff_cv = ff_init(nodes, eos=gas_model.eos)
    exp_ff_cv = op.project(dcoll, "vol", BTAG_ALL, ff_cv)

    from mirgecom.flux import num_flux_central

    def gradient_flux_interior(int_tpair):
        from arraycontext import outer
        normal = actx.thaw(dcoll.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        return op.project(dcoll, int_tpair.dd, "all_faces", flux_weak)

    # utility to compare stuff on the boundary only
    # from functools import partial
    # bnd_norm = partial(op.norm, dcoll, p=np.inf, dd=BTAG_ALL)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocities in each direction
    # err_max = 0.0
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))

        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity
            initializer = Uniform(dim=dim, velocity=vel)
            uniform_cv = initializer(nodes, eos=gas_model.eos)
            uniform_state = make_fluid_state(cv=uniform_cv, gas_model=gas_model)
            state_minus = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                              uniform_state, gas_model)

            print(f"Volume state: {uniform_state=}")
            temper = uniform_state.temperature
            print(f"Volume state: {temper=}")

            cv_interior_pairs = interior_trace_pairs(dcoll, uniform_state.cv)
            cv_int_tpair = cv_interior_pairs[0]

            state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs, gas_model)
            state_pair = state_pairs[0]

            cv_flux_int = gradient_flux_interior(cv_int_tpair)
            print(f"{cv_flux_int=}")

            ff_bndry_state = bndry.farfield_state(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)
            print(f"{ff_bndry_state=}")
            ff_bndry_temperature = ff_bndry_state.temperature
            print(f"{ff_bndry_temperature=}")

            cv_grad_flux_bndry = bndry.cv_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                        gas_model=gas_model,
                                                        state_minus=state_minus)

            cv_grad_flux_allfaces = \
                op.project(dcoll, as_dofdesc(BTAG_ALL),
                           as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                           cv_grad_flux_bndry)

            print(f"{cv_grad_flux_bndry=}")

            cv_flux_bnd = cv_grad_flux_allfaces + cv_flux_int

            temperature_bc = bndry.temperature_bc(
                dcoll, dd_bdry=BTAG_ALL, state_minus=state_minus)
            print(f"{temperature_bc=}")

            t_int_tpair = interior_trace_pair(dcoll, temper)
            t_flux_int = gradient_flux_interior(t_int_tpair)
            t_flux_bc = bndry.temperature_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                        gas_model=gas_model,
                                                        state_minus=state_minus)

            t_flux_bc = op.project(dcoll, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    t_flux_bc)

            t_flux_bnd = t_flux_bc + t_flux_int

            i_flux_bc = bndry.inviscid_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus)

            nhat = actx.thaw(dcoll.normal(state_pair.dd))
            bnd_flux = flux_func(state_pair, gas_model, nhat)
            dd = state_pair.dd
            dd_allfaces = dd.with_dtag("all_faces")
            i_flux_int = op.project(dcoll, dd, dd_allfaces, bnd_flux)
            bc_dd = as_dofdesc(BTAG_ALL)
            i_flux_bc = op.project(dcoll, bc_dd, dd_allfaces, i_flux_bc)

            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{cv_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import grad_operator
            dd_vol = as_dofdesc("vol")
            dd_faces = as_dofdesc("all_faces")
            grad_cv_minus = \
                op.project(dcoll, "vol", BTAG_ALL,
                              grad_operator(dcoll, dd_vol, dd_faces,
                                            uniform_state.cv, cv_flux_bnd))
            grad_t_minus = op.project(dcoll, "vol", BTAG_ALL,
                                         grad_operator(dcoll, dd_vol, dd_faces,
                                                       temper, t_flux_bnd))

            print(f"{grad_cv_minus=}")
            print(f"{grad_t_minus=}")

            v_flux_bc = bndry.viscous_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                      gas_model=gas_model,
                                                      state_minus=state_minus,
                                                      grad_cv_minus=grad_cv_minus,
                                                      grad_t_minus=grad_t_minus)
            print(f"{v_flux_bc=}")

            assert actx.np.equal(ff_bndry_state.cv, exp_ff_cv)
            assert actx.np.all(temperature_bc == ff_temp)
            for idim in range(dim):
                assert actx.np.all(ff_bndry_state.momentum_density[idim]
                                   == ff_momentum[idim])
            assert actx.np.all(ff_bndry_state.energy_density == ff_energy)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
def test_outflow_boundary(actx_factory, dim, flux_func):
    """Check PressureOutflowBoundary boundary treatment."""
    actx = actx_factory()
    order = 1
    from mirgecom.boundary import PressureOutflowBoundary

    kappa = 3.0
    sigma = 5.0

    gas_const = 1.0
    flowbnd_press = 1.0/1.01
    flowbnd_press_bc = 2.*flowbnd_press - 1.

    c = np.sqrt(1.4)
    print(f"{flowbnd_press=}")
    print(f"{flowbnd_press_bc=}")
    print(f"{c=}")

    eos = IdealSingleGas(gas_const=gas_const)

    from mirgecom.transport import SimpleTransport
    from mirgecom.initializers import Uniform

    gas_model = GasModel(eos=eos,
                         transport=SimpleTransport(viscosity=sigma,
                                                   thermal_conductivity=kappa))
    bndry = PressureOutflowBoundary(boundary_pressure=flowbnd_press)

    nels_geom = 16
    a = 1.0
    b = 2.0
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    nhat = actx.thaw(dcoll.normal(BTAG_ALL))
    print(f"{nhat=}")

    # utility to compare stuff on the boundary only
    # from functools import partial
    # bnd_norm = partial(op.norm, dcoll, p=np.inf, dd=BTAG_ALL)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocities in each direction
    # err_max = 0.0
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))

        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity

            initializer = Uniform(dim=dim, velocity=vel)
            uniform_cv = initializer(nodes, eos=gas_model.eos)
            uniform_state = make_fluid_state(cv=uniform_cv, gas_model=gas_model)
            state_minus = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                              uniform_state, gas_model)

            ones = 0*state_minus.mass_density + 1.0
            print(f"Volume state: {uniform_state=}")
            temper = uniform_state.temperature
            speed_of_sound = uniform_state.speed_of_sound
            print(f"Volume state: {temper=}")
            print(f"Volume state: {speed_of_sound=}")

            flowbnd_bndry_state = bndry.outflow_state(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)
            print(f"{flowbnd_bndry_state=}")
            flowbnd_bndry_temperature = flowbnd_bndry_state.temperature
            flowbnd_bndry_pressure = flowbnd_bndry_state.pressure
            print(f"{flowbnd_bndry_temperature=}")
            print(f"{flowbnd_bndry_pressure=}")

            nhat = actx.thaw(dcoll.normal(BTAG_ALL))

            bnd_dens = 1.0*ones
            bnd_mom = bnd_dens*vel
            bnd_ener = flowbnd_press_bc/.4 + .5*np.dot(bnd_mom, bnd_mom)
            exp_flowbnd_cv = make_conserved(dim=dim, mass=bnd_dens,
                                              momentum=bnd_mom, energy=bnd_ener)

            print(f"{exp_flowbnd_cv=}")

            assert actx.np.equal(flowbnd_bndry_state.cv, exp_flowbnd_cv)
            assert actx.np.all(flowbnd_bndry_temperature == flowbnd_press_bc)
            assert actx.np.all(flowbnd_bndry_pressure == flowbnd_press_bc)

            vel = 1.5*vel

            initializer = Uniform(dim=dim, velocity=vel)

            uniform_cv = initializer(nodes, eos=gas_model.eos)
            uniform_state = make_fluid_state(cv=uniform_cv, gas_model=gas_model)
            state_minus = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                              uniform_state, gas_model)

            print(f"Volume state: {uniform_state=}")
            temper = uniform_state.temperature
            speed_of_sound = uniform_state.speed_of_sound
            print(f"Volume state: {temper=}")
            print(f"Volume state: {speed_of_sound=}")

            flowbnd_bndry_state = bndry.outflow_state(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)
            print(f"{flowbnd_bndry_state=}")
            flowbnd_bndry_temperature = flowbnd_bndry_state.temperature
            print(f"{flowbnd_bndry_temperature=}")

            nhat = actx.thaw(dcoll.normal(BTAG_ALL))

            bnd_dens = 1.0*ones
            bnd_mom = bnd_dens*vel
            bnd_ener = flowbnd_press_bc/.4 + .5*np.dot(bnd_mom, bnd_mom)
            exp_flowbnd_cv = make_conserved(dim=dim, mass=bnd_dens,
                                              momentum=bnd_mom, energy=bnd_ener)

            assert actx.np.equal(flowbnd_bndry_state.cv, exp_flowbnd_cv)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
def test_isothermal_wall_boundary(actx_factory, dim, flux_func):
    """Check IsothermalWallBoundary boundary treatment."""
    actx = actx_factory()
    order = 1

    wall_temp = 2.0
    kappa = 3.0
    sigma = 5.0
    exp_temp_bc_val = wall_temp

    from mirgecom.transport import SimpleTransport
    from mirgecom.boundary import IsothermalWallBoundary

    gas_model = GasModel(eos=IdealSingleGas(gas_const=1.0),
                         transport=SimpleTransport(viscosity=sigma,
                                                   thermal_conductivity=kappa))

    wall = IsothermalWallBoundary(wall_temperature=wall_temp)

    nels_geom = 16
    a = 1.0
    b = 2.0
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    nhat = actx.thaw(dcoll.normal(BTAG_ALL))
    print(f"{nhat=}")

    from mirgecom.flux import num_flux_central

    def gradient_flux_interior(int_tpair):
        from arraycontext import outer
        normal = actx.thaw(dcoll.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        return op.project(dcoll, int_tpair.dd, "all_faces", flux_weak)

    # utility to compare stuff on the boundary only
    # from functools import partial
    # bnd_norm = partial(op.norm, dcoll, p=np.inf, dd=BTAG_ALL)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocities in each direction
    # err_max = 0.0
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))

        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity
            from mirgecom.initializers import Uniform
            initializer = Uniform(dim=dim, velocity=vel)
            uniform_cv = initializer(nodes, eos=gas_model.eos)
            uniform_state = make_fluid_state(cv=uniform_cv, gas_model=gas_model)
            state_minus = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                              uniform_state, gas_model)
            ones = state_minus.mass_density*0 + 1.0
            print(f"{uniform_state=}")
            temper = uniform_state.temperature
            print(f"{temper=}")

            expected_noslip_cv = 1.0*state_minus.cv
            expected_noslip_cv = expected_noslip_cv.replace(
                momentum=0*expected_noslip_cv.momentum)

            expected_noslip_momentum = 0*vel*state_minus.mass_density
            expected_wall_temperature = \
                exp_temp_bc_val * ones

            print(f"{expected_wall_temperature=}")
            print(f"{expected_noslip_cv=}")

            cv_interior_pairs = interior_trace_pairs(dcoll, uniform_state.cv)
            cv_int_tpair = cv_interior_pairs[0]

            state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs, gas_model)
            state_pair = state_pairs[0]

            cv_flux_int = gradient_flux_interior(cv_int_tpair)
            print(f"{cv_flux_int=}")

            wall_state = wall.state_bc(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)
            print(f"{wall_state=}")
            wall_temperature = wall_state.temperature
            print(f"{wall_temperature=}")

            cv_grad_flux_wall = wall.cv_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                 gas_model=gas_model,
                                                 state_minus=state_minus)

            cv_grad_flux_allfaces = \
                op.project(dcoll, as_dofdesc(BTAG_ALL),
                           as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                           cv_grad_flux_wall)

            print(f"{cv_grad_flux_wall=}")

            cv_flux_bnd = cv_grad_flux_allfaces + cv_flux_int

            temperature_bc = wall.temperature_bc(
                dcoll, dd_bdry=BTAG_ALL, state_minus=state_minus)
            print(f"{temperature_bc=}")

            t_int_tpair = interior_trace_pair(dcoll, temper)
            t_flux_int = gradient_flux_interior(t_int_tpair)
            t_flux_bc = wall.temperature_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus)

            t_flux_bc = op.project(dcoll, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    t_flux_bc)

            t_flux_bnd = t_flux_bc + t_flux_int

            i_flux_bc = wall.inviscid_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                      gas_model=gas_model,
                                                      state_minus=state_minus)

            nhat = actx.thaw(dcoll.normal(state_pair.dd))
            bnd_flux = flux_func(state_pair, gas_model, nhat)
            dd = state_pair.dd
            dd_allfaces = dd.with_dtag("all_faces")
            i_flux_int = op.project(dcoll, dd, dd_allfaces, bnd_flux)
            bc_dd = as_dofdesc(BTAG_ALL)
            i_flux_bc = op.project(dcoll, bc_dd, dd_allfaces, i_flux_bc)

            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{cv_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import grad_operator
            dd_vol = as_dofdesc("vol")
            dd_faces = as_dofdesc("all_faces")
            grad_cv_minus = \
                op.project(dcoll, "vol", BTAG_ALL,
                              grad_operator(dcoll, dd_vol, dd_faces,
                                            uniform_state.cv, cv_flux_bnd))
            grad_t_minus = op.project(dcoll, "vol", BTAG_ALL,
                                         grad_operator(dcoll, dd_vol, dd_faces,
                                                       temper, t_flux_bnd))

            print(f"{grad_cv_minus=}")
            print(f"{grad_t_minus=}")

            v_flux_bc = wall.viscous_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                     gas_model=gas_model,
                                                     state_minus=state_minus,
                                                     grad_cv_minus=grad_cv_minus,
                                                     grad_t_minus=grad_t_minus)
            print(f"{v_flux_bc=}")

            assert actx.np.equal(wall_state.cv, expected_noslip_cv)
            assert actx.np.all(temperature_bc == expected_wall_temperature)
            for idim in range(dim):
                assert actx.np.all(wall_state.momentum_density[idim]
                                   == expected_noslip_momentum[idim])


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
def test_adiabatic_noslip_wall_boundary(actx_factory, dim, flux_func):
    """Check AdiabaticNoslipWallBoundary boundary treatment."""
    actx = actx_factory()
    order = 1

    kappa = 3.0
    sigma = 5.0

    from mirgecom.transport import SimpleTransport
    from mirgecom.boundary import AdiabaticNoslipWallBoundary

    gas_model = GasModel(eos=IdealSingleGas(gas_const=4.0),
                         transport=SimpleTransport(viscosity=sigma,
                                                   thermal_conductivity=kappa))
    exp_temp = 1.0/4.0

    wall = AdiabaticNoslipWallBoundary()

    nels_geom = 16
    a = 1.0
    b = 2.0
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    nhat = actx.thaw(dcoll.normal(BTAG_ALL))
    print(f"{nhat=}")

    from mirgecom.flux import num_flux_central

    def gradient_flux_interior(int_tpair):
        from arraycontext import outer
        normal = actx.thaw(dcoll.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        return op.project(dcoll, int_tpair.dd, "all_faces", flux_weak)

    # utility to compare stuff on the boundary only
    # from functools import partial
    # bnd_norm = partial(op.norm, dcoll, p=np.inf, dd=BTAG_ALL)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocities in each direction
    # err_max = 0.0
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))

        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity
            from mirgecom.initializers import Uniform
            initializer = Uniform(dim=dim, velocity=vel)
            uniform_cv = initializer(nodes, eos=gas_model.eos)
            uniform_state = make_fluid_state(cv=uniform_cv, gas_model=gas_model)
            state_minus = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                              uniform_state, gas_model)
            print(f"{uniform_state=}")
            temper = uniform_state.temperature
            print(f"{temper=}")

            expected_adv_wall_cv = 1.0*state_minus.cv
            expected_adv_wall_cv = expected_adv_wall_cv.replace(
                momentum=-expected_adv_wall_cv.momentum)
            expected_diff_wall_cv = 1.0*state_minus.cv
            expected_diff_wall_cv = expected_diff_wall_cv.replace(
                momentum=0*expected_diff_wall_cv.momentum)

            expected_adv_momentum = -state_minus.momentum_density
            expected_diff_momentum = 0*state_minus.momentum_density

            print(f"{expected_adv_wall_cv=}")
            print(f"{expected_diff_wall_cv=}")

            expected_wall_temperature = exp_temp + 0*state_minus.temperature

            print(f"{expected_wall_temperature=}")
            cv_interior_pairs = interior_trace_pairs(dcoll, uniform_state.cv)
            cv_int_tpair = cv_interior_pairs[0]

            state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs, gas_model)
            state_pair = state_pairs[0]

            cv_flux_int = gradient_flux_interior(cv_int_tpair)
            print(f"{cv_flux_int=}")

            adv_wall_state = wall.state_plus(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)
            diff_wall_state = wall.state_bc(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)

            print(f"{adv_wall_state=}")
            print(f"{diff_wall_state=}")

            wall_temperature = adv_wall_state.temperature
            print(f"{wall_temperature=}")

            cv_grad_flux_wall = wall.cv_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                 gas_model=gas_model,
                                                 state_minus=state_minus)

            cv_grad_flux_allfaces = \
                op.project(dcoll, as_dofdesc(BTAG_ALL),
                           as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                           cv_grad_flux_wall)

            print(f"{cv_grad_flux_wall=}")

            temperature_bc = wall.temperature_bc(
                dcoll, dd_bdry=BTAG_ALL, state_minus=state_minus)
            print(f"{temperature_bc=}")

            cv_flux_bnd = cv_grad_flux_allfaces + cv_flux_int

            t_int_tpair = interior_trace_pair(dcoll, temper)
            t_flux_int = gradient_flux_interior(t_int_tpair)
            t_flux_bc = wall.temperature_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus)

            t_flux_bc = op.project(dcoll, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    t_flux_bc)

            t_flux_bnd = t_flux_bc + t_flux_int

            i_flux_bc = wall.inviscid_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                      gas_model=gas_model,
                                                      state_minus=state_minus)

            nhat = actx.thaw(dcoll.normal(state_pair.dd))
            bnd_flux = flux_func(state_pair, gas_model, nhat)
            dd = state_pair.dd
            dd_allfaces = dd.with_dtag("all_faces")
            i_flux_int = op.project(dcoll, dd, dd_allfaces, bnd_flux)
            bc_dd = as_dofdesc(BTAG_ALL)
            i_flux_bc = op.project(dcoll, bc_dd, dd_allfaces, i_flux_bc)

            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{cv_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import grad_operator
            dd_vol = as_dofdesc("vol")
            dd_faces = as_dofdesc("all_faces")
            grad_cv_minus = \
                op.project(dcoll, "vol", BTAG_ALL,
                              grad_operator(dcoll, dd_vol, dd_faces,
                                            uniform_state.cv, cv_flux_bnd))
            grad_t_minus = op.project(dcoll, "vol", BTAG_ALL,
                                         grad_operator(dcoll, dd_vol, dd_faces,
                                                       temper, t_flux_bnd))

            print(f"{grad_cv_minus=}")
            print(f"{grad_t_minus=}")

            v_flux_bc = wall.viscous_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                     gas_model=gas_model,
                                                     state_minus=state_minus,
                                                     grad_cv_minus=grad_cv_minus,
                                                     grad_t_minus=grad_t_minus)
            print(f"{v_flux_bc=}")

            assert actx.np.equal(adv_wall_state.cv, expected_adv_wall_cv)
            assert actx.np.equal(diff_wall_state.cv, expected_diff_wall_cv)
            assert actx.np.all(temperature_bc == expected_wall_temperature)
            for idim in range(dim):
                assert actx.np.all(adv_wall_state.momentum_density[idim]
                                   == expected_adv_momentum[idim])

                assert actx.np.all(diff_wall_state.momentum_density[idim]
                                   == expected_diff_momentum[idim])


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
def test_symmetry_wall_boundary(actx_factory, dim, flux_func):
    """Check AdiabaticSlipBoundary for symmetry treatment."""
    actx = actx_factory()
    order = 1

    kappa = 3.0
    sigma = 5.0

    from mirgecom.transport import SimpleTransport
    from mirgecom.boundary import AdiabaticSlipBoundary

    gas_const = 4.0
    gas_model = GasModel(eos=IdealSingleGas(gas_const=gas_const),
                         transport=SimpleTransport(viscosity=sigma,
                                                   thermal_conductivity=kappa))
    exp_temp = 1.0/gas_const

    wall = AdiabaticSlipBoundary()

    nels_geom = 16
    a = 1.0
    b = 2.0
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    nhat = actx.thaw(dcoll.normal(BTAG_ALL))
    print(f"{nhat=}")

    from mirgecom.flux import num_flux_central

    def gradient_flux_interior(int_tpair):
        from arraycontext import outer
        normal = actx.thaw(dcoll.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        return op.project(dcoll, int_tpair.dd, "all_faces", flux_weak)

    # utility to compare stuff on the boundary only
    # from functools import partial
    # bnd_norm = partial(op.norm, dcoll, p=np.inf, dd=BTAG_ALL)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocities in each direction
    # err_max = 0.0
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))

        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity
            from mirgecom.initializers import Uniform
            initializer = Uniform(dim=dim, velocity=vel)
            uniform_cv = initializer(nodes, eos=gas_model.eos)
            uniform_state = make_fluid_state(cv=uniform_cv, gas_model=gas_model)
            state_minus = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                              uniform_state, gas_model)

            print(f"{state_minus.temperature=}")
            bnd_normal = actx.thaw(dcoll.normal(BTAG_ALL))
            print(f"{bnd_normal=}")
            expected_temp_boundary = exp_temp + 0*state_minus.temperature

            bnd_velocity = vel + 0*bnd_normal
            print(f"{bnd_velocity=}")

            bnd_speed = np.dot(bnd_velocity, bnd_normal)
            print(f"{bnd_speed=}")

            exp_diff_vel = vel - bnd_speed*bnd_normal
            exp_adv_vel = exp_diff_vel - bnd_speed*bnd_normal
            print(f"{exp_diff_vel=}")
            print(f"{exp_adv_vel=}")

            print(f"{uniform_state=}")
            temper = uniform_state.temperature
            print(f"{temper=}")

            expected_adv_momentum = state_minus.cv.mass*exp_adv_vel
            expected_diff_momentum = state_minus.cv.mass*exp_diff_vel

            expected_adv_wall_cv = 1.0*state_minus.cv
            expected_adv_wall_cv = expected_adv_wall_cv.replace(
                momentum=expected_adv_momentum)
            expected_diff_wall_cv = 1.0*state_minus.cv
            expected_diff_wall_cv = expected_diff_wall_cv.replace(
                momentum=expected_diff_momentum)

            print(f"{expected_adv_wall_cv=}")
            print(f"{expected_diff_wall_cv=}")

            cv_interior_pairs = interior_trace_pairs(dcoll, uniform_state.cv)
            cv_int_tpair = cv_interior_pairs[0]

            state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs, gas_model)
            state_pair = state_pairs[0]

            cv_flux_int = gradient_flux_interior(cv_int_tpair)
            print(f"{cv_flux_int=}")

            adv_wall_state = wall.state_plus(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)
            diff_wall_state = wall.state_bc(
                dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                state_minus=state_minus)

            print(f"{adv_wall_state=}")
            print(f"{diff_wall_state=}")

            wall_temperature = adv_wall_state.temperature
            print(f"{wall_temperature=}")

            cv_grad_flux_wall = wall.cv_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                 gas_model=gas_model,
                                                 state_minus=state_minus)

            cv_grad_flux_allfaces = \
                op.project(dcoll, as_dofdesc(BTAG_ALL),
                           as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                           cv_grad_flux_wall)

            print(f"{cv_grad_flux_wall=}")

            cv_flux_bnd = cv_grad_flux_allfaces + cv_flux_int

            t_int_tpair = interior_trace_pair(dcoll, temper)
            t_flux_int = gradient_flux_interior(t_int_tpair)
            t_flux_bc = wall.temperature_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus)

            t_flux_bc = op.project(dcoll, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    t_flux_bc)

            t_flux_bnd = t_flux_bc + t_flux_int

            i_flux_bc = wall.inviscid_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                      gas_model=gas_model,
                                                      state_minus=state_minus)

            nhat = actx.thaw(dcoll.normal(state_pair.dd))
            bnd_flux = flux_func(state_pair, gas_model, nhat)
            dd = state_pair.dd
            dd_allfaces = dd.with_dtag("all_faces")
            i_flux_int = op.project(dcoll, dd, dd_allfaces, bnd_flux)
            bc_dd = as_dofdesc(BTAG_ALL)
            i_flux_bc = op.project(dcoll, bc_dd, dd_allfaces, i_flux_bc)

            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{cv_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import grad_operator
            dd_vol = as_dofdesc("vol")
            dd_faces = as_dofdesc("all_faces")
            grad_cv_minus = \
                op.project(dcoll, "vol", BTAG_ALL,
                              grad_operator(dcoll, dd_vol, dd_faces,
                                            uniform_state.cv, cv_flux_bnd))
            grad_t_minus = op.project(dcoll, "vol", BTAG_ALL,
                                         grad_operator(dcoll, dd_vol, dd_faces,
                                                       temper, t_flux_bnd))

            print(f"{grad_cv_minus=}")
            print(f"{grad_t_minus=}")

            v_flux_bc = wall.viscous_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                     gas_model=gas_model,
                                                     state_minus=state_minus,
                                                     grad_cv_minus=grad_cv_minus,
                                                     grad_t_minus=grad_t_minus)
            print(f"{v_flux_bc=}")
            temperature_bc = wall.temperature_bc(
                dcoll, dd_bdry=BTAG_ALL, state_minus=state_minus)

            assert actx.np.equal(adv_wall_state.cv, expected_adv_wall_cv)
            assert actx.np.equal(diff_wall_state.cv, expected_diff_wall_cv)
            assert actx.np.all(temperature_bc == expected_temp_boundary)

            for idim in range(dim):
                assert actx.np.all(adv_wall_state.momentum_density[idim]
                                   == expected_adv_momentum[idim])

                assert actx.np.all(diff_wall_state.momentum_density[idim]
                                   == expected_diff_momentum[idim])


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_slipwall_identity(actx_factory, dim):
    """Identity test - check for the expected boundary solution.

    Checks that the slipwall implements the expected boundary solution:
    rho_plus = rho_minus
    v_plus = v_minus - 2 * (n_hat . v_minus) * n_hat
    mom_plus = rho_plus * v_plus
    E_plus = E_minus
    """
    actx = actx_factory()

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    dcoll = create_discretization_collection(actx, mesh, order)
    nodes = actx.thaw(dcoll.nodes())
    eos = IdealSingleGas()
    orig = np.zeros(shape=(dim,))
    nhat = actx.thaw(dcoll.normal(BTAG_ALL))
    gas_model = GasModel(eos=eos)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    from mirgecom.boundary import AdiabaticSlipBoundary
    # for velocity going along each direction
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))
        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity  # Check incoming normal
            initializer = Lump(dim=dim, center=orig, velocity=vel, rhoamp=0.0)
            wall = AdiabaticSlipBoundary()

            uniform_state = initializer(nodes)
            cv_minus = op.project(dcoll, "vol", BTAG_ALL, uniform_state)
            state_minus = make_fluid_state(cv=cv_minus, gas_model=gas_model)

            def bnd_norm(vec):
                return actx.to_numpy(op.norm(dcoll, vec, p=np.inf, dd=BTAG_ALL))

            state_plus = \
                wall.state_plus(
                    dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                    state_minus=state_minus)

            bnd_pair = TracePair(
                as_dofdesc(BTAG_ALL),
                interior=state_minus.cv,
                exterior=state_plus.cv)

            # check that mass and energy are preserved
            mass_resid = bnd_pair.int.mass - bnd_pair.ext.mass
            mass_err = bnd_norm(mass_resid)
            assert mass_err == 0.0

            energy_resid = bnd_pair.int.energy - bnd_pair.ext.energy
            energy_err = bnd_norm(energy_resid)
            assert energy_err == 0.0

            # check that exterior momentum term is mom_interior - 2 * mom_normal
            mom_norm_comp = np.dot(bnd_pair.int.momentum, nhat)
            mom_norm = nhat * mom_norm_comp
            expected_mom_ext = bnd_pair.int.momentum - 2.0 * mom_norm
            mom_resid = bnd_pair.ext.momentum - expected_mom_ext
            mom_err = bnd_norm(mom_resid)

            assert mom_err == 0.0


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
def test_slipwall_flux(actx_factory, dim, order, flux_func):
    """Check for zero boundary flux.

    Check for vanishing flux across the slipwall.
    """
    actx = actx_factory()

    from mirgecom.boundary import AdiabaticSlipBoundary
    wall = AdiabaticSlipBoundary()
    gas_model = GasModel(eos=IdealSingleGas())

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for nel_1d in [4, 8, 12]:
        from meshmode.mesh.generation import generate_regular_rect_mesh

        mesh = generate_regular_rect_mesh(
            a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
        )

        dcoll = create_discretization_collection(actx, mesh, order=order)
        nodes = actx.thaw(dcoll.nodes())
        nhat = actx.thaw(dcoll.normal(BTAG_ALL))
        h = 1.0 / nel_1d

        def bnd_norm(vec):
            return actx.to_numpy(op.norm(dcoll, vec, p=np.inf, dd=BTAG_ALL))  # noqa

        logger.info(f"Number of {dim}d elems: {mesh.nelements}")
        # for velocities in each direction
        err_max = 0.0
        for vdir in range(dim):
            vel = np.zeros(shape=(dim,))

            # for velocity directions +1, and -1
            for parity in [1.0, -1.0]:
                vel[vdir] = parity
                from mirgecom.initializers import Uniform
                initializer = Uniform(dim=dim, velocity=vel)
                uniform_state = initializer(nodes)
                fluid_state = make_fluid_state(uniform_state, gas_model)

                interior_soln = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                                    state=fluid_state,
                                                    gas_model=gas_model)

                bnd_soln = wall.state_plus(dcoll,
                    dd_bdry=BTAG_ALL, gas_model=gas_model, state_minus=interior_soln)

                bnd_pair = TracePair(
                    as_dofdesc(BTAG_ALL),
                    interior=interior_soln.cv,
                    exterior=bnd_soln.cv)
                state_pair = TracePair(
                    as_dofdesc(BTAG_ALL),
                    interior=interior_soln,
                    exterior=bnd_soln)

                # Check the total velocity component normal
                # to each surface.  It should be zero.  The
                # numerical fluxes cannot be zero.
                avg_state = 0.5*(bnd_pair.int + bnd_pair.ext)
                err_max = max(err_max, bnd_norm(np.dot(avg_state.momentum, nhat)))

                normal = actx.thaw(dcoll.normal(BTAG_ALL))
                bnd_flux = flux_func(state_pair, gas_model, normal)

                err_max = max(err_max, bnd_norm(bnd_flux.mass),
                              bnd_norm(bnd_flux.energy))

        eoc.add_data_point(h, err_max)

    message = (f"EOC:\n{eoc}")
    logger.info(message)
    assert (
        eoc.order_estimate() >= order - 0.5
        or eoc.max_error() < 1e-12
    )


class _VortexSoln:

    def __init__(self, **kwargs):
        self._dim = 2
        from mirgecom.initializers import Vortex2D
        origin = np.zeros(2)
        velocity = origin + 1
        self._initializer = Vortex2D(center=origin, velocity=velocity)

    def dim(self):
        return self._dim

    def __call__(self, r, eos, **kwargs):
        return self._initializer(x_vec=r, eos=eos)


class _PoiseuilleSoln:

    def __init__(self, **kwargs):
        self._dim = 2
        from mirgecom.initializers import PlanarPoiseuille
        self._initializer = PlanarPoiseuille()

    def dim(self):
        return self._dim

    def __call__(self, r, eos, **kwargs):
        return self._initializer(x_vec=r, eos=eos)


# This simple type of boundary is used when the user
# wants to prescribe a fixed (or time-dependent) state
# at the boundary (e.g., prescribed exact soln on the
# domain boundary).
# This test makes sure that the boundary type
# works mechanically, and gives the correct boundary solution
# given the prescribed soln.
@pytest.mark.parametrize("prescribed_soln", [_VortexSoln(), _PoiseuilleSoln()])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
#                                             ShearflowSoln,
def test_prescribed(actx_factory, prescribed_soln, flux_func):
    """Check prescribed boundary treatment."""
    actx = actx_factory()
    order = 1
    dim = prescribed_soln.dim()

    kappa = 3.0
    sigma = 5.0

    from mirgecom.transport import SimpleTransport
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa)

    # Functions that control PrescribedViscousBoundary (pvb):
    # Specify none to get a DummyBoundary-like behavior
    # Specify q_func to prescribe soln(Q) at the boundary (InflowOutflow likely)
    # > q_plus = q_func(nodes, eos, q_minus, **kwargs)
    # Specify (*note) q_flux_func to prescribe flux of Q through the boundary:
    # > q_flux_func(nodes, eos, q_minus, nhat, **kwargs)
    # Specify grad_q_func to prescribe grad(Q) at the boundary:
    # > s_plus = grad_q_func(nodes, eos, q_minus, grad_q_minus ,**kwargs)
    # Specify t_func to prescribe temperature at the boundary: (InflowOutflow likely)
    # > t_plus = t_func(nodes, eos, q_minus, **kwargs)
    # Prescribe (*note) t_flux to prescribe "flux of temperature" at the boundary:
    # > t_flux_func(nodes, eos, q_minus, nhat, **kwargs)
    # Prescribe grad(temperature) at the boundary with grad_t_func:
    # > grad_t_plus = grad_t_func(nodes, eos, q_minus, grad_t_minus, **kwargs)
    # Fully prescribe the inviscid or viscous flux - unusual
    # inviscid_flux_func(nodes, eos, q_minus, **kwargs)
    # viscous_flux_func(nodes, eos, q_minus, grad_q_minus, t_minus,
    #                   grad_t_minus, nhat, **kwargs)
    #
    # (*note): Most people will never change these as they are used internally
    #          to compute a DG gradient of Q and temperature.

    def _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        return make_fluid_state(prescribed_soln(r=nodes, eos=gas_model.eos,
                                            **kwargs), gas_model)

    from mirgecom.boundary import PrescribedFluidBoundary
    domain_boundary = \
        PrescribedFluidBoundary(boundary_state_func=_boundary_state_func)

    gas_model = GasModel(eos=IdealSingleGas(gas_const=1.0),
                         transport=transport_model)

    nels_geom = 16
    a = 1.0
    b = 2.0
    mesh = get_box_mesh(dim=dim, a=a, b=b, n=nels_geom)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    boundary_discr = dcoll.discr_from_dd(BTAG_ALL)
    boundary_nodes = actx.thaw(boundary_discr.nodes())
    expected_boundary_solution = prescribed_soln(r=boundary_nodes, eos=gas_model.eos)

    nhat = actx.thaw(dcoll.normal(BTAG_ALL))
    print(f"{nhat=}")

    from mirgecom.flux import num_flux_central

    def scalar_flux_interior(int_tpair):
        from arraycontext import outer
        normal = actx.thaw(dcoll.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext),
                          normal)
        return op.project(dcoll, int_tpair.dd, "all_faces", flux_weak)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")
    # for velocities in each direction
    # err_max = 0.0
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))

        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity
            from mirgecom.initializers import Uniform
            initializer = Uniform(dim=dim, velocity=vel)
            cv = initializer(nodes, eos=gas_model.eos)
            state = make_fluid_state(cv, gas_model)
            state_minus = project_fluid_state(dcoll, "vol", BTAG_ALL,
                                              state, gas_model)

            print(f"{cv=}")
            temper = state.temperature
            print(f"{temper=}")

            cv_int_tpair = interior_trace_pair(dcoll, cv)
            cv_flux_int = scalar_flux_interior(cv_int_tpair)
            cv_flux_bc = domain_boundary.cv_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                               gas_model=gas_model,
                                               state_minus=state_minus)

            cv_flux_bc = op.project(dcoll, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    cv_flux_bc)

            cv_flux_bnd = cv_flux_bc + cv_flux_int

            t_int_tpair = interior_trace_pair(dcoll, temper)
            t_flux_int = scalar_flux_interior(t_int_tpair)
            t_flux_bc = \
                domain_boundary.temperature_gradient_flux(dcoll, dd_bdry=BTAG_ALL,
                                                          gas_model=gas_model,
                                                          state_minus=state_minus)

            t_flux_bc = op.project(dcoll, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    t_flux_bc)

            t_flux_bnd = t_flux_bc + t_flux_int

            i_flux_bc = \
                domain_boundary.inviscid_divergence_flux(dcoll, dd_bdry=BTAG_ALL,
                                                         gas_model=gas_model,
                                                         state_minus=state_minus)

            cv_int_pairs = interior_trace_pairs(dcoll, cv)
            state_pairs = make_fluid_state_trace_pairs(cv_int_pairs, gas_model)
            state_pair = state_pairs[0]

            nhat = actx.thaw(dcoll.normal(state_pair.dd))
            bnd_flux = flux_func(state_pair, gas_model, nhat)
            dd = state_pair.dd
            dd_allfaces = dd.with_boundary_tag(FACE_RESTR_ALL)
            dd_bdry = as_dofdesc(BTAG_ALL)
            i_flux_bc = op.project(dcoll, dd_bdry, dd_allfaces, i_flux_bc)
            i_flux_int = op.project(dcoll, dd, dd_allfaces, bnd_flux)

            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{cv_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import grad_operator
            dd_vol = as_dofdesc("vol")
            dd_allfaces = as_dofdesc("all_faces")
            grad_cv = grad_operator(dcoll, dd_vol, dd_allfaces, cv, cv_flux_bnd)
            grad_t = grad_operator(dcoll, dd_vol, dd_allfaces, temper, t_flux_bnd)
            grad_cv_minus = op.project(dcoll, "vol", BTAG_ALL, grad_cv)
            grad_t_minus = op.project(dcoll, "vol", BTAG_ALL, grad_t)

            print(f"{grad_cv_minus=}")
            print(f"{grad_t_minus=}")

            v_flux_bc = \
                domain_boundary.viscous_divergence_flux(
                    dcoll=dcoll, dd_bdry=BTAG_ALL, gas_model=gas_model,
                    state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                    grad_t_minus=grad_t_minus)
            print(f"{v_flux_bc=}")
            bc_soln = \
                domain_boundary._boundary_state_pair(dcoll, BTAG_ALL, gas_model,
                                                     state_minus=state_minus).ext.cv
            assert actx.np.equal(bc_soln, expected_boundary_solution)
