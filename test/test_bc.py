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

from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.initializers import Lump
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.eos import IdealSingleGas
from grudge.eager import EagerDGDiscretization
from grudge.trace_pair import interior_trace_pair, interior_trace_pairs
from grudge.trace_pair import TracePair
from grudge.dof_desc import as_dofdesc

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

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

logger = logging.getLogger(__name__)


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
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(discr.nodes(), actx)
    eos = IdealSingleGas()
    orig = np.zeros(shape=(dim,))
    nhat = thaw(discr.normal(BTAG_ALL), actx)
    gas_model = GasModel(eos=eos)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocity going along each direction
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))
        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity  # Check incoming normal
            initializer = Lump(dim=dim, center=orig, velocity=vel, rhoamp=0.0)
            wall = AdiabaticSlipBoundary()

            uniform_state = initializer(nodes)
            cv_minus = discr.project("vol", BTAG_ALL, uniform_state)
            state_minus = make_fluid_state(cv=cv_minus, gas_model=gas_model)

            def bnd_norm(vec):
                return actx.to_numpy(discr.norm(vec, p=np.inf, dd=BTAG_ALL))

            state_plus = \
                wall.adiabatic_slip_state(discr, btag=BTAG_ALL, gas_model=gas_model,
                                          state_minus=state_minus)

            bnd_pair = TracePair(BTAG_ALL, interior=state_minus.cv,
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

    wall = AdiabaticSlipBoundary()
    gas_model = GasModel(eos=IdealSingleGas())

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for nel_1d in [4, 8, 12]:
        from meshmode.mesh.generation import generate_regular_rect_mesh

        mesh = generate_regular_rect_mesh(
            a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(discr.nodes(), actx)
        nhat = thaw(discr.normal(BTAG_ALL), actx)
        h = 1.0 / nel_1d

        def bnd_norm(vec):
            return actx.to_numpy(discr.norm(vec, p=np.inf, dd=BTAG_ALL))

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

                interior_soln = project_fluid_state(discr, "vol", BTAG_ALL,
                                                    state=fluid_state,
                                                    gas_model=gas_model)

                bnd_soln = wall.adiabatic_slip_state(discr, btag=BTAG_ALL,
                                                     gas_model=gas_model,
                                                     state_minus=interior_soln)

                bnd_pair = TracePair(BTAG_ALL, interior=interior_soln.cv,
                                     exterior=bnd_soln.cv)
                state_pair = TracePair(BTAG_ALL, interior=interior_soln,
                                       exterior=bnd_soln)

                # Check the total velocity component normal
                # to each surface.  It should be zero.  The
                # numerical fluxes cannot be zero.
                avg_state = 0.5*(bnd_pair.int + bnd_pair.ext)
                err_max = max(err_max, bnd_norm(np.dot(avg_state.momentum, nhat)))

                normal = thaw(discr.normal(BTAG_ALL), actx)
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


# Box grid generator widget lifted from @majosm's diffusion tester
def _get_box_mesh(dim, a, b, n):
    dim_names = ["x", "y", "z"]
    boundary_tag_to_face = {}
    for i in range(dim):
        boundary_tag_to_face["-"+str(i+1)] = ["-"+dim_names[i]]
        boundary_tag_to_face["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh
    return generate_regular_rect_mesh(a=(a,)*dim, b=(b,)*dim, n=(n,)*dim,
        boundary_tag_to_face=boundary_tag_to_face)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("flux_func", [inviscid_facial_flux_rusanov,
                                       inviscid_facial_flux_hll])
def test_noslip(actx_factory, dim, flux_func):
    """Check IsothermalNoSlipBoundary viscous boundary treatment."""
    actx = actx_factory()
    order = 1

    wall_temp = 2.0
    kappa = 3.0
    sigma = 5.0
    fluid_temp = 1.0
    exp_temp_bc_val = 2*wall_temp - fluid_temp

    from mirgecom.transport import SimpleTransport
    from mirgecom.boundary import IsothermalNoSlipBoundary

    gas_model = GasModel(eos=IdealSingleGas(gas_const=1.0),
                         transport=SimpleTransport(viscosity=sigma,
                                                   thermal_conductivity=kappa))

    wall = IsothermalNoSlipBoundary(wall_temperature=wall_temp)

    npts_geom = 17
    a = 1.0
    b = 2.0
    mesh = _get_box_mesh(dim=dim, a=a, b=b, n=npts_geom)

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(discr.nodes(), actx)
    nhat = thaw(discr.normal(BTAG_ALL), actx)
    print(f"{nhat=}")

    from mirgecom.flux import num_flux_central

    def scalar_flux_interior(int_tpair):
        from arraycontext import outer
        normal = thaw(discr.normal(int_tpair.dd), actx)
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext), normal)
        return discr.project(int_tpair.dd, "all_faces", flux_weak)

    # utility to compare stuff on the boundary only
    # from functools import partial
    # bnd_norm = partial(discr.norm, p=np.inf, dd=BTAG_ALL)

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
            state_minus = project_fluid_state(discr, "vol", BTAG_ALL,
                                              uniform_state, gas_model)

            print(f"{uniform_state=}")
            temper = uniform_state.temperature
            print(f"{temper=}")

            expected_noslip_cv = 1.0*state_minus.cv
            expected_noslip_cv = expected_noslip_cv.replace(
                momentum=-expected_noslip_cv.momentum)

            expected_noslip_momentum = -vel*state_minus.mass_density
            expected_temperature_bc = \
                exp_temp_bc_val * state_minus.mass_density

            print(f"{expected_temperature_bc=}")
            print(f"{expected_noslip_cv=}")

            cv_interior_pairs = interior_trace_pairs(discr, uniform_state.cv)
            cv_int_tpair = cv_interior_pairs[0]

            state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs, gas_model)
            state_pair = state_pairs[0]

            cv_flux_int = scalar_flux_interior(cv_int_tpair)
            print(f"{cv_flux_int=}")

            wall_state = wall.isothermal_noslip_state(
                discr, btag=BTAG_ALL, gas_model=gas_model, state_minus=state_minus)
            print(f"{wall_state=}")

            cv_grad_flux_wall = wall.cv_gradient_flux(discr, btag=BTAG_ALL,
                                                 gas_model=gas_model,
                                                 state_minus=state_minus)

            cv_grad_flux_allfaces = \
                op.project(discr, as_dofdesc(BTAG_ALL),
                           as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                           cv_grad_flux_wall)

            print(f"{cv_grad_flux_wall=}")

            cv_flux_bnd = cv_grad_flux_allfaces + cv_flux_int

            temperature_bc = wall.temperature_bc(state_minus)
            print(f"{temperature_bc=}")

            t_int_tpair = interior_trace_pair(discr, temper)
            t_flux_int = scalar_flux_interior(t_int_tpair)
            t_flux_bc = wall.temperature_gradient_flux(discr, btag=BTAG_ALL,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus)

            t_flux_bc = op.project(discr, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    t_flux_bc)

            t_flux_bnd = t_flux_bc + t_flux_int

            i_flux_bc = wall.inviscid_divergence_flux(discr, btag=BTAG_ALL,
                                                      gas_model=gas_model,
                                                      state_minus=state_minus)

            nhat = thaw(discr.normal(state_pair.dd), actx)
            bnd_flux = flux_func(state_pair, gas_model, nhat)
            dd = state_pair.dd
            dd_allfaces = dd.with_dtag("all_faces")
            i_flux_int = discr.project(dd, dd_allfaces, bnd_flux)
            bc_dd = as_dofdesc(BTAG_ALL)
            i_flux_bc = discr.project(bc_dd, dd_allfaces, i_flux_bc)

            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{cv_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import grad_operator
            dd_vol = as_dofdesc("vol")
            dd_faces = as_dofdesc("all_faces")
            grad_cv_minus = \
                discr.project("vol", BTAG_ALL,
                              grad_operator(discr, dd_vol, dd_faces,
                                            uniform_state.cv, cv_flux_bnd))
            grad_t_minus = discr.project("vol", BTAG_ALL,
                                         grad_operator(discr, dd_vol, dd_faces,
                                                       temper, t_flux_bnd))

            print(f"{grad_cv_minus=}")
            print(f"{grad_t_minus=}")

            v_flux_bc = wall.viscous_divergence_flux(discr, btag=BTAG_ALL,
                                                     gas_model=gas_model,
                                                     state_minus=state_minus,
                                                     grad_cv_minus=grad_cv_minus,
                                                     grad_t_minus=grad_t_minus)
            print(f"{v_flux_bc=}")

            assert wall_state.cv == expected_noslip_cv
            assert actx.np.all(temperature_bc == expected_temperature_bc)
            for idim in range(dim):
                assert actx.np.all(wall_state.momentum_density[idim]
                                   == expected_noslip_momentum[idim])


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

    def _boundary_state_func(discr, btag, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = discr.discr_from_dd(btag)
        nodes = thaw(bnd_discr.nodes(), actx)
        return make_fluid_state(prescribed_soln(r=nodes, eos=gas_model.eos,
                                            **kwargs), gas_model)

    from mirgecom.boundary import PrescribedFluidBoundary
    domain_boundary = \
        PrescribedFluidBoundary(boundary_state_func=_boundary_state_func)

    gas_model = GasModel(eos=IdealSingleGas(gas_const=1.0),
                         transport=transport_model)

    npts_geom = 17
    a = 1.0
    b = 2.0
    mesh = _get_box_mesh(dim=dim, a=a, b=b, n=npts_geom)

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(discr.nodes(), actx)
    boundary_discr = discr.discr_from_dd(BTAG_ALL)
    boundary_nodes = thaw(boundary_discr.nodes(), actx)
    expected_boundary_solution = prescribed_soln(r=boundary_nodes, eos=gas_model.eos)

    nhat = thaw(discr.normal(BTAG_ALL), actx)
    print(f"{nhat=}")

    from mirgecom.flux import num_flux_central

    def scalar_flux_interior(int_tpair):
        from arraycontext import outer
        normal = thaw(discr.normal(int_tpair.dd), actx)
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = outer(num_flux_central(int_tpair.int, int_tpair.ext),
                          normal)
        return discr.project(int_tpair.dd, "all_faces", flux_weak)

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
            state_minus = project_fluid_state(discr, "vol", BTAG_ALL,
                                              state, gas_model)

            print(f"{cv=}")
            temper = state.temperature
            print(f"{temper=}")

            cv_int_tpair = interior_trace_pair(discr, cv)
            cv_flux_int = scalar_flux_interior(cv_int_tpair)
            cv_flux_bc = domain_boundary.cv_gradient_flux(discr, btag=BTAG_ALL,
                                               gas_model=gas_model,
                                               state_minus=state_minus)

            cv_flux_bc = op.project(discr, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    cv_flux_bc)

            cv_flux_bnd = cv_flux_bc + cv_flux_int

            t_int_tpair = interior_trace_pair(discr, temper)
            t_flux_int = scalar_flux_interior(t_int_tpair)
            t_flux_bc = \
                domain_boundary.temperature_gradient_flux(discr, btag=BTAG_ALL,
                                                          gas_model=gas_model,
                                                          state_minus=state_minus)

            t_flux_bc = op.project(discr, as_dofdesc(BTAG_ALL),
                                    as_dofdesc(BTAG_ALL).with_dtag("all_faces"),
                                    t_flux_bc)

            t_flux_bnd = t_flux_bc + t_flux_int

            i_flux_bc = \
                domain_boundary.inviscid_divergence_flux(discr, btag=BTAG_ALL,
                                                         gas_model=gas_model,
                                                         state_minus=state_minus)

            cv_int_pairs = interior_trace_pairs(discr, cv)
            state_pairs = make_fluid_state_trace_pairs(cv_int_pairs, gas_model)
            state_pair = state_pairs[0]

            nhat = thaw(discr.normal(state_pair.dd), actx)
            bnd_flux = flux_func(state_pair, gas_model, nhat)
            dd = state_pair.dd
            dd_allfaces = dd.with_dtag("all_faces")
            bc_dd = as_dofdesc(BTAG_ALL)
            i_flux_bc = discr.project(bc_dd, dd_allfaces, i_flux_bc)
            i_flux_int = discr.project(dd, dd_allfaces, bnd_flux)

            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{cv_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import grad_operator
            dd_vol = as_dofdesc("vol")
            dd_faces = as_dofdesc("all_faces")
            grad_cv = grad_operator(discr, dd_vol, dd_faces, cv, cv_flux_bnd)
            grad_t = grad_operator(discr, dd_vol, dd_faces, temper, t_flux_bnd)
            grad_cv_minus = discr.project("vol", BTAG_ALL, grad_cv)
            grad_t_minus = discr.project("vol", BTAG_ALL, grad_t)

            print(f"{grad_cv_minus=}")
            print(f"{grad_t_minus=}")

            v_flux_bc = \
                domain_boundary.viscous_divergence_flux(discr=discr, btag=BTAG_ALL,
                                                        gas_model=gas_model,
                                                        state_minus=state_minus,
                                                        grad_cv_minus=grad_cv_minus,
                                                        grad_t_minus=grad_t_minus)
            print(f"{v_flux_bc=}")
            bc_soln = \
                domain_boundary._boundary_state_pair(discr, BTAG_ALL, gas_model,
                                                     state_minus=state_minus).ext.cv
            assert bc_soln == expected_boundary_solution
