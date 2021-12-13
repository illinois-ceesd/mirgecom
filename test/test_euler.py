"""Test the Euler gas dynamics module."""

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
import math
from functools import partial

from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)

from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.euler import euler_operator
from mirgecom.fluid import make_conserved
from mirgecom.initializers import Vortex2D, Lump, MulticomponentLump
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    DummyBoundary
)
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from mirgecom.simutil import max_component_norm

from grudge.shortcuts import make_visualizer
from mirgecom.inviscid import get_inviscid_timestep

from mirgecom.integrators import rk4_step

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("nspecies", [0, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_uniform_rhs(actx_factory, nspecies, dim, order):
    """Test the inviscid rhs using a trivial constant/uniform state.

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

        discr = EagerDGDiscretization(actx, mesh, order=order)
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
        gas_model = GasModel(eos=IdealSingleGas())
        fluid_state = make_fluid_state(cv, gas_model)

        expected_rhs = make_conserved(
            dim, q=make_obj_array([discr.zeros(actx)
                                   for i in range(num_equations)])
        )

        from mirgecom.inviscid import inviscid_flux_rusanov
        boundaries = {BTAG_ALL: DummyBoundary()}
        inviscid_rhs = euler_operator(
            discr, state=fluid_state, gas_model=gas_model, boundaries=boundaries,
            time=0.0, inviscid_numerical_flux_func=inviscid_flux_rusanov)

        rhs_resid = inviscid_rhs - expected_rhs

        rho_resid = rhs_resid.mass
        rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum
        rhoy_resid = rhs_resid.species_mass

        rho_rhs = inviscid_rhs.mass
        rhoe_rhs = inviscid_rhs.energy
        rhov_rhs = inviscid_rhs.momentum
        rhoy_rhs = inviscid_rhs.species_mass

        logger.info(
            f"rho_rhs  = {rho_rhs}\n"
            f"rhoe_rhs = {rhoe_rhs}\n"
            f"rhov_rhs = {rhov_rhs}\n"
            f"rhoy_rhs = {rhoy_rhs}\n"
        )

        def inf_norm(x):
            return actx.to_numpy(discr.norm(x, np.inf))

        assert inf_norm(rho_resid) < tolerance
        assert inf_norm(rhoe_resid) < tolerance
        for i in range(dim):
            assert inf_norm(mom_resid[i]) < tolerance
        for i in range(nspecies):
            assert inf_norm(rhoy_resid[i]) < tolerance

        err_max = inf_norm(rho_resid)
        eoc_rec0.add_data_point(1.0 / nel_1d, err_max)

        # set a non-zero, but uniform velocity component
        for i in range(len(mom_input)):
            mom_input[i] = discr.zeros(actx) + (-1.0) ** i

        cv = make_conserved(
            dim, mass=mass_input, energy=energy_input, momentum=mom_input,
            species_mass=species_mass_input)
        gas_model = GasModel(eos=IdealSingleGas())
        fluid_state = make_fluid_state(cv, gas_model)

        boundaries = {BTAG_ALL: DummyBoundary()}
        inviscid_rhs = euler_operator(
            discr, state=fluid_state, gas_model=gas_model, boundaries=boundaries,
            time=0.0, inviscid_numerical_flux_func=inviscid_flux_rusanov)
        rhs_resid = inviscid_rhs - expected_rhs

        rho_resid = rhs_resid.mass
        rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum
        rhoy_resid = rhs_resid.species_mass

        assert inf_norm(rho_resid) < tolerance
        assert inf_norm(rhoe_resid) < tolerance

        for i in range(dim):
            assert inf_norm(mom_resid[i]) < tolerance
        for i in range(nspecies):
            assert inf_norm(rhoy_resid[i]) < tolerance

        err_max = inf_norm(rho_resid)
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


@pytest.mark.parametrize("order", [1, 2, 3])
def test_vortex_rhs(actx_factory, order):
    """Test the inviscid rhs using the non-trivial 2D isentropic vortex.

    The case is configured to yield rhs = 0. Checks several different orders
    and refinement levels to check error behavior.
    """
    actx = actx_factory()

    dim = 2

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    from meshmode.mesh.generation import generate_regular_rect_mesh

    for nel_1d in [32, 48, 64]:

        mesh = generate_regular_rect_mesh(
            a=(-5,) * dim, b=(5,) * dim, nelements_per_axis=(nel_1d,) * dim,
        )

        logger.info(
            f"Number of {dim}d elements:  {mesh.nelements}"
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(discr.nodes(), actx)

        # Init soln with Vortex and expected RHS = 0
        vortex = Vortex2D(center=[0, 0], velocity=[0, 0])
        vortex_soln = vortex(nodes)
        gas_model = GasModel(eos=IdealSingleGas())
        fluid_state = make_fluid_state(vortex_soln, gas_model)

        def _vortex_boundary(discr, btag, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            return make_fluid_state(vortex(x_vec=nodes, **kwargs), gas_model)

        boundaries = {
            BTAG_ALL: PrescribedFluidBoundary(boundary_state_func=_vortex_boundary)
        }

        from mirgecom.inviscid import inviscid_flux_rusanov
        inviscid_rhs = euler_operator(
            discr, state=fluid_state, gas_model=gas_model, boundaries=boundaries,
            time=0.0, inviscid_numerical_flux_func=inviscid_flux_rusanov)

        err_max = max_component_norm(discr, inviscid_rhs, np.inf)

        eoc_rec.add_data_point(1.0 / nel_1d, err_max)

    logger.info(
        f"Error for (dim,order) = ({dim},{order}):\n"
        f"{eoc_rec}"
    )

    assert (
        eoc_rec.order_estimate() >= order - 0.5
        or eoc_rec.max_error() < 1e-11
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_lump_rhs(actx_factory, dim, order):
    """Test the inviscid rhs using the non-trivial mass lump case.

    The case is tested against the analytic expressions of the RHS.
    Checks several different orders and refinement levels to check error behavior.
    """
    actx = actx_factory()

    tolerance = 1e-10
    maxxerr = 0.0

    from pytools.convergence import EOCRecorder

    eoc_rec = EOCRecorder()

    for nel_1d in [4, 8, 12]:
        from meshmode.mesh.generation import (
            generate_regular_rect_mesh,
        )

        mesh = generate_regular_rect_mesh(
            a=(-5,) * dim, b=(5,) * dim, nelements_per_axis=(nel_1d,) * dim,
        )

        logger.info(f"Number of elements: {mesh.nelements}")

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(discr.nodes(), actx)

        # Init soln with Lump and expected RHS = 0
        center = np.zeros(shape=(dim,))
        velocity = np.zeros(shape=(dim,))
        lump = Lump(dim=dim, center=center, velocity=velocity)
        lump_soln = lump(nodes)
        gas_model = GasModel(eos=IdealSingleGas())
        fluid_state = make_fluid_state(lump_soln, gas_model)

        def _lump_boundary(discr, btag, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            return make_fluid_state(lump(x_vec=nodes, cv=state_minus, **kwargs),
                                    gas_model)

        boundaries = {
            BTAG_ALL: PrescribedFluidBoundary(boundary_state_func=_lump_boundary)
        }

        from mirgecom.inviscid import inviscid_flux_rusanov
        inviscid_rhs = euler_operator(
            discr, state=fluid_state, gas_model=gas_model, boundaries=boundaries,
            time=0.0, inviscid_numerical_flux_func=inviscid_flux_rusanov
        )
        expected_rhs = lump.exact_rhs(discr, cv=lump_soln, time=0)

        err_max = max_component_norm(discr, inviscid_rhs-expected_rhs, np.inf)
        if err_max > maxxerr:
            maxxerr = err_max

        eoc_rec.add_data_point(1.0 / nel_1d, err_max)
    logger.info(f"Max error: {maxxerr}")

    logger.info(
        f"Error for (dim,order) = ({dim},{order}):\n"
        f"{eoc_rec}"
    )

    assert (
        eoc_rec.order_estimate() >= order - 0.5
        or eoc_rec.max_error() < tolerance
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("v0", [0.0, 1.0])
def test_multilump_rhs(actx_factory, dim, order, v0):
    """Test the Euler rhs using the non-trivial 1, 2, and 3D mass lump case.

    The case is tested against the analytic expressions of the RHS. Checks several
    different orders and refinement levels to check error behavior.
    """
    actx = actx_factory()
    nspecies = 10
    tolerance = 1e-8
    maxxerr = 0.0

    from pytools.convergence import EOCRecorder

    eoc_rec = EOCRecorder()

    for nel_1d in [4, 8, 12]:
        from meshmode.mesh.generation import (
            generate_regular_rect_mesh,
        )

        mesh = generate_regular_rect_mesh(
            a=(-1,) * dim, b=(1,) * dim, nelements_per_axis=(nel_1d,) * dim,
        )

        logger.info(f"Number of elements: {mesh.nelements}")

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(discr.nodes(), actx)

        centers = make_obj_array([np.zeros(shape=(dim,)) for i in range(nspecies)])
        spec_y0s = np.ones(shape=(nspecies,))
        spec_amplitudes = np.ones(shape=(nspecies,))

        velocity = np.zeros(shape=(dim,))
        velocity[0] = v0
        rho0 = 2.0

        lump = MulticomponentLump(dim=dim, nspecies=nspecies, rho0=rho0,
                                  spec_centers=centers, velocity=velocity,
                                  spec_y0s=spec_y0s, spec_amplitudes=spec_amplitudes)

        lump_soln = lump(nodes)
        gas_model = GasModel(eos=IdealSingleGas())
        fluid_state = make_fluid_state(lump_soln, gas_model)

        def _my_boundary(discr, btag, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            return make_fluid_state(lump(x_vec=nodes, **kwargs), gas_model)

        boundaries = {
            BTAG_ALL: PrescribedFluidBoundary(boundary_state_func=_my_boundary)
        }

        from mirgecom.inviscid import inviscid_flux_rusanov
        inviscid_rhs = euler_operator(
            discr, state=fluid_state, gas_model=gas_model, boundaries=boundaries,
            time=0.0, inviscid_numerical_flux_func=inviscid_flux_rusanov
        )
        expected_rhs = lump.exact_rhs(discr, cv=lump_soln, time=0)

        print(f"inviscid_rhs = {inviscid_rhs}")
        print(f"expected_rhs = {expected_rhs}")

        err_max = actx.to_numpy(
            discr.norm((inviscid_rhs-expected_rhs), np.inf))
        if err_max > maxxerr:
            maxxerr = err_max

        eoc_rec.add_data_point(1.0 / nel_1d, err_max)

        logger.info(f"Max error: {maxxerr}")

    logger.info(
        f"Error for (dim,order) = ({dim},{order}):\n"
        f"{eoc_rec}"
    )

    assert (
        eoc_rec.order_estimate() >= order - 0.5
        or eoc_rec.max_error() < tolerance
    )


# Basic timestepping loop for the Euler operator
def _euler_flow_stepper(actx, parameters):
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    mesh = parameters["mesh"]
    t = parameters["time"]
    order = parameters["order"]
    t_final = parameters["tfinal"]
    initializer = parameters["initializer"]
    exittol = parameters["exittol"]
    casename = parameters["casename"]
    boundaries = parameters["boundaries"]
    eos = parameters["eos"]
    cfl = parameters["cfl"]
    dt = parameters["dt"]
    constantcfl = parameters["constantcfl"]
    nstepstatus = parameters["nstatus"]

    if t_final <= t:
        return(0.0)

    rank = 0
    dim = mesh.dim
    istep = 0

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(discr.nodes(), actx)

    cv = initializer(nodes)
    gas_model = GasModel(eos=eos)
    fluid_state = make_fluid_state(cv, gas_model)

    sdt = cfl * get_inviscid_timestep(discr, fluid_state)

    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    logger.info(
        f"Num {dim}d order-{order} elements: {mesh.nelements}\n"
        f"Timestep:        {dt}\n"
        f"Final time:      {t_final}\n"
        f"Status freq:     {nstepstatus}\n"
        f"Initialization:  {initname}\n"
        f"EOS:             {eosname}"
    )

    vis = make_visualizer(discr, order)

    def write_soln(state, write_status=True):
        dv = eos.dependent_vars(cv=state)
        expected_result = initializer(nodes, t=t)
        result_resid = state - expected_result
        maxerr = [np.max(np.abs(result_resid[i].get())) for i in range(dim + 2)]
        mindv = [np.min(dvfld.get()) for dvfld in dv]
        maxdv = [np.max(dvfld.get()) for dvfld in dv]

        if write_status is True:
            statusmsg = (
                f"Status: Step({istep}) Time({t})\n"
                f"------   P({mindv[0]},{maxdv[0]})\n"
                f"------   T({mindv[1]},{maxdv[1]})\n"
                f"------   dt,cfl = ({dt},{cfl})\n"
                f"------   Err({maxerr})"
            )
            logger.info(statusmsg)

        io_fields = ["cv", state]
        io_fields += eos.split_fields(dim, dv)
        io_fields.append(("exact_soln", expected_result))
        io_fields.append(("residual", result_resid))
        nameform = casename + "-{iorank:04d}-{iostep:06d}.vtu"
        visfilename = nameform.format(iorank=rank, iostep=istep)
        vis.write_vtk_file(visfilename, io_fields)

        return maxerr

    from mirgecom.inviscid import inviscid_flux_rusanov

    def rhs(t, q):
        fluid_state = make_fluid_state(q, gas_model)
        return euler_operator(discr, fluid_state, boundaries=boundaries,
                              gas_model=gas_model, time=t,
                              inviscid_numerical_flux_func=inviscid_flux_rusanov)

    filter_order = 8
    eta = .5
    alpha = -1.0*np.log(np.finfo(float).eps)
    nummodes = int(1)
    for _ in range(dim):
        nummodes *= int(order + dim + 1)
    nummodes /= math.factorial(int(dim))
    cutoff = int(eta * order)

    from mirgecom.filter import (
        exponential_mode_response_function as xmrfunc,
        filter_modally
    )
    frfunc = partial(xmrfunc, alpha=alpha, filter_order=filter_order)

    while t < t_final:

        if constantcfl is True:
            dt = sdt
        else:
            cfl = dt / sdt

        if nstepstatus > 0:
            if istep % nstepstatus == 0:
                write_soln(state=cv)

        cv = rk4_step(cv, t, dt, rhs)
        cv = filter_modally(discr, "vol", cutoff, frfunc, cv)
        fluid_state = make_fluid_state(cv, gas_model)

        t += dt
        istep += 1

        sdt = cfl * get_inviscid_timestep(discr, fluid_state)

    if nstepstatus > 0:
        logger.info("Writing final dump.")
        maxerr = max(write_soln(cv, False))
    else:
        expected_result = initializer(nodes, time=t)
        maxerr = max_component_norm(discr, cv-expected_result, np.inf)

    logger.info(f"Max Error: {maxerr}")
    if maxerr > exittol:
        raise ValueError("Solution failed to follow expected result.")

    return(maxerr)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_isentropic_vortex(actx_factory, order):
    """Advance the 2D isentropic vortex case in time with non-zero velocities.

    This test uses an RK4 timestepping scheme, and checks the advanced field values
    against the exact/analytic expressions. This tests all parts of the Euler module
    working together, with results converging at the expected rates vs. the order.
    """
    actx = actx_factory()

    dim = 2

    from pytools.convergence import EOCRecorder

    eoc_rec = EOCRecorder()

    for nel_1d in [16, 32, 64]:
        from meshmode.mesh.generation import (
            generate_regular_rect_mesh,
        )

        mesh = generate_regular_rect_mesh(
            a=(-5.0,) * dim, b=(5.0,) * dim, nelements_per_axis=(nel_1d,) * dim
        )

        exittol = 1.0
        t_final = 0.001
        cfl = 1.0
        vel = np.zeros(shape=(dim,))
        orig = np.zeros(shape=(dim,))
        vel[:dim] = 1.0
        dt = .0001
        initializer = Vortex2D(center=orig, velocity=vel)
        casename = "Vortex"

        def _vortex_boundary(discr, btag, state_minus, gas_model, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            return make_fluid_state(initializer(x_vec=nodes, **kwargs), gas_model)

        boundaries = {
            BTAG_ALL: PrescribedFluidBoundary(boundary_state_func=_vortex_boundary)
        }

        eos = IdealSingleGas()
        t = 0
        flowparams = {"dim": dim, "dt": dt, "order": order, "time": t,
                      "boundaries": boundaries, "initializer": initializer,
                      "eos": eos, "casename": casename, "mesh": mesh,
                      "tfinal": t_final, "exittol": exittol, "cfl": cfl,
                      "constantcfl": False, "nstatus": 0}
        maxerr = _euler_flow_stepper(actx, flowparams)
        eoc_rec.add_data_point(1.0 / nel_1d, maxerr)

    logger.info(
        f"Error for (dim,order) = ({dim},{order}):\n"
        f"{eoc_rec}"
    )

    assert (
        eoc_rec.order_estimate() >= order - 0.5
        or eoc_rec.max_error() < 1e-11
    )
