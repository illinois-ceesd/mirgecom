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

from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import split_conserved
from mirgecom.initializers import Lump
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.eos import IdealSingleGas
from grudge.eager import (
    EagerDGDiscretization,
    interior_trace_pair
)
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
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 3
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    eos = IdealSingleGas()
    orig = np.zeros(shape=(dim,))
    nhat = thaw(actx, discr.normal(BTAG_ALL))
    #    normal_mag = actx.np.sqrt(np.dot(normal, normal))
    #    nhat_mult = 1.0 / normal_mag
    #    nhat = normal * make_obj_array([nhat_mult])

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
            from functools import partial
            bnd_norm = partial(discr.norm, p=np.inf, dd=BTAG_ALL)

            bnd_pair = wall.boundary_pair(discr, uniform_state, t=0.0,
                                          btag=BTAG_ALL, eos=eos)
            bnd_cv_int = split_conserved(dim, bnd_pair.int)
            bnd_cv_ext = split_conserved(dim, bnd_pair.ext)

            # check that mass and energy are preserved
            mass_resid = bnd_cv_int.mass - bnd_cv_ext.mass
            mass_err = bnd_norm(mass_resid)
            assert mass_err == 0.0
            energy_resid = bnd_cv_int.energy - bnd_cv_ext.energy
            energy_err = bnd_norm(energy_resid)
            assert energy_err == 0.0

            # check that exterior momentum term is mom_interior - 2 * mom_normal
            mom_norm_comp = np.dot(bnd_cv_int.momentum, nhat)
            mom_norm = nhat * mom_norm_comp
            expected_mom_ext = bnd_cv_int.momentum - 2.0 * mom_norm
            mom_resid = bnd_cv_ext.momentum - expected_mom_ext
            mom_err = bnd_norm(mom_resid)

            assert mom_err == 0.0


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_slipwall_flux(actx_factory, dim, order):
    """Check for zero boundary flux.

    Check for vanishing flux across the slipwall.
    """
    actx = actx_factory()

    wall = AdiabaticSlipBoundary()
    eos = IdealSingleGas()

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for np1 in [4, 8, 12]:
        from meshmode.mesh.generation import generate_regular_rect_mesh

        mesh = generate_regular_rect_mesh(
            a=(-0.5,) * dim, b=(0.5,) * dim, n=(np1,) * dim
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(actx, discr.nodes())
        nhat = thaw(actx, discr.normal(BTAG_ALL))
        h = 1.0 / np1

        from functools import partial
        bnd_norm = partial(discr.norm, p=np.inf, dd=BTAG_ALL)

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
                bnd_pair = wall.boundary_pair(discr, uniform_state, t=0.0,
                                              btag=BTAG_ALL, eos=eos)

                # Check the total velocity component normal
                # to each surface.  It should be zero.  The
                # numerical fluxes cannot be zero.
                avg_state = 0.5*(bnd_pair.int + bnd_pair.ext)
                acv = split_conserved(dim, avg_state)
                err_max = max(err_max, bnd_norm(np.dot(acv.momentum, nhat)))

                from mirgecom.inviscid import inviscid_facial_flux
                bnd_flux = split_conserved(dim, inviscid_facial_flux(discr, eos,
                                                             bnd_pair, local=True))
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
# @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
# def test_noslip(actx_factory, dim, order):
def test_noslip(actx_factory, dim):
    """Check IsothermalNoSlipBoundary viscous boundary treatment."""
    actx = actx_factory()
    order = 1

    wall_temp = 1.0
    kappa = 3.0
    sigma = 5.0

    from mirgecom.transport import SimpleTransport
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa)

    from mirgecom.boundary import IsothermalNoSlipBoundary
    wall = IsothermalNoSlipBoundary(wall_temperature=wall_temp)
    eos = IdealSingleGas(transport_model=transport_model, gas_const=1.0)

    # from pytools.convergence import EOCRecorder
    # eoc = EOCRecorder()

    #    for np1 in [4, 8, 12]:
    npts_geom = 17
    a = 1.0
    b = 2.0
    mesh = _get_box_mesh(dim=dim, a=a, b=b, n=npts_geom)
    #    boundaries = {BTAG_ALL: wall}
    # for i in range(dim):
    #     boundaries[DTAG_BOUNDARY("-"+str(i+1))] = 0
    #     boundaries[DTAG_BOUNDARY("+"+str(i+1))] = 0

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    nhat = thaw(actx, discr.normal(BTAG_ALL))
    print(f"{nhat=}")
    # h = 1.0 / np1

    from mirgecom.flux import central_scalar_flux

    def scalar_flux_interior(int_tpair):
        normal = thaw(actx, discr.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = central_scalar_flux(int_tpair, normal)
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
            uniform_state = initializer(nodes, eos=eos)
            cv = split_conserved(dim, uniform_state)
            print(f"{cv=}")
            temperature = eos.temperature(cv)
            print(f"{temperature=}")

            q_int_tpair = interior_trace_pair(discr, uniform_state)
            q_flux_int = scalar_flux_interior(q_int_tpair)
            q_flux_bc = wall.get_q_flux(discr, btag=BTAG_ALL,
                                        eos=eos, q=uniform_state)
            q_flux_bnd = q_flux_bc + q_flux_int

            t_int_tpair = interior_trace_pair(discr, temperature)
            t_flux_int = scalar_flux_interior(t_int_tpair)
            t_flux_bc = wall.get_t_flux(discr, btag=BTAG_ALL, eos=eos,
                                        q=uniform_state, temperature=temperature)
            t_flux_bnd = t_flux_bc + t_flux_int

            from mirgecom.inviscid import inviscid_facial_flux
            i_flux_bc = wall.get_inviscid_flux(discr, btag=BTAG_ALL, eos=eos,
                                               q=uniform_state)
            i_flux_int = inviscid_facial_flux(discr, eos=eos, q_tpair=q_int_tpair)
            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{q_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import dg_grad_low
            grad_q = dg_grad_low(discr, uniform_state, q_flux_bnd)
            grad_t = dg_grad_low(discr, temperature, t_flux_bnd)
            print(f"{grad_q=}")
            print(f"{grad_t=}")

            v_flux_bc = wall.get_viscous_flux(discr, btag=BTAG_ALL, eos=eos,
                                              q=uniform_state, grad_q=grad_q,
                                              t=temperature, grad_t=grad_t)
            print(f"{v_flux_bc=}")


@pytest.mark.parametrize("dim", [1, 2, 3])
# @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
# def test_noslip(actx_factory, dim, order):
def test_prescribedviscous(actx_factory, dim):
    """Check viscous prescribed boundary treatment."""
    actx = actx_factory()
    order = 1

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

    from mirgecom.boundary import PrescribedViscousBoundary
    wall = PrescribedViscousBoundary()
    eos = IdealSingleGas(transport_model=transport_model, gas_const=1.0)

    npts_geom = 17
    a = 1.0
    b = 2.0
    mesh = _get_box_mesh(dim=dim, a=a, b=b, n=npts_geom)
    #    boundaries = {BTAG_ALL: wall}
    # for i in range(dim):
    #     boundaries[DTAG_BOUNDARY("-"+str(i+1))] = 0
    #     boundaries[DTAG_BOUNDARY("+"+str(i+1))] = 0

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    nhat = thaw(actx, discr.normal(BTAG_ALL))
    print(f"{nhat=}")

    from mirgecom.flux import central_scalar_flux

    def scalar_flux_interior(int_tpair):
        normal = thaw(actx, discr.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = central_scalar_flux(int_tpair, normal)
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
            uniform_state = initializer(nodes, eos=eos)
            cv = split_conserved(dim, uniform_state)
            print(f"{cv=}")
            temperature = eos.temperature(cv)
            print(f"{temperature=}")

            q_int_tpair = interior_trace_pair(discr, uniform_state)
            q_flux_int = scalar_flux_interior(q_int_tpair)
            q_flux_bc = wall.get_q_flux(discr, btag=BTAG_ALL,
                                        eos=eos, q=uniform_state)
            q_flux_bnd = q_flux_bc + q_flux_int

            t_int_tpair = interior_trace_pair(discr, temperature)
            t_flux_int = scalar_flux_interior(t_int_tpair)
            t_flux_bc = wall.get_t_flux(discr, btag=BTAG_ALL, eos=eos,
                                        q=uniform_state, temperature=temperature)
            t_flux_bnd = t_flux_bc + t_flux_int

            from mirgecom.inviscid import inviscid_facial_flux
            i_flux_bc = wall.get_inviscid_flux(discr, btag=BTAG_ALL, eos=eos,
                                               q=uniform_state)
            i_flux_int = inviscid_facial_flux(discr, eos=eos, q_tpair=q_int_tpair)
            i_flux_bnd = i_flux_bc + i_flux_int

            print(f"{q_flux_bnd=}")
            print(f"{t_flux_bnd=}")
            print(f"{i_flux_bnd=}")

            from mirgecom.operators import dg_grad_low
            grad_q = dg_grad_low(discr, uniform_state, q_flux_bnd)
            grad_t = dg_grad_low(discr, temperature, t_flux_bnd)
            print(f"{grad_q=}")
            print(f"{grad_t=}")

            v_flux_bc = wall.get_viscous_flux(discr, btag=BTAG_ALL, eos=eos,
                                              q=uniform_state, grad_q=grad_q,
                                              t=temperature, grad_t=grad_t)
            print(f"{v_flux_bc=}")
