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
from grudge.eager import EagerDGDiscretization
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
    """Check IsothermalNoSlip viscous boundary treatment."""
    actx = actx_factory()
    order = 1

    wall_temp = 100.0
    kappa = 3.0
    sigma = 5.0

    from mirgecom.transport import SimpleTransport
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa)

    from mirgecom.boundary import IsothermalNoSlip
    wall = IsothermalNoSlip(wall_temperature=wall_temp)
    eos = IdealSingleGas(transport_model=transport_model)

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
    # nhat = thaw(actx, discr.normal(BTAG_ALL))
    # h = 1.0 / np1

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

            # here's a boundary pair for the wall(s) just in case
            q_flux = wall.get_q_flux(discr, btag=BTAG_ALL, eos=eos, q=uniform_state)
            t_flux = wall.get_t_flux(discr, btag=BTAG_ALL, eos=eos, q=uniform_state)
            i_flux = wall.get_inviscid_flux(discr, btag=BTAG_ALL, eos=eos,
                                            q=uniform_state)
            print(f"{q_flux=}")
            print(f"{t_flux=}")
            print(f"{i_flux=}")

            # from mirgecom.operators import dg_grad_low
            # grad_q = dg_grad_low(discr, uniform_state, q_flux)

            # grad_cv = split_conserved(dim, grad_q)
            temperature = eos.temperature(cv)
            print(f"{temperature=}")
            # grad_t = dg_grad_low(discr, temperature, t_flux)

            # v_flux = wall.get_inviscid_flux(discr, btag=BTAG_ALL, eos=eos,
            #                                uniform_state, grad_uniform_state)
