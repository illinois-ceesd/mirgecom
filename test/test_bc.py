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

from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.euler import split_conserved
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
    normal = thaw(actx, discr.normal(BTAG_ALL))
    normal_mag = actx.np.sqrt(np.dot(normal, normal))
    nhat_mult = 1.0 / normal_mag
    nhat = normal * make_obj_array([nhat_mult])

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocity going along each direction
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))
        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity  # Check incoming normal
            initializer = Lump(center=orig, velocity=vel, rhoamp=0.0)
            wall = AdiabaticSlipBoundary()

            uniform_state = initializer(0, nodes)
            from functools import partial
            bnd_norm = partial(discr.norm, p=np.inf, dd=BTAG_ALL)

            bnd_pair = wall.boundary_pair(discr, uniform_state, t=0.0,
                                          btag=BTAG_ALL, eos=eos)
            bnd_cv_int = split_conserved(dim, bnd_pair.int)
            bnd_cv_ext = split_conserved(dim, bnd_pair.ext)

            # check that mass and energy are preserved
            mass_resid = bnd_cv_int.mass - bnd_cv_ext.mass
            mass_err = bnd_norm(mass_resid)
            assert(mass_err == 0.0)
            energy_resid = bnd_cv_int.energy - bnd_cv_ext.energy
            energy_err = bnd_norm(energy_resid)
            assert(energy_err == 0.0)

            # check that exterior momentum term is mom_interior - 2 * mom_normal
            mom_norm_comp = np.dot(bnd_cv_int.momentum, nhat)
            mom_norm = nhat * make_obj_array([mom_norm_comp])
            expected_mom_ext = bnd_cv_int.momentum - 2.0 * mom_norm
            mom_resid = bnd_cv_ext.momentum - expected_mom_ext
            mom_err = bnd_norm(mom_resid)

            assert(mom_err == 0.0)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_slipwall_flux(actx_factory, dim):
    """Check for zero boundary flux.

    Check for vanishing flux across the slipwall.
    """
    actx = actx_factory()

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 1
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    eos = IdealSingleGas()
    orig = np.zeros(shape=(dim,))
    normal = thaw(actx, discr.normal(BTAG_ALL))
    normal_mag = actx.np.sqrt(np.dot(normal, normal))
    nhat_mult = 1.0 / normal_mag
    nhat = normal * make_obj_array([nhat_mult])
    wall = AdiabaticSlipBoundary()

    from functools import partial
    bnd_norm = partial(discr.norm, p=np.inf, dd=BTAG_ALL)

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    # for velocities in each direction
    for vdir in range(dim):
        vel = np.zeros(shape=(dim,))

        # for velocity directions +1, and -1
        for parity in [1.0, -1.0]:
            vel[vdir] = parity
            initializer = Lump(center=orig, velocity=vel, rhoamp=0.0)
            uniform_state = initializer(0, nodes)

            bnd_pair = wall.boundary_pair(discr, uniform_state, t=0.0,
                                          btag=BTAG_ALL, eos=eos)

            # Check the total velocity component normal
            # to each surface.  It should be zero.  The
            # numerical fluxes cannot be zero.
            tol = 1e-16
            avg_state = 0.5*(bnd_pair.int + bnd_pair.ext)
            acv = split_conserved(dim, avg_state)
            bnd_norm_mom = np.dot(acv.momentum, nhat)
            bnd_mom = bnd_norm(bnd_norm_mom)
            assert(bnd_mom < tol)

            from mirgecom.euler import _facial_flux
            bnd_flux = split_conserved(dim, _facial_flux(discr, eos,
                                                         bnd_pair, local=True))
            mass_flux = bnd_flux.mass
            energy_flux = bnd_flux.energy

            # mass and energy flux at the boundary should be zero
            massflux_max = bnd_norm(mass_flux)
            energyflux_max = bnd_norm(energy_flux)
            assert(massflux_max < tol)
            assert(energyflux_max < tol)
