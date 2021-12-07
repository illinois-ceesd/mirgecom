"""Test the different flux methods."""

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
from grudge.eager import interior_trace_pair
from grudge.symbolic.primitives import TracePair
from mirgecom.fluid import make_conserved
from mirgecom.eos import IdealSingleGas
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from mirgecom.inviscid import inviscid_flux

logger = logging.getLogger(__name__)

# @pytest.mark.parametrize("nspecies", [0, 1, 10])
# @pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("nspecies", [0])
@pytest.mark.parametrize("dim", [1])
def test_lfr_flux(actx_factory, nspecies, dim):
    """Check inviscid flux against exact expected result.

    Directly check Lax-Friedrichs/Rusanov flux routine,
    :func:`mirgecom.flux.flux_lfr`,
    against the exact expected result. This test is designed to fail if the flux
    routine is broken.

    The expected inviscid flux is:
      F(q) = <rhoV, (E+p)V, rho(V.x.V) + pI, rhoY V>
    """
    actx = actx_factory()
    numeq = dim + 2 + nspecies

    gamma = 1.4
    eos = IdealSingleGas(gamma=gamma)
    # interior state
    p0 = 15.0
    rho0 = 2.0
    vel0 = np.zeros(shape=dim, )
    #vel0[0] = -4.0
    vel0[0] = 0.
    y0 = np.zeros(shape=nspecies, )
    c0 = np.sqrt(gamma*p0/rho0)
    # exterior state
    p1 = 20.0
    rho1 = 4.0
    vel1 = np.zeros(shape=dim, )
    #vel1[0] = 0.5
    vel1[0] = 0.
    y1 = np.zeros(shape=nspecies, )
    c1 = np.sqrt(gamma*p1/rho1)

    #q_int = np.zeros(shape=2+dim+nspecies, )
    #q_int[0] = rho0
    #vel_int = make_obj_array([vel0[idir] for idir in range(dim)])
    #mom_int =  mass_int*vel_int
    #energy_int = p0/0.4 + 0.5 * np.dot(mom_int, mom_int)/mass_int
    #species_mass_int = mass_int*y0

    mass_int = rho0
    vel_int = make_obj_array([vel0[idir] for idir in range(dim)])
    mom_int =  mass_int*vel_int
    p_int = p0
    energy_int = p_int/0.4 + 0.5 * np.dot(mom_int, mom_int)/mass_int
    species_mass_int = mass_int*make_obj_array([y0[idir] for idir in range(nspecies)])
    cv_int = make_conserved(dim, mass=mass_int, energy=energy_int, momentum=mom_int, species_mass=species_mass_int)
    flux_int = inviscid_flux(p_int, cv_int)

    mass_ext = rho1
    vel_ext = make_obj_array([vel1[idir] for idir in range(dim)])
    mom_ext =  mass_ext*vel_ext
    p_ext = p1
    energy_ext = p_ext/0.4 + 0.5 * np.dot(mom_ext, mom_ext)/mass_ext
    species_mass_ext = mass_ext*make_obj_array([y1[idir] for idir in range(nspecies)])
    cv_ext = make_conserved(dim, mass=mass_ext, energy=energy_ext, momentum=mom_ext, species_mass=species_mass_ext)
    flux_ext = inviscid_flux(p_ext, cv_ext)

    print(f"{cv_int=}")
    print(f"{flux_int=}")
    print(f"{cv_ext=}")
    print(f"{flux_ext=}")

    # interface normal
    normal = np.zeros(shape=dim, )
    normal[0] = 1

    # wave speed 
    lam = np.maximum(np.linalg.norm(vel0)+c0, np.linalg.norm(vel1)+c1)
    from mirgecom.flux import lfr

    flux_bnd = lfr(flux_ext@normal, flux_int@normal, cv_ext, cv_int, lam)
    print(f"{flux_bnd=}")

    # compute the flux in the interface normal direction
    vel_int_norm = np.dot(vel0, normal)
    vel_ext_norm = np.dot(vel1, normal)
    mass_flux_exact = 0.5*(mass_int*vel_int_norm + mass_ext*vel_ext_norm
                           - lam*(mass_ext - mass_int))
    mom_flux_exact = 0.5*(mass_int*vel_int_norm*vel_int +
                          mass_ext*vel_ext_norm*vel_ext +
                          p_int + p_ext -
                          lam*(mass_ext*vel_ext - mass_int*vel_int))
    mom_flux_exact = make_obj_array([mom_flux_exact for _ in range(dim)])
    energy_flux_exact = 0.5*(vel_ext_norm*(energy_ext+p_ext) +
                             vel_int_norm*(energy_int+p_int) -
                             lam*(energy_ext - energy_int))
    species_mass_flux_exact = 0.5*(vel_int_norm*species_mass_int +
                                   vel_int_norm*species_mass_ext -
                                   lam*(species_mass_ext - species_mass_int))
    species_mass_flux_exact = make_obj_array([species_mass_flux_exact[ispec] for ispec in range(nspecies)])

    flux_bnd_exact = make_conserved(dim, mass=mass_flux_exact, energy=energy_flux_exact,
                                momentum=mom_flux_exact, species_mass=species_mass_flux_exact)
    print(f"{flux_bnd_exact=}")


    flux_resid = flux_bnd - flux_bnd_exact
    print(f"{flux_resid=}")

    for i in range(numeq, dim):
        for j in range(dim):
            assert (la.norm(flux_resid[i, j].get())) == 0.0

    assert 1==0
