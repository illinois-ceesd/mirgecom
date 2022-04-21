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

from pytools.obj_array import make_obj_array
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.symbolic.primitives import TracePair
from mirgecom.fluid import make_conserved
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from mirgecom.inviscid import inviscid_flux

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("nspecies", [0, 1, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("norm_dir", [1, -1])
@pytest.mark.parametrize("vel_mag", [0, 1, -1])
def test_lfr_flux(actx_factory, nspecies, dim, norm_dir, vel_mag):
    """Check inviscid flux against exact expected result.

    Directly check hll flux routine,
    :func:`mirgecom.flux.hll`,
    against the exact expected result. This test is designed to fail if the flux
    routine is broken.

    The expected inviscid flux is:
      F(q) = <rhoV, (E+p)V, rho(V.x.V) + pI, rhoY V>
    """
    tolerance = 1e-12
    actx = actx_factory()

    gamma = 1.4
    eos = IdealSingleGas(gamma=gamma)
    gas_model = GasModel(eos=eos)
    # interior state
    p0 = 15.0
    rho0 = 2.0
    vel0 = np.zeros(shape=dim, )
    for i in range(dim):
        vel0[i] = 3.0*vel_mag + (i+1)
    energy0 = p0/(gamma-1) + 0.5*rho0*np.dot(vel0, vel0)
    y0 = np.zeros(shape=nspecies, )
    for i in range(nspecies):
        y0[i] = (i+1)
    c0 = np.sqrt(gamma*p0/rho0)
    # exterior state
    p1 = 20.0
    rho1 = 4.0
    vel1 = np.zeros(shape=dim, )
    for i in range(dim):
        vel1[i] = 1.0*vel_mag + (i+1)
    energy1 = p1/(gamma-1) + 0.5*rho1*np.dot(vel1, vel1)
    y1 = np.zeros(shape=nspecies, )
    for i in range(nspecies):
        y1[i] = 2*(i+1)
    c1 = np.sqrt(gamma*p1/rho1)

    rho0_dof_array = DOFArray(actx, data=(actx.from_numpy(np.array(rho0)),))

    mass_int = 1.*rho0_dof_array
    vel_int = make_obj_array([vel0[idir] for idir in range(dim)])
    mom_int = mass_int*vel_int
    energy_int = p0/0.4 + 0.5 * np.dot(mom_int, mom_int)/mass_int
    species_mass_int = mass_int*make_obj_array([y0[idir]
                                                for idir in range(nspecies)])
    cv_int = make_conserved(dim, mass=mass_int, energy=energy_int,
                               momentum=mom_int, species_mass=species_mass_int)

    fluid_state_int = make_fluid_state(cv=cv_int, gas_model=gas_model)
    flux_int = inviscid_flux(fluid_state_int)

    mass_ext = DOFArray(actx, data=(actx.from_numpy(np.array(rho1)), ))
    vel_ext = make_obj_array([vel1[idir] for idir in range(dim)])
    mom_ext = mass_ext*vel_ext
    energy_ext = p1/0.4 + 0.5 * np.dot(mom_ext, mom_ext)/mass_ext
    species_mass_ext = mass_ext*make_obj_array([y1[idir]
                                                for idir in range(nspecies)])
    cv_ext = make_conserved(dim, mass=mass_ext, energy=energy_ext,
                            momentum=mom_ext, species_mass=species_mass_ext)

    fluid_state_ext = make_fluid_state(cv=cv_ext, gas_model=gas_model)
    flux_ext = inviscid_flux(fluid_state_ext)

    print(f"{cv_int=}")
    print(f"{flux_int=}")
    print(f"{cv_ext=}")
    print(f"{flux_ext=}")

    # interface normal
    normal = np.ones(shape=dim, )
    mag = np.linalg.norm(normal)
    normal = norm_dir*normal/mag

    state_pair = TracePair("vol", interior=fluid_state_int, exterior=fluid_state_ext)

    # code passes in fluxes in the direction of the surface normal,
    # so we will too
    from mirgecom.inviscid import inviscid_flux_rusanov
    flux_bnd = inviscid_flux_rusanov(state_pair, gas_model, normal)

    print(f"{normal=}")
    print(f"{flux_ext@normal=}")
    print(f"{flux_int@normal=}")

    # compute the exact flux in the interface normal direction, as calculated by lfr

    # wave speed
    lam = np.maximum(np.linalg.norm(vel0)+c0, np.linalg.norm(vel1)+c1)
    print(f"{lam=}")

    # compute the velocity in the direction of the surface normal
    vel0_norm = np.dot(vel0, normal)
    vel1_norm = np.dot(vel1, normal)

    mass_flux_exact = 0.5*(rho0*vel0_norm + rho1*vel1_norm
                           - lam*(rho1 - rho0))
    mom_flux_exact = np.zeros(shape=(dim, dim), )
    for i in range(dim):
        for j in range(dim):
            mom_flux_exact[i][j] = (rho0*vel0[i]*vel0[j]+(p0 if i == j else 0)
                                    + rho1*vel1[i]*vel1[j]+(p1 if i == j else 0))/2.
    mom_flux_norm_exact = np.zeros(shape=dim, )
    for i in range(dim):
        mom_flux_norm_exact[i] = (np.dot(mom_flux_exact[i], normal)
                                  - 0.5*lam*(rho1*vel1[i] - rho0*vel0[i]))
    mom_flux_norm_exact = make_obj_array([mom_flux_norm_exact[idim]
                                          for idim in range(dim)])
    energy_flux_exact = 0.5*(vel1_norm*(energy1+p1) + vel0_norm*(energy0+p0)
                             - lam*(energy1 - energy0))
    species_mass_flux_exact = 0.5*(vel0_norm*y0*rho0 + vel1_norm*y1*rho1
                                   - lam*(y1*rho1 - y0*rho0))
    species_mass_flux_exact = make_obj_array([species_mass_flux_exact[ispec]
                                              for ispec in range(nspecies)])

    exact_massflux = DOFArray(actx, data=(actx.from_numpy(mass_flux_exact),))
    exact_energyflux = DOFArray(actx, data=(actx.from_numpy(energy_flux_exact),))
    exact_momflux = make_obj_array(
        [DOFArray(actx, data=(actx.from_numpy(mom_flux_norm_exact[i]),))
         for i in range(dim)])
    exact_specflux = make_obj_array(
        [DOFArray(actx, data=(actx.from_numpy(species_mass_flux_exact[i]),))
         for i in range(nspecies)])

    flux_bnd_exact = make_conserved(dim, mass=exact_massflux,
                                    energy=exact_energyflux,
                                    momentum=exact_momflux,
                                    species_mass=exact_specflux)

    print(f"{flux_bnd=}")
    print(f"{flux_bnd_exact=}")

    flux_resid = flux_bnd - flux_bnd_exact
    print(f"{flux_resid=}")

    assert np.abs(actx.np.linalg.norm(flux_resid)) < tolerance


# velocities are tuned to exercise different wave configurations:
#    vel_mag = 0, rarefaction, zero velocity
#    vel_mag = 1, right traveling rarefaction
#    vel_mag = 2, right traveling shock
#    vel_mag = -1, left traveling rarefaction
#    vel_mag = -4, right traveling shock
@pytest.mark.parametrize("vel_mag", [0, 1, 2, -1, -4])
@pytest.mark.parametrize("nspecies", [0, 1, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("norm_dir", [1, -1])
def test_hll_flux(actx_factory, nspecies, dim, norm_dir, vel_mag):
    """Check inviscid flux against exact expected result.

    Directly check hll flux routine,
    :func:`mirgecom.flux.hll`,
    against the exact expected result. This test is designed to fail if the flux
    routine is broken.

    The expected inviscid flux is:
      F(q) = <rhoV, (E+p)V, rho(V.x.V) + pI, rhoY V>
    """
    tolerance = 1e-12
    actx = actx_factory()

    gamma = 1.4
    eos = IdealSingleGas(gamma=gamma)
    gas_model = GasModel(eos=eos)
    # interior state
    p0 = 15.0
    rho0 = 2.0
    vel0 = np.zeros(shape=dim, )
    for i in range(dim):
        vel0[i] = 3.0*vel_mag + (i+1)
    energy0 = p0/(gamma-1) + 0.5*rho0*np.dot(vel0, vel0)
    y0 = np.zeros(shape=nspecies, )
    for i in range(nspecies):
        y0[i] = (i+1)
    c0 = np.sqrt(gamma*p0/rho0)
    # exterior state
    p1 = 20.0
    rho1 = 4.0
    vel1 = np.zeros(shape=dim, )
    for i in range(dim):
        vel1[i] = 1.0*vel_mag + (i+1)
    energy1 = p1/(gamma-1) + 0.5*rho1*np.dot(vel1, vel1)
    y1 = np.zeros(shape=nspecies, )
    for i in range(nspecies):
        y1[i] = 2*(i+1)
    c1 = np.sqrt(gamma*p1/rho1)

    mass_int = DOFArray(actx, data=(actx.from_numpy(np.array(rho0)), ))
    vel_int = make_obj_array([vel0[idir] for idir in range(dim)])
    mom_int = mass_int*vel_int
    energy_int = p0/0.4 + 0.5 * np.dot(mom_int, mom_int)/mass_int
    species_mass_int = mass_int*make_obj_array([y0[idir]
                                                for idir in range(nspecies)])
    cv_int = make_conserved(dim, mass=mass_int, energy=energy_int,
                               momentum=mom_int, species_mass=species_mass_int)
    fluid_state_int = make_fluid_state(cv=cv_int, gas_model=gas_model)
    flux_int = inviscid_flux(fluid_state_int)

    mass_ext = DOFArray(actx, data=(actx.from_numpy(np.array(rho1)), ))
    vel_ext = make_obj_array([vel1[idir] for idir in range(dim)])
    mom_ext = mass_ext*vel_ext
    energy_ext = p1/0.4 + 0.5 * np.dot(mom_ext, mom_ext)/mass_ext
    species_mass_ext = mass_ext*make_obj_array([y1[idir]
                                                for idir in range(nspecies)])
    cv_ext = make_conserved(dim, mass=mass_ext, energy=energy_ext,
                            momentum=mom_ext, species_mass=species_mass_ext)
    fluid_state_ext = make_fluid_state(cv=cv_ext, gas_model=gas_model)
    flux_ext = inviscid_flux(fluid_state_ext)

    print(f"{cv_int=}")
    print(f"{flux_int=}")
    print(f"{cv_ext=}")
    print(f"{flux_ext=}")

    # interface normal
    normal = np.ones(shape=dim, )
    mag = np.linalg.norm(normal)
    normal = norm_dir*normal/mag

    state_pair = TracePair("vol", interior=fluid_state_int, exterior=fluid_state_ext)

    from mirgecom.inviscid import inviscid_flux_hll
    flux_bnd = inviscid_flux_hll(state_pair, gas_model, normal)

    print(f"{normal=}")
    print(f"{flux_ext@normal=}")
    print(f"{flux_int@normal=}")

    # compute the exact flux in the interface normal direction, as calculated by hll

    # compute the left and right wave_speeds
    u_int = np.dot(vel0, normal)
    u_ext = np.dot(vel1, normal)
    p_star = (0.5*(p0 + p1) + (1./8.)*(u_int - u_ext)
              * (rho0 + rho1) * (c0 + c1))
    print(f"{p_star=}")

    # the code checks that the pressure ratio is > 0, don't need to do that here
    q_int = 1.
    q_ext = 1.
    if p_star > p0:
        q_int = np.sqrt(1 + (gamma + 1)/(2*gamma)*(p_star/p0 - 1))
    if p_star > p1:
        q_ext = np.sqrt(1 + (gamma + 1)/(2*gamma)*(p_star/p1 - 1))
    s_int = u_int - c0*q_int
    s_ext = u_ext + c1*q_ext

    print(f"wave speeds {s_int=} {s_ext=}")

    # compute the velocity in the direction of the surface normal
    vel0_norm = np.dot(vel0, normal)
    vel1_norm = np.dot(vel1, normal)
    if s_ext <= 0.:
        print("exterior flux")
        # the flux from the exterior state
        print("s_int <= 0")
        mass_flux_exact = rho1*vel1_norm
        mom_flux_exact = np.zeros(shape=(dim, dim), )
        for i in range(dim):
            for j in range(dim):
                mom_flux_exact[i][j] = rho1*vel1[i]*vel1[j] + (p1 if i == j else 0)
        mom_flux_norm_exact = np.zeros(shape=dim, )
        for i in range(dim):
            mom_flux_norm_exact[i] = np.dot(mom_flux_exact[i], normal)
        mom_flux_norm_exact = make_obj_array([mom_flux_norm_exact[idim]
                                              for idim in range(dim)])
        energy_flux_exact = vel1_norm*(energy1+p1)
        species_mass_flux_exact = vel1_norm*y1*rho1
        species_mass_flux_exact = make_obj_array([species_mass_flux_exact[ispec]
                                                  for ispec in range(nspecies)])

    elif s_int >= 0.:
        print("interior flux")
        # the flux from the interior state
        mass_flux_exact = rho0*vel0_norm
        mom_flux_exact = np.zeros(shape=(dim, dim), )
        for i in range(dim):
            for j in range(dim):
                mom_flux_exact[i][j] = rho0*vel0[i]*vel0[j] + (p0 if i == j else 0)
        mom_flux_norm_exact = np.zeros(shape=dim, )
        for i in range(dim):
            mom_flux_norm_exact[i] = np.dot(mom_flux_exact[i], normal)
        mom_flux_norm_exact = make_obj_array([mom_flux_norm_exact[idim]
                                              for idim in range(dim)])
        energy_flux_exact = vel0_norm*(energy0+p0)
        species_mass_flux_exact = vel0_norm*y0*rho0
        species_mass_flux_exact = make_obj_array([species_mass_flux_exact[ispec]
                                                  for ispec in range(nspecies)])

    else:
        print("star flux")
        # the flux from the star state
        mass_flux_exact = (s_ext*rho0*vel0_norm - s_int*rho1*vel1_norm
                           + s_int*s_ext*(rho1 - rho0))/(s_ext - s_int)
        mom_flux_exact = np.zeros(shape=(dim, dim), )
        for i in range(dim):
            for j in range(dim):
                mom_flux_exact[i][j] = (s_ext*(rho0*vel0[i]*vel0[j]
                                               + (p0 if i == j else 0))
                                        - s_int*(rho1*vel1[i]*vel1[j]
                                                 + (p1 if i == j else 0)))
        mom_flux_norm_exact = np.zeros(shape=dim, )
        for i in range(dim):
            mom_flux_norm_exact[i] = ((np.dot(mom_flux_exact[i], normal)
                                      + s_int*s_ext*(rho1*vel1[i] - rho0*vel0[i]))
                                      / (s_ext - s_int))
        mom_flux_norm_exact = make_obj_array([mom_flux_norm_exact[idim]
                                              for idim in range(dim)])
        energy_flux_exact = (s_ext*vel0_norm*(energy0+p0)
                             - s_int*vel1_norm*(energy1+p1)
                             + s_int*s_ext*(energy1 - energy0))/(s_ext - s_int)
        species_mass_flux_exact = (s_ext*vel0_norm*rho0*y0
                                   - s_int*vel1_norm*rho1*y1
                                   + s_int*s_ext*(rho1*y1 - rho0*y0))/(s_ext - s_int)
        species_mass_flux_exact = make_obj_array([species_mass_flux_exact[ispec]
                                                  for ispec in range(nspecies)])

    exact_massflux = DOFArray(actx, data=(actx.from_numpy(mass_flux_exact),))
    exact_energyflux = DOFArray(actx, data=(actx.from_numpy(energy_flux_exact),))
    exact_momflux = make_obj_array(
        [DOFArray(actx, data=(actx.from_numpy(mom_flux_norm_exact[i]),))
         for i in range(dim)])
    exact_specflux = make_obj_array(
        [DOFArray(actx, data=(actx.from_numpy(species_mass_flux_exact[i]),))
         for i in range(nspecies)])

    flux_bnd_exact = make_conserved(dim, mass=exact_massflux,
                                    energy=exact_energyflux,
                                    momentum=exact_momflux,
                                    species_mass=exact_specflux)

    print(f"{flux_bnd=}")
    print(f"{flux_bnd_exact=}")

    flux_resid = flux_bnd - flux_bnd_exact
    print(f"{flux_resid=}")

    assert np.abs(actx.np.linalg.norm(flux_resid)) < tolerance
