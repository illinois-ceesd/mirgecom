"""Test the EOS interfaces."""

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
import logging
import numpy as np
import numpy.linalg as la  # noqa
import pytest

from arraycontext import (  # noqa
    thaw,
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
)

from pytools.obj_array import make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

import cantera
import pyrometheus as pyro
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.initializers import (
    Vortex2D, Lump,
    MixtureInitializer
)
from mirgecom.fluid import split_conserved
from mirgecom.mechanisms import get_mechanism_cti

from grudge.discretization import DiscretizationCollection
import grudge.op as op

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("mechname", "rate_tol"),
                         [("uiuc", 1e-12),
                          ("sanDiego", 1e-8)])
@pytest.mark.parametrize("y0", [0, 1])
def test_pyrometheus_mechanisms(actx_factory, mechname, rate_tol, y0):
    """Test known pyrometheus mechanisms.

    This test reproduces a pyrometheus-native test in the MIRGE context.

    Tests that the Pyrometheus mechanism code  gets the same thermo properties as the
    corresponding mechanism in Cantera.
    """
    actx = actx_factory()

    dim = 1
    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    # Pyrometheus initialization
    mech_cti = get_mechanism_cti(mechname)
    sol = cantera.Solution(phase_id="gas", source=mech_cti)
    prometheus_mechanism = pyro.get_thermochem_class(sol)(actx.np)

    nspecies = prometheus_mechanism.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    press0 = 101500.0
    temp0 = 300.0
    y0s = np.zeros(shape=(nspecies,))
    for i in range(nspecies-1):
        y0s[i] = y0 / (10.0 ** (i + 1))
    y0s[-1] = 1.0 - np.sum(y0s[:-1])

    for fac in range(1, 11):
        pressin = fac * press0
        tempin = fac * temp0

        print(f"Testing (t,P) = ({tempin}, {pressin})")
        cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
        cantera_soln.TPY = tempin, pressin, y0s
        cantera_soln.equilibrate("UV")
        can_t, can_rho, can_y = cantera_soln.TDY
        can_p = cantera_soln.P
        can_e = cantera_soln.int_energy_mass
        can_k = cantera_soln.forward_rate_constants
        can_c = cantera_soln.concentrations

        # Chemistry functions for testing pyro chem
        can_r = cantera_soln.net_rates_of_progress
        can_omega = cantera_soln.net_production_rates

        ones = dcoll.zeros(actx) + 1.0
        tin = can_t * ones
        pin = can_p * ones
        yin = make_obj_array([can_y[i] * ones for i in range(nspecies)])

        prom_rho = prometheus_mechanism.get_density(pin, tin, yin)
        prom_e = prometheus_mechanism.get_mixture_internal_energy_mass(tin, yin)
        prom_t = prometheus_mechanism.get_temperature(prom_e, tin, yin, True)
        prom_p = prometheus_mechanism.get_pressure(prom_rho, tin, yin)
        prom_c = prometheus_mechanism.get_concentrations(prom_rho, yin)
        prom_k = prometheus_mechanism.get_fwd_rate_coefficients(prom_t, prom_c)

        # Pyro chemistry functions
        prom_r = prometheus_mechanism.get_net_rates_of_progress(prom_t,
                                                                prom_c)
        prom_omega = prometheus_mechanism.get_net_production_rates(prom_rho,
                                                                   prom_t, yin)

        print(f"can(rho, y, p, t, e, k) = ({can_rho}, {can_y}, "
              f"{can_p}, {can_t}, {can_e}, {can_k})")
        print(f"prom(rho, y, p, t, e, k) = ({prom_rho}, {y0s}, "
              f"{prom_p}, {prom_t}, {prom_e}, {prom_k})")

        # For pyro chem testing
        print(f"can_r = {can_r}")
        print(f"prom_r = {prom_r}")
        print(f"can_omega = {can_omega}")
        print(f"prom_omega = {prom_omega}")

        assert op.norm(dcoll, (prom_c - can_c) / can_c, np.inf) < 1e-14
        assert op.norm(dcoll, (prom_t - can_t) / can_t, np.inf) < 1e-14
        assert op.norm(dcoll, (prom_rho - can_rho) / can_rho, np.inf) < 1e-14
        assert op.norm(dcoll, (prom_p - can_p) / can_p, np.inf) < 1e-14
        assert op.norm(dcoll, (prom_e - can_e) / can_e, np.inf) < 1e-6
        assert op.norm(dcoll, (prom_k - can_k) / can_k, np.inf) < 1e-10

        # Pyro chem test comparisons
        for i, rate in enumerate(can_r):
            assert op.norm(dcoll, (prom_r[i] - rate), np.inf) < rate_tol
        for i, rate in enumerate(can_omega):
            assert op.norm(dcoll, (prom_omega[i] - rate), np.inf) < rate_tol


@pytest.mark.parametrize("mechname", ["uiuc", "sanDiego"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("y0", [0, 1])
@pytest.mark.parametrize("vel", [0.0, 1.0])
def test_pyrometheus_eos(actx_factory, mechname, dim, y0, vel):
    """Test PyrometheusMixture EOS for all available mechanisms.

    Tests that the PyrometheusMixture EOS gets the same thermo properties
    (p, T, e) as the Pyrometheus-native mechanism code.
    """
    actx = actx_factory()

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)

    # Pyrometheus initialization
    mech_cti = get_mechanism_cti(mechname)
    sol = cantera.Solution(phase_id="gas", source=mech_cti)
    prometheus_mechanism = pyro.get_thermochem_class(sol)(actx.np)

    nspecies = prometheus_mechanism.num_species
    print(f"PrometheusMixture::Mechanism = {mechname}")
    print(f"PrometheusMixture::NumSpecies = {nspecies}")

    press0 = 101500.0
    temp0 = 300.0
    y0s = np.zeros(shape=(nspecies,))
    for i in range(1, nspecies):
        y0s[i] = y0 / (10.0 ** i)
    y0s[0] = 1.0 - np.sum(y0s[1:])
    velocity = vel * np.ones(shape=(dim,))

    for fac in range(1, 11):
        tempin = fac * temp0
        pressin = fac * press0

        print(f"Testing {mechname}(t,P) = ({tempin}, {pressin})")

        ones = dcoll.zeros(actx) + 1.0
        tin = tempin * ones
        pin = pressin * ones
        yin = y0s * ones
        tguess = 300.0

        pyro_rho = prometheus_mechanism.get_density(pin, tin, yin)
        pyro_e = prometheus_mechanism.get_mixture_internal_energy_mass(tin, yin)
        pyro_t = prometheus_mechanism.get_temperature(pyro_e, tguess, yin, True)
        pyro_p = prometheus_mechanism.get_pressure(pyro_rho, pyro_t, yin)

        print(f"prom(rho, y, p, t, e) = ({pyro_rho}, {y0s}, "
              f"{pyro_p}, {pyro_t}, {pyro_e})")

        eos = PyrometheusMixture(prometheus_mechanism)
        initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                         pressure=pyro_p, temperature=pyro_t,
                                         massfractions=y0s, velocity=velocity)

        q = initializer(eos=eos, t=0, x_vec=nodes)
        cv = split_conserved(dim, q)

        p = eos.pressure(cv)
        temperature = eos.temperature(cv)
        internal_energy = eos.get_internal_energy(tin, yin)
        y = eos.species_fractions(cv)

        print(f"pyro_y = {y}")
        print(f"pyro_eos.p = {p}")
        print(f"pyro_eos.temp = {temperature}")
        print(f"pyro_eos.e = {internal_energy}")

        tol = 1e-14
        assert op.norm(dcoll, (cv.mass - pyro_rho) / pyro_rho, np.inf) < tol
        assert op.norm(dcoll, (temperature - pyro_t) / pyro_t, np.inf) < tol
        assert op.norm(dcoll, (internal_energy - pyro_e) / pyro_e, np.inf) < tol
        assert op.norm(dcoll, (p - pyro_p) / pyro_p, np.inf) < tol


@pytest.mark.parametrize(("mechname", "rate_tol"),
                         [("uiuc", 1e-12),
                          ("sanDiego", 1e-8)])
@pytest.mark.parametrize("y0", [0, 1])
def test_pyrometheus_kinetics(actx_factory, mechname, rate_tol, y0):
    """Test known pyrometheus reaction mechanisms.

    This test reproduces a pyrometheus-native test in the MIRGE context.

    Tests that the Pyrometheus mechanism code gets the same chemical properties
    and reaction rates as the corresponding mechanism in Cantera. The reactions
    are integrated in time and verified against a homogeneous reactor in
    Cantera.
    """
    actx = actx_factory()

    dim = 1
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    ones = dcoll.zeros(actx) + 1.0

    # Pyrometheus initialization
    mech_cti = get_mechanism_cti(mechname)
    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    pyro_obj = pyro.get_thermochem_class(cantera_soln)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PrometheusMixture::NumSpecies = {nspecies}")

    tempin = 1500.0
    pressin = cantera.one_atm
    print(f"Testing (t,P) = ({tempin}, {pressin})")

    # Homogeneous reactor to get test data
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 0.5
    i_fu = cantera_soln.species_index("H2")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    x = np.zeros(shape=(nspecies,))
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio

    cantera_soln.TPX = tempin, pressin, x
    #    cantera_soln.equilibrate("UV")
    can_t, can_rho, can_y = cantera_soln.TDY
    #    can_p = cantera_soln.P

    reactor = cantera.IdealGasConstPressureReactor(cantera_soln)
    sim = cantera.ReactorNet([reactor])
    time = 0.0
    for _ in range(50):
        time += 1.0e-6
        sim.advance(time)

        # Cantera kinetics
        can_r = reactor.kinetics.net_rates_of_progress
        can_omega = reactor.kinetics.net_production_rates

        # Get state from Cantera
        can_t = reactor.T
        can_rho = reactor.density
        can_y = reactor.Y
        print(f"can_y = {can_y}")

        tin = can_t * ones
        rhoin = can_rho * ones
        yin = can_y * ones

        # Prometheus kinetics
        pyro_c = pyro_obj.get_concentrations(rhoin, yin)
        print(f"pyro_conc = {pyro_c}")

        pyro_r = pyro_obj.get_net_rates_of_progress(tin, pyro_c)
        pyro_omega = pyro_obj.get_net_production_rates(rhoin, tin, yin)

        # Print
        print(f"can_r = {can_r}")
        print(f"pyro_r = {pyro_r}")
        abs_diff = op.norm(dcoll, pyro_r - can_r, np.inf)
        if abs_diff > 1e-14:
            min_r = (np.abs(can_r)).min()
            if min_r > 0:
                assert op.norm(dcoll, (pyro_r - can_r) / can_r, np.inf) < rate_tol
            else:
                assert op.norm(dcoll, pyro_r, np.inf) < rate_tol

        print(f"can_omega = {can_omega}")
        print(f"pyro_omega = {pyro_omega}")
        for i, omega in enumerate(can_omega):
            omin = np.abs(omega).min()
            if omin > 1e-12:
                assert op.norm(dcoll, (pyro_omega[i] - omega) / omega, np.inf) < 1e-8
            else:
                assert op.norm(dcoll, pyro_omega[i], np.inf) < 1e-12


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_idealsingle_lump(actx_factory, dim):
    """Test IdealSingle EOS with mass lump.

    Tests that the IdealSingleGas EOS returns the correct (uniform) pressure for the
    Lump solution field.
    """
    actx = actx_factory()

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    velocity[0] = 1
    lump = Lump(dim=dim, center=center, velocity=velocity)
    eos = IdealSingleGas()
    lump_soln = lump(nodes)

    cv = split_conserved(dim, lump_soln)
    p = eos.pressure(cv)
    exp_p = 1.0
    errmax = op.norm(dcoll, p - exp_p, np.inf)

    exp_ke = 0.5 * cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = op.norm(dcoll, ke - exp_ke, np.inf)

    te = eos.total_energy(cv, p)
    terr = op.norm(dcoll, te - cv.energy, np.inf)

    logger.info(f"lump_soln = {lump_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15


def test_idealsingle_vortex(actx_factory):
    r"""Test EOS with isentropic vortex.

    Tests that the IdealSingleGas EOS returns the correct pressure (p) for the
    Vortex2D solution field (i.e. $p = \rho^{\gamma}$).
    """
    actx = actx_factory()

    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    nodes = thaw(dcoll.nodes(), actx)
    eos = IdealSingleGas()
    # Init soln with Vortex
    vortex = Vortex2D()
    vortex_soln = vortex(nodes)
    cv = split_conserved(dim, vortex_soln)
    gamma = eos.gamma()
    p = eos.pressure(cv)
    exp_p = cv.mass ** gamma
    errmax = op.norm(dcoll, p - exp_p, np.inf)

    exp_ke = 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = op.norm(dcoll, ke - exp_ke, np.inf)

    te = eos.total_energy(cv, p)
    terr = op.norm(dcoll, te - cv.energy, np.inf)

    logger.info(f"vortex_soln = {vortex_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15
