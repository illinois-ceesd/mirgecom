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
import pyopencl as cl
import pytest
from pytools.obj_array import make_obj_array

import grudge.op as op

from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import cantera
from mirgecom.transport import (
    SimpleTransport,
    MixtureAveragedTransport
)
from mirgecom.fluid import make_conserved
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.initializers import (
    Vortex2D, Lump, Uniform
)
from mirgecom.discretization import create_discretization_collection
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from mirgecom.mechanisms import get_mechanism_input

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("mechname", ["sandiego"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 3, 5])
def test_pyrometheus_transport(ctx_factory, mechname, dim, order):
    """Test mixture-averaged transport properties."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    ones = dcoll.zeros(actx) + 1.0
    zeros = dcoll.zeros(actx)

    # Pyrometheus initialization
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    tempin = 1500.0
    pressin = cantera.one_atm
    print(f"Testing (t,P) = ({tempin}, {pressin})")

    # Transport data initilization
    transport_model = MixtureAveragedTransport(pyro_obj)
    eos = PyrometheusMixture(pyro_obj, temperature_guess=tempin)
    gas_model = GasModel(eos=eos, transport=transport_model)

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

    for tempin in ([300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0]):

        cantera_soln.TPX = tempin, pressin, x
        cantera_soln.equilibrate("TP")
        can_t, can_rho, can_y = cantera_soln.TDY
        can_p = cantera_soln.P

        tin = can_t * ones
        rhoin = can_rho * ones
        yin = can_y * ones

        # Cantera transport
        mu_ct = cantera_soln.viscosity
        kappa_ct = cantera_soln.thermal_conductivity
        diff_ct = cantera_soln.mix_diff_coeffs

        cv = make_conserved(dim=2, mass=rhoin,
                            momentum=make_obj_array([zeros, zeros]),
                            energy=rhoin*gas_model.eos.get_internal_energy(tin, yin),
                            species_mass=can_rho * can_y * ones)

        fluid_state = make_fluid_state(cv, gas_model, tin)

        # Pyrometheus transport
        mu = fluid_state.tv.viscosity
        kappa = fluid_state.tv.thermal_conductivity
        diff = fluid_state.tv.species_diffusivity

        def inf_norm(x):
            return actx.to_numpy(op.norm(dcoll, x, np.inf))

        # Making sure both pressure and temperature are correct
        err_p = np.abs(inf_norm(fluid_state.dv.pressure) - can_p)
        assert err_p < 1.0e-10

        err_t = np.abs(inf_norm(fluid_state.dv.temperature) - can_t)
        assert err_t < 2.0e-12

        # Viscosity
        err_mu = np.abs(inf_norm(mu) - mu_ct)
        assert err_mu < 1.0e-12

        # Thermal conductivity
        err_kappa = np.abs(inf_norm(kappa) - kappa_ct)
        assert err_kappa < 1.0e-12

        # Species diffusivities
        for i in range(nspecies):
            err_diff = np.abs(inf_norm(diff[i]) - diff_ct[i])
            assert err_diff < 1.0e-12


@pytest.mark.parametrize("mechname", ["air_3sp", "uiuc_7sp", "sandiego",
                                      "uiuc_8sp_phenol", "uiuc_4sp_oxidation"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 3, 5])
def test_mixture_dependent_properties(ctx_factory, mechname, dim, order):
    """Test MixtureEOS functionality."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    ones = dcoll.zeros(actx) + 1.0
    zeros = dcoll.zeros(actx)

    # Pyrometheus initialization
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PrometheusMixture::NumSpecies = {nspecies}")

    pressin = 101325.0

    # Transport data initilization
    transport_model = SimpleTransport(viscosity=1e-3, thermal_conductivity=0.0,
                                      species_diffusivity=np.zeros(nspecies,))

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    # first check each species individually for a fixed temperature

    tempin = 600.0
    eos = PyrometheusMixture(pyro_obj, temperature_guess=tempin)
    gas_model = GasModel(eos=eos, transport=transport_model)
    for i in range(nspecies):
        x = np.zeros(nspecies,)
        x[i] = 1.0

        cantera_soln.TPX = tempin, pressin, x
        can_t, can_rho, can_y = cantera_soln.TDY

        tin = can_t * ones
        yin = can_y * ones
        rhoin = can_rho * ones

        # First, check density
        mass = eos.get_density(pressin*ones, tempin*ones, yin)
        rel_err_t = (inf_norm(mass) - can_rho)/can_rho
        abs_err_t = (inf_norm(mass) - can_rho)
        assert rel_err_t < 1.0e-14
        assert abs_err_t < 1.0e-10

        # then proceed to conserved variables
        cv = make_conserved(dim=2, mass=rhoin,
                            momentum=make_obj_array([zeros, zeros]),
                            energy=rhoin*gas_model.eos.get_internal_energy(tin, yin),
                            species_mass=can_rho * can_y * ones)

        fluid_state = make_fluid_state(cv, gas_model, tin)

        # check the state built
        rel_err_t = (inf_norm(fluid_state.dv.temperature) - can_t)/can_t
        abs_err_t = (inf_norm(fluid_state.dv.temperature) - can_t)
        assert rel_err_t < 1.0e-14
        assert abs_err_t < 1.0e-10

        # heat capacity per mass unit
        can_cv = cantera_soln.cv
        heat_cap_cv = eos.heat_capacity_cv(cv, tin)
        rel_err_cv = (inf_norm(heat_cap_cv) - np.abs(can_cv))/np.abs(can_cv)
        abs_err_cv = (inf_norm(heat_cap_cv) - np.abs(can_cv))
        assert rel_err_cv < 1.0e-12
        assert abs_err_cv < 1.0e-6

        can_cp = cantera_soln.cp
        heat_cap_cp = eos.heat_capacity_cp(cv, tin)
        rel_err_cp = (inf_norm(heat_cap_cp) - np.abs(can_cp))/np.abs(can_cp)
        abs_err_cp = (inf_norm(heat_cap_cp) - np.abs(can_cp))
        assert rel_err_cp < 1.0e-12
        assert abs_err_cp < 1.0e-6

        # internal energy and energy per mass unit
        can_e = cantera_soln.int_energy_mass
        int_energy = eos.get_internal_energy(tin, yin)
        rel_err_e = (inf_norm(int_energy) - np.abs(can_e))/np.abs(can_e)
        abs_err_e = (inf_norm(int_energy) - np.abs(can_e))
        assert rel_err_e < 1.0e-12
        assert abs_err_e < 1.0e-6

        can_h = cantera_soln.enthalpy_mass
        enthalpy = fluid_state.dv.species_enthalpies[i]
        rel_err_h = (inf_norm(enthalpy) - np.abs(can_h))/np.abs(can_h)
        abs_err_h = (inf_norm(enthalpy) - np.abs(can_h))
        assert rel_err_h < 1.0e-12
        assert abs_err_h < 1.0e-6

    # ~~~ Now check an actual mixture at different temperatures

    x = 1.0/nspecies*np.ones(nspecies,)

    for tempin in ([300.0, 600.0, 900.0, 1200.0]):

        print(f"Testing (t,P) = ({tempin}, {pressin})")

        eos = PyrometheusMixture(pyro_obj, temperature_guess=tempin)
        gas_model = GasModel(eos=eos, transport=transport_model)

        cantera_soln.TPX = tempin, pressin, x
        can_t, can_rho, can_y = cantera_soln.TDY

        tin = can_t * ones
        rhoin = can_rho * ones
        yin = can_y * ones

        mass = eos.get_density(pressin*ones, tempin*ones, yin)
        rel_err_t = (inf_norm(mass) - can_rho)/can_rho
        abs_err_t = (inf_norm(mass) - can_rho)
        assert rel_err_t < 1.0e-14
        assert abs_err_t < 1.0e-10

        cv = make_conserved(dim=2, mass=rhoin,
                            momentum=make_obj_array([zeros, zeros]),
                            energy=rhoin*gas_model.eos.get_internal_energy(tin, yin),
                            species_mass=can_rho * can_y * ones)

        fluid_state = make_fluid_state(cv, gas_model, tin)

        rel_err_t = (inf_norm(fluid_state.dv.temperature) - can_t)/can_t
        abs_err_t = (inf_norm(fluid_state.dv.temperature) - can_t)
        assert rel_err_t < 1.0e-14
        assert abs_err_t < 1.0e-10

        can_cv = cantera_soln.cv
        heat_cap_cv = eos.heat_capacity_cv(cv, tin)
        rel_err_cv = (inf_norm(heat_cap_cv) - np.abs(can_cv))/np.abs(can_cv)
        abs_err_cv = (inf_norm(heat_cap_cv) - np.abs(can_cv))
        assert rel_err_cv < 1.0e-12
        assert abs_err_cv < 1.0e-6

        can_cp = cantera_soln.cp
        heat_cap_cp = eos.heat_capacity_cp(cv, tin)
        rel_err_cp = (inf_norm(heat_cap_cp) - np.abs(can_cp))/np.abs(can_cp)
        abs_err_cp = (inf_norm(heat_cap_cp) - np.abs(can_cp))
        assert rel_err_cp < 1.0e-12
        assert abs_err_cp < 1.0e-6

        can_e = cantera_soln.int_energy_mass
        int_energy = eos.get_internal_energy(tin, yin)
        rel_err_e = (inf_norm(int_energy) - np.abs(can_e))/np.abs(can_e)
        abs_err_e = (inf_norm(int_energy) - np.abs(can_e))
        assert rel_err_e < 1.0e-12
        assert abs_err_e < 1.0e-6

        can_h = cantera_soln.enthalpy_mass
        enthalpy = eos.get_enthalpy(tin, yin)
        rel_err_h = (inf_norm(enthalpy) - np.abs(can_h))/np.abs(can_h)
        abs_err_h = (inf_norm(enthalpy) - np.abs(can_h))
        assert rel_err_h < 1.0e-12
        assert abs_err_h < 1.0e-6


@pytest.mark.parametrize(("mechname", "rate_tol"),
                         [("uiuc_7sp", 1e-12),
                          ("sandiego", 1e-8)])
@pytest.mark.parametrize("y0", [0, 1])
def test_pyrometheus_mechanisms(ctx_factory, mechname, rate_tol, y0):
    """Test known pyrometheus mechanisms.

    This test reproduces a pyrometheus-native test in the MIRGE context.

    Tests that the Pyrometheus mechanism code  gets the same thermo properties as the
    corresponding mechanism in Cantera.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    sol = cantera.Solution(name="gas", yaml=mech_input)
    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    prometheus_mechanism = make_pyrometheus_mechanism_class(sol)(actx.np)

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
        cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
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
        prom_t = prometheus_mechanism.get_temperature(prom_e, tin, yin)
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

        def inf_norm(x):
            return actx.to_numpy(op.norm(dcoll, x, np.inf))

        assert inf_norm((prom_c - can_c)) < 1e-14
        assert inf_norm((prom_t - can_t) / can_t) < 1e-14
        assert inf_norm((prom_rho - can_rho) / can_rho) < 1e-14
        assert inf_norm((prom_p - can_p) / can_p) < 1e-14
        assert inf_norm((prom_e - can_e) / can_e) < 1e-6
        assert inf_norm((prom_k - can_k) / can_k) < 1e-10

        # Pyro chem test comparisons
        for i, rate in enumerate(can_r):
            assert inf_norm(prom_r[i] - rate) < rate_tol
        for i, rate in enumerate(can_omega):
            assert inf_norm(prom_omega[i] - rate) < rate_tol


@pytest.mark.parametrize("mechname", ["uiuc_7sp", "sandiego"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("y0", [0, 1])
@pytest.mark.parametrize("vel", [0.0, 1.0])
def test_pyrometheus_eos(ctx_factory, mechname, dim, y0, vel):
    """Test PyrometheusMixture EOS for all available mechanisms.

    Tests that the PyrometheusMixture EOS gets the same thermo properties
    (p, T, e) as the Pyrometheus-native mechanism code.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    sol = cantera.Solution(name="gas", yaml=mech_input)
    from mirgecom.thermochemistry import (
        make_pyrometheus_mechanism_class,
        get_pyrometheus_wrapper_class_from_cantera
    )
    prometheus_mechanism = make_pyrometheus_mechanism_class(sol)(actx.np)

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

    for fac in range(1, 7):
        tempin = fac * temp0
        pressin = fac * press0

        print(f"Testing {mechname}(t,P) = ({tempin}, {pressin})")

        ones = dcoll.zeros(actx) + 1.0
        tin = tempin * ones
        pin = pressin * ones
        yin = y0s * ones
        tguess = 0*ones + 300.0

        pyro_rho = prometheus_mechanism.get_density(pin, tin, yin)
        pyro_e = prometheus_mechanism.get_mixture_internal_energy_mass(tin, yin)
        pyro_t = prometheus_mechanism.get_temperature(pyro_e, tguess, yin)
        pyro_p = prometheus_mechanism.get_pressure(pyro_rho, pyro_t, yin)

        print(f"prom(rho, y, p, t, e) = ({pyro_rho}, {y0s}, "
              f"{pyro_p}, {pyro_t}, {pyro_e})")

        eos = PyrometheusMixture(prometheus_mechanism)
        gas_model = GasModel(eos=eos)
        initializer = Uniform(
            dim=dim, pressure=pyro_p, temperature=pyro_t,
            species_mass_fractions=y0s, velocity=velocity)

        cv = initializer(eos=eos, t=0, x_vec=nodes)
        fluid_state = make_fluid_state(cv, gas_model, temperature_seed=tguess)
        p = fluid_state.pressure
        temperature = fluid_state.temperature
        internal_energy = eos.get_internal_energy(temperature=tin,
                                                  species_mass_fractions=yin)
        y = cv.species_mass_fractions
        rho = cv.mass

        print(f"pyro_y = {y}")
        print(f"pyro_eos.p = {p}")
        print(f"pyro_eos.temp = {temperature}")
        print(f"pyro_eos.e = {internal_energy}")

        def inf_norm(x):
            return actx.to_numpy(op.norm(dcoll, x, np.inf))

        tol = 1e-14
        assert inf_norm((cv.mass - pyro_rho) / pyro_rho) < tol
        assert inf_norm((temperature - pyro_t) / pyro_t) < tol
        assert inf_norm((internal_energy - pyro_e) / pyro_e) < tol
        assert inf_norm((p - pyro_p) / pyro_p) < tol

        # Test the concetrations zero level
        y = -1.0*y
        print(f"{y=}")
        conc = prometheus_mechanism.get_concentrations(rho, y)
        print(f"{conc=}")
        for spec in range(nspecies):
            assert max(conc[spec]).all() >= 0

        zlev = 1e-3
        test_mech = \
            get_pyrometheus_wrapper_class_from_cantera(sol, zero_level=zlev)(actx.np)

        y = 0*y + zlev
        print(f"{y=}")
        conc = test_mech.get_concentrations(rho, y)
        print(f"{conc=}")
        for spec in range(nspecies):
            assert max(conc[spec]).all() == 0


@pytest.mark.parametrize(("mechname", "rate_tol"),
                         [("uiuc_7sp", 2e-12),
                          ("sandiego", 1e-8)])
@pytest.mark.parametrize("y0", [0, 1])
def test_pyrometheus_kinetics(ctx_factory, mechname, rate_tol, y0):
    """Test known pyrometheus reaction mechanisms.

    This test reproduces a pyrometheus-native test in the MIRGE context.

    Tests that the Pyrometheus mechanism code gets the same chemical properties
    and reaction rates as the corresponding mechanism in Cantera. The reactions
    are integrated in time and verified against a homogeneous reactor in
    Cantera.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    ones = dcoll.zeros(actx) + 1.0

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    pyro_obj = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)

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
        def inf_norm(x):
            return actx.to_numpy(op.norm(dcoll, x, np.inf))

        print(f"can_r = {can_r}")
        print(f"pyro_r = {pyro_r}")
        for i, can in enumerate(can_r):
            min_r = np.abs(can)
            if min_r > 0:
                assert inf_norm((pyro_r[i] - can) / can) < rate_tol
            else:
                assert inf_norm(pyro_r[i]) < rate_tol

        print(f"can_omega = {can_omega}")
        print(f"pyro_omega = {pyro_omega}")
        for i, omega in enumerate(can_omega):
            omin = np.abs(omega)
            if omin > 1e-12:
                assert inf_norm((pyro_omega[i] - omega) / omega) < 1e-8
            else:
                assert inf_norm(pyro_omega[i]) < 1e-12


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_idealsingle_lump(ctx_factory, dim):
    """Test IdealSingle EOS with mass lump.

    Tests that the IdealSingleGas EOS returns the correct (uniform) pressure for the
    Lump solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    velocity[0] = 1
    lump = Lump(dim=dim, center=center, velocity=velocity)
    eos = IdealSingleGas()
    cv = lump(nodes)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    p = eos.pressure(cv)
    exp_p = 1.0
    errmax = inf_norm(p - exp_p)

    exp_ke = 0.5 * cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = inf_norm(ke - exp_ke)

    te = eos.total_energy(cv, p)
    terr = inf_norm(te - cv.energy)

    logger.info(f"lump_soln = {cv}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15


def test_idealsingle_vortex(ctx_factory):
    r"""Test EOS with isentropic vortex.

    Tests that the IdealSingleGas EOS returns the correct pressure (p) for the
    Vortex2D solution field (i.e. $p = \rho^{\gamma}$).
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    eos = IdealSingleGas()
    # Init soln with Vortex
    vortex = Vortex2D()
    cv = vortex(nodes)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    gamma = eos.gamma()
    p = eos.pressure(cv)
    exp_p = cv.mass ** gamma
    errmax = inf_norm(p - exp_p)

    exp_ke = 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = inf_norm(ke - exp_ke)

    te = eos.total_energy(cv, p)
    terr = inf_norm(te - cv.energy)

    logger.info(f"vortex_soln = {cv}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15
