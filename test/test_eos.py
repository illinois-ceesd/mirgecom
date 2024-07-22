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
import cantera
from pytools.obj_array import make_obj_array

from grudge import op

from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from meshmode.mesh.generation import generate_regular_rect_mesh
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from mirgecom.fluid import make_conserved
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.initializers import Vortex2D, Lump, Uniform
from mirgecom.discretization import create_discretization_collection
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("mechname", ["air_3sp", "uiuc_7sp", "sandiego",
                                      "uiuc_8sp_phenol", "uiuc_4sp_oxidation"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("pressure", [25000.0, 101325.0])
def test_mixture_dependent_properties(ctx_factory, mechname, dim, pressure):
    """Test MixtureEOS functionality."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 1

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    ones = dcoll.zeros(actx) + 1.0
    zeros = dcoll.zeros(actx)

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    # ~~~ First check each species individually for a fixed temperature

    tempin = 600.0
    eos = PyrometheusMixture(pyro_obj, temperature_guess=tempin)
    gas_model = GasModel(eos=eos)
    for i in range(nspecies):
        x = np.zeros(nspecies,)
        x[i] = 1.0

        cantera_soln.TPX = tempin, pressure, x
        can_t, can_rho, can_y = cantera_soln.TDY
        can_x = cantera_soln.X
        can_p = cantera_soln.P

        pin = can_p * ones
        tin = can_t * ones
        xin = can_x * ones
        rhoin = can_rho * ones

        yin = eos.mass_fractions_from_mole_fractions(xin)

        # First, check density
        mass = eos.get_density(pin, tin, yin)
        abs_err_m = inf_norm(mass - can_rho)
        assert abs_err_m/can_rho < 1.0e-14
        assert abs_err_m < 1.0e-10

        # then proceed to conserved variables
        cv = make_conserved(dim=2, mass=rhoin,
                            momentum=make_obj_array([zeros, zeros]),
                            energy=rhoin*gas_model.eos.get_internal_energy(tin, yin),
                            species_mass=rhoin*yin)

        fluid_state = make_fluid_state(cv, gas_model, tin)

        # check the state built
        abs_err_t = inf_norm(fluid_state.dv.temperature - can_t)
        assert abs_err_t/can_t < 1.0e-14
        assert abs_err_t < 1.0e-10

        abs_err_p = inf_norm(fluid_state.dv.pressure - can_p)
        assert abs_err_p/can_p < 1.0e-14
        assert abs_err_p < 1.0e-9

        # heat capacity per mass unit
        can_cv = cantera_soln.cv
        heat_cap_cv = eos.heat_capacity_cv(cv, tin)
        abs_err_cv = inf_norm(heat_cap_cv - can_cv)
        assert abs_err_cv/np.abs(can_cv) < 1.0e-12
        assert abs_err_cv < 1.0e-6

        can_cp = cantera_soln.cp
        heat_cap_cp = eos.heat_capacity_cp(cv, tin)
        abs_err_cp = inf_norm(heat_cap_cp - can_cp)
        assert abs_err_cp/np.abs(can_cp) < 1.0e-12
        assert abs_err_cp < 1.0e-6

        can_gamma = cantera_soln.cp/cantera_soln.cv
        gamma = eos.gamma(cv, tin)
        abs_err_gamma = inf_norm(gamma - can_gamma)
        assert abs_err_gamma/np.abs(can_gamma) < 1.0e-12
        assert abs_err_gamma < 1.0e-6

        # internal energy and energy per mass unit
        can_e = cantera_soln.int_energy_mass
        int_energy = eos.get_internal_energy(tin, yin)
        abs_err_e = inf_norm(int_energy - can_e)
        assert abs_err_e/np.abs(can_e) < 1.0e-12
        assert abs_err_e < 1.0e-6

        # enthalpy
        can_h = cantera_soln.enthalpy_mass
        enthalpy = fluid_state.dv.species_enthalpies[i]  # pylint: disable=no-member
        abs_err_h = inf_norm(enthalpy - can_h)
        assert abs_err_h/np.abs(can_h) < 1.0e-12
        assert abs_err_h < 1.0e-6

    # ~~~ Now check an actual mixture at different temperatures

    x = 1.0/nspecies*np.ones(nspecies,)

    for tempin in ([300.0, 600.0, 900.0, 1200.0]):

        print(f"Testing (t,P) = ({tempin}, {pressure})")

        cantera_soln.TPX = tempin, pressure, x
        can_t, can_rho, can_y = cantera_soln.TDY
        can_x = cantera_soln.X
        can_p = cantera_soln.P

        pin = can_p * ones
        tin = can_t * ones
        xin = can_x * ones
        rhoin = can_rho * ones

        yin = eos.mass_fractions_from_mole_fractions(xin)
        for i in range(nspecies):
            abs_err_y = inf_norm(yin[i] - can_y[i])
            assert abs_err_y < 1.0e-14

        xin_p = eos.mole_fractions_from_mass_fractions(yin)
        for i in range(nspecies):
            abs_err_x = inf_norm(xin_p[i] - can_x[i])
            assert abs_err_x < 1.0e-14

        mass = eos.get_density(pin, tin, yin)
        abs_err_m = inf_norm(mass - can_rho)
        assert abs_err_m/can_rho < 1.0e-14
        assert abs_err_m < 1.0e-10

        cv = make_conserved(dim=2, mass=rhoin,
                            momentum=make_obj_array([zeros, zeros]),
                            energy=rhoin*gas_model.eos.get_internal_energy(tin, yin),
                            species_mass=rhoin*yin)

        fluid_state = make_fluid_state(cv, gas_model, tin)

        abs_err_t = inf_norm(fluid_state.dv.temperature - can_t)
        assert abs_err_t/can_t < 1.0e-14
        assert abs_err_t < 1.0e-10

        abs_err_p = inf_norm(fluid_state.dv.pressure - can_p)
        assert abs_err_p/can_p < 1.0e-14
        assert abs_err_p < 1.0e-9

        can_cv = cantera_soln.cv
        heat_cap_cv = eos.heat_capacity_cv(cv, tin)
        abs_err_cv = inf_norm(heat_cap_cv - can_cv)
        assert abs_err_cv/np.abs(can_cv) < 1.0e-12
        assert abs_err_cv < 1.0e-6

        can_cp = cantera_soln.cp
        heat_cap_cp = eos.heat_capacity_cp(cv, tin)
        abs_err_cp = inf_norm(heat_cap_cp - can_cp)
        assert abs_err_cp/np.abs(can_cp) < 1.0e-12
        assert abs_err_cp < 1.0e-6

        can_gamma = cantera_soln.cp/cantera_soln.cv
        gamma = eos.gamma(cv, tin)
        abs_err_gamma = inf_norm(gamma - can_gamma)
        assert abs_err_gamma/np.abs(can_gamma) < 1.0e-12
        assert abs_err_gamma < 1.0e-6

        can_e = cantera_soln.int_energy_mass
        int_energy = eos.get_internal_energy(tin, yin)
        abs_err_e = inf_norm(int_energy - can_e)
        assert abs_err_e/np.abs(can_e) < 1.0e-12
        assert abs_err_e < 1.0e-6

        can_h = cantera_soln.enthalpy_mass
        enthalpy = eos.get_enthalpy(tin, yin)
        t_from_h = eos.temperature_from_enthalpy(enthalpy, tin, yin)
        t_from_e = eos.temperature(cv, tin)
        t_resid = inf_norm((t_from_h - t_from_e)/t_from_e)
        print(f"{t_from_h=}")
        assert t_resid < 1.0e-12

        abs_err_h = inf_norm(enthalpy - can_h)
        assert abs_err_h/np.abs(can_h) < 1.0e-12
        assert abs_err_h < 1.0e-6


@pytest.mark.parametrize("mechname", ["air_3sp", "uiuc_7sp", "sandiego",
                                      "uiuc_8sp_phenol", "uiuc_4sp_oxidation"])
def test_pyrometheus_mechanisms(ctx_factory, mechname, output_mechanism=True):
    """Test known pyrometheus mechanisms.

    This test reproduces a pyrometheus-native test in the MIRGE context and
    compare thermo properties to the corresponding mechanism in Cantera.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 2

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)

    if output_mechanism:
        import pyrometheus
        # write then load the mechanism file to yield better readable pytest coverage
        with open(f"./{mechname}.py", "w") as mech_file:
            code = pyrometheus.codegen.python.gen_thermochem_code(cantera_soln)
            print(code, file=mech_file)

        import importlib
        from mirgecom.thermochemistry import get_pyrometheus_wrapper_class
        pyromechlib = importlib.import_module(f"{mechname}")
        pyro_mechanism = get_pyrometheus_wrapper_class(
            pyro_class=pyromechlib.Thermochemistry)(actx.np)
    else:
        pyro_mechanism = \
            get_pyrometheus_wrapper_class_from_cantera(cantera_soln)(actx.np)

    nspecies = pyro_mechanism.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    press0 = 101325.0
    temp0 = 300.0
    y0s = np.ones(shape=(nspecies,))/float(nspecies)

    for fac in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        pressin = fac * press0
        tempin = fac * temp0

        print(f"Testing (t,P) = ({tempin}, {pressin})")
        cantera_soln.TPY = tempin, pressin, y0s

        can_t, can_m, can_y = cantera_soln.TDY
        can_e = cantera_soln.int_energy_mass
        can_p = cantera_soln.P
        can_e = cantera_soln.int_energy_mass
        can_h = cantera_soln.enthalpy_mass
        can_c = cantera_soln.concentrations

        ones = dcoll.zeros(actx) + 1.0
        tin = can_t * ones
        pin = can_p * ones
        yin = make_obj_array([can_y[i] * ones for i in range(nspecies)])

        pyro_m = pyro_mechanism.get_density(pin, tin, yin)
        pyro_e = pyro_mechanism.get_mixture_internal_energy_mass(tin, yin)
        pyro_t = pyro_mechanism.get_temperature(pyro_e, tin, yin)
        pyro_p = pyro_mechanism.get_pressure(pyro_m, pyro_t, yin)
        pyro_h = pyro_mechanism.get_mixture_enthalpy_mass(pyro_t, yin)
        pyro_c = pyro_mechanism.get_concentrations(pyro_m, yin)

        assert inf_norm((pyro_c - can_c)) < 1e-14
        assert inf_norm((pyro_t - can_t) / can_t) < 1e-14
        assert inf_norm((pyro_m - can_m) / can_m) < 1e-14
        assert inf_norm((pyro_p - can_p) / can_p) < 1e-14
        assert inf_norm((pyro_e - can_e) / can_e) < 1e-10
        assert inf_norm((pyro_h - can_h) / can_h) < 1e-10

        # Test the concentrations zero level
        y = -ones*y0s
        print(f"{y=}")
        conc = pyro_mechanism.get_concentrations(pyro_m, y)
        print(f"{conc=}")
        for spec in range(nspecies):
            assert inf_norm(conc[spec]) < 1e-14

        zlev = 1e-3
        test_mech = get_pyrometheus_wrapper_class_from_cantera(
            cantera_soln, zero_level=zlev)(actx.np)

        y = 0*y + zlev
        print(f"{y=}")
        conc = test_mech.get_concentrations(pyro_m, y)
        print(f"{conc=}")
        for spec in range(nspecies):
            assert inf_norm(conc[spec]) < 1e-14


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

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    ones = dcoll.zeros(actx) + 1.0

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_mechanism = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=5)(actx.np)

    nspecies = pyro_mechanism.num_species
    print(f"PyrometheusMixture::Mechanism = {mechname}")
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    eos = PyrometheusMixture(pyro_mechanism)
    gas_model = GasModel(eos=eos)

    press0 = 101500.0
    temp0 = 300.0
    y0s = np.zeros(shape=(nspecies,))
    for i in range(1, nspecies):
        y0s[i] = y0 / (10.0 ** i)
    y0s[0] = 1.0 - np.sum(y0s[1:])
    velocity = vel * np.ones(shape=(dim,))

    for fac in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        tempin = fac * temp0
        pressin = fac * press0

        print(f"Testing {mechname}(t,P) = ({tempin}, {pressin})")

        tin = tempin * ones
        pin = pressin * ones
        yin = y0s * ones

        # avoid starting too far from the actual temperature
        tguess = tempin + ones*100.0*np.random.random()
        assert inf_norm(tguess) > 0.0

        pyro_rho = pyro_mechanism.get_density(pin, tin, yin)
        pyro_e = pyro_mechanism.get_mixture_internal_energy_mass(tin, yin)
        pyro_t = pyro_mechanism.get_temperature(pyro_e, tguess, yin)
        pyro_p = pyro_mechanism.get_pressure(pyro_rho, pyro_t, yin)

        # print(f"prom(rho, y, p, t, e) = ({pyro_rho}, {y0s}, "
        #       f"{pyro_p}, {pyro_t}, {pyro_e})")

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

        # print(f"pyro_y = {y}")
        # print(f"pyro_eos.p = {p}")
        # print(f"pyro_eos.temp = {temperature}")
        # print(f"pyro_eos.e = {internal_energy}")

        tol = 5e-14
        assert inf_norm((cv.mass - pyro_rho) / pyro_rho) < tol
        assert inf_norm((temperature - pyro_t) / pyro_t) < tol
        assert inf_norm((internal_energy - pyro_e) / pyro_e) < tol
        assert inf_norm((p - pyro_p) / pyro_p) < tol

        # Test the concentrations zero level
        y = -1.0*y
        # print(f"{y=}")
        conc = pyro_mechanism.get_concentrations(rho, y)
        # print(f"{conc=}")
        for spec in range(nspecies):
            assert max(conc[spec]).all() >= 0

        zlev = 1e-3
        test_mech = \
            get_pyrometheus_wrapper_class_from_cantera(cantera_soln,
                                                       zero_level=zlev)(actx.np)

        y = 0*y + zlev
        # print(f"{y=}")
        conc = test_mech.get_concentrations(rho, y)
        # print(f"{conc=}")
        for spec in range(nspecies):
            assert max(conc[spec]).all() == 0


@pytest.mark.parametrize("mechname", ["uiuc_7sp_const_gamma"])
def test_temperature_constant_cv(ctx_factory, mechname):
    """Test mechanism with constant heat capacity."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 1

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    zeros = dcoll.zeros(actx)
    ones = dcoll.zeros(actx) + 1.0

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    cantera_soln.set_equivalence_ratio(phi=1.0, fuel="C2H4:1",
                                       oxidizer="O2:1.0,N2:3.76")

    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=0)(actx.np)

    pressin = cantera.one_atm  # pylint: disable=no-member
    eos = PyrometheusMixture(pyro_obj, temperature_guess=666.)

    for tin in ([300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0]):
        cantera_soln.TP = tin, pressin
        print(f"Testing (t,P) = ({tin}, {pressin})")
        cantera_soln.equilibrate("TP")

        rhoin = cantera_soln.density * ones
        yin = cantera_soln.Y * ones
        ein = rhoin*eos.get_internal_energy(temperature=tin,
                                            species_mass_fractions=yin)

        cv = make_conserved(dim=dim, mass=rhoin, energy=ein,
            momentum=make_obj_array([zeros]), species_mass=rhoin*yin)

        temp = eos.temperature(cv=cv, temperature_seed=tin)

        assert inf_norm(temp - tin) > 1e-15


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
