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
import pyopencl as cl
import pyopencl.clrandom
import pyopencl.clmath
import pyopencl.tools as cl_tools
import pytest
from pytools.obj_array import make_obj_array

import grudge.op as op

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import cantera
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.initializers import (
    Vortex2D, Lump,
    MixtureInitializer
)
from mirgecom.discretization import create_discretization_collection
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from mirgecom.mechanisms import get_mechanism_input

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("mechname", "rate_tol"),
                         [("uiuc", 1e-12), ])
@pytest.mark.parametrize("y0", [0, 1])
def do_not_test_lazy_pyro(ctx_factory, mechname, rate_tol, y0):
    """Test lazy pyrometheus mechanisms.

    This test reproduces a pyrometheus-native test in the MIRGE context using both
    eager and lazy evaluation protocols. The purpose of this test is making sure that
    lazy evaluation mode is getting the same answers as eager (within a tolerance).

    Some sanity checks to make sure eager is matching Cantera are also performed.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx_eager = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    actx_lazy = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 2

    logger.info(f"Number of elements {mesh.nelements}")

    discr_eager = create_discretization_collection(actx_eager, mesh, order=order)
    discr_lazy = create_discretization_collection(actx_lazy, mesh, order=order)

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    sol = cantera.Solution(name="gas", yaml=mech_input)

    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    pyro_eager = make_pyrometheus_mechanism_class(sol)(actx_eager.np)
    pyro_lazy = make_pyrometheus_mechanism_class(sol)(actx_lazy.np)

    nspecies = pyro_eager.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    press0 = 101500.0
    temp0 = 300.0
    y0s = np.zeros(shape=(nspecies,))
    for i in range(nspecies-1):
        y0s[i] = y0 / (10.0 ** (i + 1))
    y0s[-1] = 1.0 - np.sum(y0s[:-1])

    def get_temperature_lazy(energy, y, tguess):
        return make_obj_array(
            [pyro_lazy.get_temperature(energy, y, tguess)]
        )

    temp_lazy = actx_lazy.compile(get_temperature_lazy)

    for fac in range(1, 11):
        pressin = fac * press0
        tempin = fac * temp0

        print(f"Testing (t,P) = ({tempin}, {pressin})")
        cantera_soln = cantera.Solution(phase_id="gas", source=mech_input)
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

        ones_lazy = discr_lazy.zeros(actx_lazy) + 1.0
        tin_lazy = can_t * ones_lazy
        pin_lazy = can_p * ones_lazy
        yin_lazy = make_obj_array([can_y[i] * ones_lazy for i in range(nspecies)])

        ones_eager = discr_eager.zeros(actx_eager) + 1.0
        tin_eager = can_t * ones_eager
        pin_eager = can_p * ones_eager
        yin_eager = make_obj_array([can_y[i] * ones_eager for i in range(nspecies)])

        pyro_rho_eager = pyro_eager.get_density(pin_eager, tin_eager, yin_eager)
        pyro_rho_lazy = pyro_lazy.get_density(pin_lazy, tin_lazy, yin_lazy)

        from arraycontext import to_numpy
        rho_lazy = to_numpy(
            actx_eager.thaw(actx_lazy.freeze(pyro_rho_lazy))
        )

        pyro_e_eager = pyro_eager.get_mixture_internal_energy_mass(tin_eager,
                                                                   yin_eager)
        pyro_e_lazy = pyro_lazy.get_mixture_internal_energy_mass(tin_lazy, yin_lazy)
        e_lazy = to_numpy(
            actx_eager.thaw(actx_lazy.freeze(pyro_e_lazy))
        )

        # These both take 5 Newton iterations
        pyro_t_eager = pyro_eager.get_temperature(pyro_e_eager, tin_eager, yin_eager)
        pyro_t_lazy = temp_lazy(pyro_e_lazy, tin_lazy, yin_lazy)

        t_lazy = to_numpy(actx_eager.thaw(actx_lazy.freeze(pyro_t_lazy)))

        pyro_p_eager = pyro_eager.get_pressure(pyro_rho_eager, tin_eager, yin_eager)
        pyro_c_eager = pyro_eager.get_concentrations(pyro_rho_eager, yin_eager)
        pyro_k_eager = pyro_eager.get_fwd_rate_coefficients(pyro_t_eager,
                                                            pyro_c_eager)

        pyro_p_lazy = pyro_lazy.get_pressure(pyro_rho_lazy, tin_lazy, yin_lazy)
        pyro_c_lazy = pyro_lazy.get_concentrations(pyro_rho_lazy, yin_lazy)
        pyro_k_lazy = pyro_lazy.get_fwd_rate_coefficients(pyro_t_lazy, pyro_c_lazy)

        c_lazy = to_numpy(
            actx_eager.thaw(actx_lazy.freeze(pyro_c_lazy))
        )
        p_lazy = to_numpy(
            actx_eager.thaw(actx_lazy.freeze(pyro_p_lazy))
        )
        k_lazy = to_numpy(
            actx_eager.thaw(actx_lazy.freeze(pyro_k_lazy))
        )

        # Pyro chemistry functions
        pyro_r_eager = pyro_eager.get_net_rates_of_progress(pyro_t_eager,
                                                            pyro_c_eager)
        pyro_omega_eager = pyro_eager.get_net_production_rates(pyro_rho_eager,
                                                               pyro_t_eager,
                                                               yin_eager)

        pyro_r_lazy = pyro_lazy.get_net_rates_of_progress(pyro_t_lazy,
                                                          pyro_c_lazy)
        pyro_omega_lazy = pyro_lazy.get_net_production_rates(pyro_rho_lazy,
                                                             pyro_t_lazy,
                                                             yin_lazy)
        r_lazy = to_numpy(
            actx_eager.thaw(actx_lazy.freeze(pyro_r_lazy))
        )
        omega_lazy = to_numpy(
            actx_eager.thaw(actx_lazy.freeze(pyro_omega_lazy))
        )

        print(f"can(rho, y, p, t, e, k) = ({can_rho}, {can_y}, "
              f"{can_p}, {can_t}, {can_e}, {can_k})")
        print(f"pyro_eager(rho, y, p, t, e, k) = ({pyro_rho_eager}, {y0s}, "
              f"{pyro_p_eager}, {pyro_t_eager}, {pyro_e_eager}, {pyro_k_eager})")
        print(f"pyro_lazy(rho, y, p, t, e, k) = ({rho_lazy}, {y0s}, "
              f"{p_lazy}, {t_lazy}, {e_lazy}, {k_lazy})")

        # For pyro chem testing
        print(f"{can_r=}")
        print(f"{pyro_r_eager=}")
        print(f"{r_lazy=}")
        print(f"{can_omega=}")
        print(f"{pyro_omega_eager=}")
        print(f"{omega_lazy=}")

        tol = 1e-10
        assert discr_eager.norm((pyro_c_eager - c_lazy), np.inf) < tol
        assert discr_eager.norm((pyro_t_eager - t_lazy), np.inf) < tol
        assert discr_eager.norm((pyro_rho_eager - rho_lazy), np.inf) < tol
        assert discr_eager.norm((pyro_p_eager - p_lazy), np.inf) < 1e-9
        assert discr_eager.norm((pyro_e_eager - e_lazy), np.inf) < 1e-5
        assert discr_eager.norm((pyro_k_eager - k_lazy), np.inf) < 1e-5

        assert discr_eager.norm((pyro_c_eager - can_c) / can_c, np.inf) < 1e-14
        assert discr_eager.norm((pyro_t_eager - can_t) / can_t, np.inf) < 1e-14
        assert discr_eager.norm((pyro_rho_eager - can_rho) / can_rho, np.inf) < 1e-14
        assert discr_eager.norm((pyro_p_eager - can_p) / can_p, np.inf) < 1e-14
        assert discr_eager.norm((pyro_e_eager - can_e) / can_e, np.inf) < 1e-6
        assert discr_eager.norm((pyro_k_eager - can_k) / can_k, np.inf) < 1e-10

        # Pyro chem test comparisons
        for i, rate in enumerate(can_r):
            assert discr_eager.norm((pyro_r_eager[i] - r_lazy[i]), np.inf) < tol
            assert discr_eager.norm((pyro_r_eager[i] - rate), np.inf) < rate_tol
        for i, rate in enumerate(can_omega):
            assert discr_eager.norm(
                (pyro_omega_eager[i] - omega_lazy[i]), np.inf) < tol
            assert discr_eager.norm((pyro_omega_eager[i] - rate), np.inf) < rate_tol


@pytest.mark.parametrize(("mechname", "rate_tol"),
                         [("uiuc", 1e-12),
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

    discr = create_discretization_collection(actx, mesh, order=order)

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

        ones = discr.zeros(actx) + 1.0
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
            return actx.to_numpy(op.norm(discr, x, np.inf))

        assert inf_norm((prom_c - can_c) / can_c) < 1e-14
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


@pytest.mark.parametrize("mechname", ["uiuc", "sandiego"])
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

    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(discr.nodes())

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    sol = cantera.Solution(name="gas", yaml=mech_input)
    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
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

        ones = discr.zeros(actx) + 1.0
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
        initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                         pressure=pyro_p, temperature=pyro_t,
                                         massfractions=y0s, velocity=velocity)

        cv = initializer(eos=eos, t=0, x_vec=nodes)
        fluid_state = make_fluid_state(cv, gas_model, temperature_seed=tguess)
        p = fluid_state.pressure
        temperature = fluid_state.temperature
        internal_energy = eos.get_internal_energy(temperature=tin,
                                                  species_mass_fractions=yin)
        y = cv.species_mass_fractions

        print(f"pyro_y = {y}")
        print(f"pyro_eos.p = {p}")
        print(f"pyro_eos.temp = {temperature}")
        print(f"pyro_eos.e = {internal_energy}")

        def inf_norm(x):
            return actx.to_numpy(op.norm(discr, x, np.inf))

        tol = 1e-14
        assert inf_norm((cv.mass - pyro_rho) / pyro_rho) < tol
        assert inf_norm((temperature - pyro_t) / pyro_t) < tol
        assert inf_norm((internal_energy - pyro_e) / pyro_e) < tol
        assert inf_norm((p - pyro_p) / pyro_p) < tol


@pytest.mark.parametrize(("mechname", "rate_tol"),
                         [("uiuc", 1e-12),
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

    discr = create_discretization_collection(actx, mesh, order=order)
    ones = discr.zeros(actx) + 1.0

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
            return actx.to_numpy(op.norm(discr, x, np.inf))

        print(f"can_r = {can_r}")
        print(f"pyro_r = {pyro_r}")
        abs_diff = inf_norm(pyro_r - can_r)
        if abs_diff > 1e-14:
            min_r = (np.abs(can_r)).min()
            if min_r > 0:
                assert inf_norm((pyro_r - can_r) / can_r) < rate_tol
            else:
                assert inf_norm(pyro_r) < rate_tol

        print(f"can_omega = {can_omega}")
        print(f"pyro_omega = {pyro_omega}")
        for i, omega in enumerate(can_omega):
            omin = np.abs(omega).min()
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

    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(discr.nodes())

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    velocity[0] = 1
    lump = Lump(dim=dim, center=center, velocity=velocity)
    eos = IdealSingleGas()
    cv = lump(nodes)

    def inf_norm(x):
        return actx.to_numpy(op.norm(discr, x, np.inf))

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

    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(discr.nodes())
    eos = IdealSingleGas()
    # Init soln with Vortex
    vortex = Vortex2D()
    cv = vortex(nodes)

    def inf_norm(x):
        return actx.to_numpy(op.norm(discr, x, np.inf))

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
