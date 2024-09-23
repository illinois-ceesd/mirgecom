"""Test the chemistry source terms."""

__copyright__ = """
Copyright (C) 2024 University of Illinois Board of Trustees
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
import pyopencl as cl
import pytest
import cantera
from pytools.obj_array import make_obj_array

from grudge import op

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.mesh.generation import generate_regular_rect_mesh
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts

from mirgecom.fluid import make_conserved
from mirgecom.eos import PyrometheusMixture
from mirgecom.discretization import create_discretization_collection
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import (
    get_pyrometheus_wrapper_class_from_cantera,
    get_pyrometheus_wrapper_class
)

pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory])


@pytest.mark.parametrize(("mechname", "fuel", "rate_tol"),
                         [("uiuc_7sp", "C2H4", 1e-11),
                          ("sandiego", "H2", 1e-9)])
@pytest.mark.parametrize("reactor_type",
                         ["IdealGasReactor", "IdealGasConstPressureReactor"])
@pytest.mark.parametrize(("pressure", "nsteps"),
                         [(25000.0, 100), (101325.0, 50)])
def test_pyrometheus_kinetics(ctx_factory, mechname, fuel, rate_tol, reactor_type,
                              pressure, nsteps, output_mechanism=True):
    """Test known pyrometheus reaction mechanisms.

    This test reproduces a pyrometheus-native test in the MIRGE context.

    Tests that the Pyrometheus mechanism code gets the same reaction rates as
    the corresponding mechanism in Cantera. The reactions are integrated in
    time and verified against homogeneous reactors in Cantera.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 4
    order = 4

    mesh = generate_regular_rect_mesh(a=(-0.5,) * dim, b=(0.5,) * dim,
                                      nelements_per_axis=(nel_1d,) * dim)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    ones = dcoll.zeros(actx) + 1.0

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
        pyromechlib = importlib.import_module(f"{mechname}")
        pyro_obj = get_pyrometheus_wrapper_class(
            pyro_class=pyromechlib.Thermochemistry)(actx.np)
    else:
        pyro_obj = get_pyrometheus_wrapper_class_from_cantera(cantera_soln)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    tempin = 1200.0
    pressin = pressure
    print(f"Testing (t,P) = ({tempin}, {pressin})")

    # Homogeneous reactor to get test data
    cantera_soln.set_equivalence_ratio(phi=1.0, fuel=fuel+":1",
                                       oxidizer="O2:1.0,N2:3.76")
    cantera_soln.TP = tempin, pressin
    reactor = None
    # constant density, variable pressure
    if reactor_type == "IdealGasReactor":
        reactor = cantera.IdealGasReactor(cantera_soln,  # pylint: disable=no-member
                                          name="Batch Reactor")

    # constant pressure, variable density
    if reactor_type == "IdealGasConstPressureReactor":
        reactor = cantera.IdealGasConstPressureReactor(  # pylint: disable=no-member
            cantera_soln, name="Batch Reactor")

    sim = cantera.ReactorNet([reactor])  # pylint: disable=no-member

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    def get_mixture_entropy_mass(pressure, temperature, mass_fractions):
        mmw = pyro_obj.get_mix_molecular_weight(mass_fractions)

        return 1.0/mmw * get_mixture_entropy_mole(pressure, temperature,
                                                  mass_fractions)

    def get_mole_average_property(mass_fractions, spec_property):
        mmw = pyro_obj.get_mix_molecular_weight(mass_fractions)
        mole_fracs = pyro_obj.get_mole_fractions(mmw, mass_fractions)
        return sum(mole_fracs[i] * spec_property[i] for i in range(nspecies))

    # def get_mixture_enthalpy_mole(temperature, mass_fractions):
    #     h0_rt = pyro_obj.get_species_enthalpies_rt(temperature)
    #     hmix = get_mole_average_property(mass_fractions, h0_rt)
    #     return pyro_obj.gas_constant * temperature * hmix

    def get_mixture_entropy_mole(pressure, temperature, mass_fractions):
        mmw = pyro_obj.get_mix_molecular_weight(mass_fractions)
        # necessary to avoid nans in the log function below
        x = actx.np.where(
            actx.np.less(pyro_obj.get_mole_fractions(mmw, mass_fractions), 1e-16),
            1e-16, pyro_obj.get_mole_fractions(mmw, mass_fractions))
        s0_r = pyro_obj.get_species_entropies_r(pressure, temperature)
        s_t_mix = get_mole_average_property(mass_fractions, s0_r)
        s_x_mix = get_mole_average_property(mass_fractions, actx.np.log(x))
        return pyro_obj.gas_constant * (s_t_mix - s_x_mix)

    time = 0.0
    dt = 2.5e-6
    for _ in range(nsteps):
        time += dt
        sim.advance(time)

        # Get state from Cantera
        can_t = reactor.T
        can_p = cantera_soln.P
        can_rho = reactor.density
        can_y = reactor.Y
        # print(f"can_y = {can_y}")

        tin = can_t * ones
        pin = can_p * ones
        rhoin = can_rho * ones
        yin = can_y * ones

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # assert properties used internaly in the chemistry

        for i in range(nspecies):
            # species entropy
            can_s = cantera_soln.standard_entropies_R[i]
            spec_entropy = pyro_obj.get_species_entropies_r(pin, tin)[i]
            abs_error_s = inf_norm(spec_entropy - can_s)
            assert abs_error_s < 1.0e-13

            # species Gibbs energy
            can_g = cantera_soln.standard_gibbs_RT[i]
            spec_gibbs = pyro_obj.get_species_gibbs_rt(pin, tin)[i]
            abs_error_g = inf_norm(spec_gibbs - can_g)
            assert abs_error_g < 1.0e-13

        # mixture entropy mole
        can_s_mix_mole = cantera_soln.entropy_mole
        s_mix_mole = get_mixture_entropy_mole(pin, tin, yin)
        abs_error_s_mix_mole = inf_norm(s_mix_mole - can_s_mix_mole)
        assert abs_error_s_mix_mole/can_s_mix_mole < 2.0e-11
        assert abs_error_s_mix_mole < 5.0e-6

        # mixture entropy mass
        can_s_mix_mass = cantera_soln.entropy_mass
        s_mix_mass = get_mixture_entropy_mass(pin, tin, yin)
        abs_error_s_mix_mass = inf_norm(s_mix_mass - can_s_mix_mass)
        assert abs_error_s_mix_mass/can_s_mix_mass < 2.0e-11
        assert abs_error_s_mix_mass < 5.0e-6

        # delta enthalpy
        can_delta_h = cantera_soln.delta_enthalpy/(pyro_obj.gas_constant*tin)
        nu = cantera_soln.product_stoich_coeffs - cantera_soln.reactant_stoich_coeffs
        delta_h = nu.T@pyro_obj.get_species_enthalpies_rt(tin)
        abs_error_delta_h = inf_norm(can_delta_h - delta_h)
        assert abs_error_delta_h < 1e-13

        # delta entropy
        # zero or negative mole fractions values are troublesome due to the log
        # see CHEMKIN manual for more details
        mmw = pyro_obj.get_mix_molecular_weight(yin)
        _x = pyro_obj.get_mole_fractions(mmw, yin)
        mole_fracs = actx.np.where(actx.np.less(_x, 1e-15), 1e-15, _x)  # noqa
        delta_s = nu.T@(pyro_obj.get_species_entropies_r(pin, tin)
                        - actx.np.log(mole_fracs))
        # exclude meaningless check on entropy for irreversible reaction
        for i, reaction in enumerate(cantera_soln.reactions()):
            if reaction.reversible:
                can_delta_s = cantera_soln.delta_entropy[i]/pyro_obj.gas_constant
                assert inf_norm(can_delta_s - delta_s[i]) < 1e-13

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        pyro_c = pyro_obj.get_concentrations(rhoin, yin)
        # print(f"pyro_conc = {pyro_c}")

        # forward rates
        kfw_pm = pyro_obj.get_fwd_rate_coefficients(tin, pyro_c)
        kfw_ct = cantera_soln.forward_rate_constants
        for i, _ in enumerate(cantera_soln.reactions()):
            assert inf_norm((kfw_pm[i] - kfw_ct[i]) / kfw_ct[i]) < 1.0e-13

        # equilibrium rates
        keq_pm = actx.np.exp(-1.*pyro_obj.get_equilibrium_constants(pin, tin))
        keq_ct = cantera_soln.equilibrium_constants
        for i, reaction in enumerate(cantera_soln.reactions()):
            if reaction.reversible:  # skip irreversible reactions
                assert inf_norm((keq_pm[i] - keq_ct[i]) / keq_ct[i]) < 1.0e-13

        # reverse rates
        krv_pm = pyro_obj.get_rev_rate_coefficients(pin, tin, pyro_c)
        krv_ct = cantera_soln.reverse_rate_constants
        for i, reaction in enumerate(cantera_soln.reactions()):
            if reaction.reversible:  # skip irreversible reactions
                assert inf_norm((krv_pm[i] - krv_ct[i]) / krv_ct[i]) < 1.0e-13

        # reaction progress
        rates_pm = pyro_obj.get_net_rates_of_progress(pin, tin, pyro_c)
        rates_ct = cantera_soln.net_rates_of_progress
        for i, rates in enumerate(rates_ct):
            assert inf_norm(rates_pm[i] - rates) < rate_tol

        # species production/destruction
        omega_pm = pyro_obj.get_net_production_rates(rhoin, tin, yin)
        omega_ct = cantera_soln.net_production_rates
        for i, omega in enumerate(omega_ct):
            assert inf_norm(omega_pm[i] - omega) < rate_tol

    # check that the reactions progress far enough (and remains stable)
    assert can_t > 1800.0
    assert can_t < 3200.0


@pytest.mark.parametrize(("mechname", "fuel", "rate_tol"),
                         [("uiuc_7sp", "C2H4", 1e-11),
                          ("sandiego", "H2", 1e-9)])
@pytest.mark.parametrize("reactor_type",
                         ["IdealGasReactor", "IdealGasConstPressureReactor"])
@pytest.mark.parametrize(("pressure", "nsteps"),
                         [(25000.0, 100), (101325.0, 50)])
def test_mirgecom_kinetics(ctx_factory, mechname, fuel, rate_tol, reactor_type,
                           pressure, nsteps):
    """Test of known pyrometheus reaction mechanisms in the MIRGE context.

    Tests that the Pyrometheus mechanism code gets the same reaction rates as
    the corresponding mechanism in Cantera. The reactions are integrated in
    time and verified against a homogeneous reactor in Cantera.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 1
    order = 4

    mesh = generate_regular_rect_mesh(a=(-0.5,) * dim, b=(0.5,) * dim,
                                      nelements_per_axis=(nel_1d,) * dim)

    dcoll = create_discretization_collection(actx, mesh, order=order)
    zeros = dcoll.zeros(actx)
    ones = dcoll.zeros(actx) + 1.0

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)

    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=5)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    tempin = 1200.0
    print(f"Testing (t,P) = ({tempin}, {pressure})")

    # Homogeneous reactor to get test data
    cantera_soln.set_equivalence_ratio(phi=1.0, fuel=fuel+":1",
                                       oxidizer="O2:1.0,N2:3.76")
    cantera_soln.TP = tempin, pressure

    eos = PyrometheusMixture(pyro_obj, temperature_guess=tempin)

    # constant density, variable pressure
    if reactor_type == "IdealGasReactor":
        reactor = cantera.IdealGasReactor(  # pylint: disable=no-member
            cantera_soln, name="Batch Reactor")

    # constant pressure, variable density
    if reactor_type == "IdealGasConstPressureReactor":
        reactor = cantera.IdealGasConstPressureReactor(  # pylint: disable=no-member
            cantera_soln, name="Batch Reactor")

    net = cantera.ReactorNet([reactor])  # pylint: disable=no-member

    time = 0.0
    dt = 2.5e-6
    for _ in range(nsteps):
        time += dt
        net.advance(time)

        tin = reactor.T * ones
        rhoin = reactor.density * ones
        yin = reactor.Y * ones
        ein = rhoin * eos.get_internal_energy(temperature=tin,
                                              species_mass_fractions=yin)

        cv = make_conserved(dim=dim, mass=rhoin, energy=ein,
            momentum=make_obj_array([zeros]), species_mass=rhoin*yin)

        temp = eos.temperature(cv=cv, temperature_seed=tin)

        # do NOT test anything else. If this match, then everything is ok
        omega_mc = eos.get_production_rates(cv, temp)
        omega_ct = cantera_soln.net_production_rates
        for i in range(cantera_soln.n_species):
            assert inf_norm((omega_mc[i] - omega_ct[i])) < rate_tol

    # check that the reactions progress far enough and are stable
    assert reactor.T > 1800.0
    assert reactor.T < 3200.0
