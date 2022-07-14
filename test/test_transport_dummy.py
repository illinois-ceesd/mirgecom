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
import pytest
from pytools.obj_array import make_obj_array

import grudge.op as op

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import cantera
from mirgecom.transport import MixtureAveragedTransport #XXX
from mirgecom.fluid import make_conserved #XXX
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



#TODO To be added to test_eos. I created this file separately to simplify
#      the testing process

@pytest.mark.parametrize(("mechname"),
                         [("uiuc")])
def test_pyrometheus_transport(ctx_factory, mechname):
    """    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 2

    logger.info(f"Number of elements {mesh.nelements}")

    discr = create_discretization_collection(actx, mesh, order=order)
    ones = discr.zeros(actx) + 1.0
    zeros = discr.zeros(actx)

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    # pyro_obj = pyro.get_thermochem_class(cantera_soln)(actx.np)
#    pyro_obj = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)

    from mirgecom.thermochemistry import get_thermochemistry_class_by_mechanism_name
    pyro_obj = get_thermochemistry_class_by_mechanism_name("uiuc",
                                                        temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PrometheusMixture::NumSpecies = {nspecies}")

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

    cantera_soln.TPX = tempin, pressin, x
    #    cantera_soln.equilibrate("UV")
    can_t, can_rho, can_y = cantera_soln.TDY
    can_p = cantera_soln.P

    reactor = cantera.IdealGasConstPressureReactor(cantera_soln)
    sim = cantera.ReactorNet([reactor])
    time = 0.0
    for _ in range(50):
        time += 1.0e-6
        sim.advance(time)

        # Get state from Cantera
        can_rho = reactor.density
        can_t = reactor.T
        can_y = reactor.Y
        print(f"can_p = {can_p}")
        print(f"can_rho = {can_rho}")
        print(f"can_t = {can_t}")
        print(f"can_y = {can_y}")

        tin = can_t * ones
        rhoin = can_rho * ones
        yin = can_y * ones

        # Cantera transport
        mu_ct = cantera_soln.viscosity
        kappa_ct = cantera_soln.thermal_conductivity
        diff_ct = cantera_soln.mix_diff_coeffs

        cv = make_conserved(dim=2, mass=can_rho * ones,
                            momentum=make_obj_array([zeros,zeros]),
                            energy=gas_model.eos.get_internal_energy(tin,can_rho*yin),
                            species_mass=can_rho*yin) #XXX this is strange.. Doesnt it need the mass?

        fluid_state = make_fluid_state(cv, gas_model, tin)

        # Pyrometheus transport
        mu = fluid_state.tv.viscosity
        kappa = fluid_state.tv.thermal_conductivity
        diff = fluid_state.tv.species_diffusivity

        # Print
        def inf_norm(x):
            return actx.to_numpy(op.norm(discr, x, np.inf))

        err_p = np.abs(inf_norm(fluid_state.dv.pressure) - can_p)
        assert err_p < 1.0e-10

        err_t = np.abs(inf_norm(fluid_state.dv.temperature) - can_t)
        assert err_t < 1.0e-12

        err_mu = np.abs(inf_norm(mu) - mu_ct)
        assert err_mu < 1.0e-12

        err_kappa = np.abs(inf_norm(kappa) - kappa_ct)
        assert err_kappa < 1.0e-12

        for i in range(nspecies):
            err_diff = np.abs(inf_norm(diff[i]) - diff_ct[i])
            assert err_diff < 1.0e-12
