"""Test the transport model interfaces."""

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

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from mirgecom.transport import MixtureAveragedTransport
from mirgecom.fluid import make_conserved
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.discretization import create_discretization_collection
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("mechname", ["sandiego"])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("order", [1, 3, 5])
def test_pyrometheus_transport(ctx_factory, mechname, dim, order):
    """Test mixture-averaged transport properties."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    logger.info(f"Number of elements {mesh.nelements}")

    dcoll = create_discretization_collection(actx, mesh, order=order)
    ones = dcoll.zeros(actx) + 1.0
    zeros = dcoll.zeros(actx)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    # Pyrometheus initialization
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    # Transport data initialization
    sing_diff = 1e-6
    transport_model = MixtureAveragedTransport(pyro_obj,
                                               epsilon=1e-4,
                                               singular_diffusivity=sing_diff)
    eos = PyrometheusMixture(pyro_obj, temperature_guess=666.)
    gas_model = GasModel(eos=eos, transport=transport_model)

    for pressin in ([0.25, 1.0]):
        for tempin in ([300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0]):

            print(f"Testing (t,P) = ({tempin}, {pressin})")

            # Loop over each individual species by making a single-species mixture
            for i, name in enumerate(cantera_soln.species_names):
                cantera_soln.TP = tempin, pressin*cantera.one_atm
                cantera_soln.Y = name + ":1"

                can_t, can_rho, can_y = cantera_soln.TDY
                tin = can_t * ones
                rhoin = can_rho * ones
                yin = can_y * ones

                cv = make_conserved(dim=2, mass=rhoin,
                        momentum=make_obj_array([zeros, zeros]),
                        energy=rhoin*gas_model.eos.get_internal_energy(tin, yin),
                        species_mass=rhoin*yin)

                fluid_state = make_fluid_state(cv, gas_model, tin)

                # Viscosity
                mu = fluid_state.tv.viscosity
                mu_ct = cantera_soln.species_viscosities
                err_mu = inf_norm(mu - mu_ct[i])
                assert err_mu < 1.0e-12

                # Thermal conductivity
                kappa = fluid_state.tv.thermal_conductivity
                kappa_ct = cantera_soln.thermal_conductivity
                err_kappa = inf_norm(kappa - kappa_ct)
                assert err_kappa < 1.0e-12

                # NOTE: Individual species are exercised in Pyrometheus.
                # Since the transport model enforce a singular-species case
                # to avoid numerical issues when Yi -> 1, we can not test the
                # individual species diffusivity. However, this tests that
                # the single-species case is enforced correctly.
                diff = fluid_state.tv.species_diffusivity
                err_diff = inf_norm(diff[i] - sing_diff)
                assert err_diff < 1.0e-15

            # prescribe an actual mixture
            cantera_soln.set_equivalence_ratio(phi=1.0, fuel="H2:1",
                                               oxidizer="O2:1.0,N2:3.76")

            cantera_soln.TP = tempin, pressin
            cantera_soln.equilibrate("TP")
            can_t, can_rho, can_y = cantera_soln.TDY
            can_p = cantera_soln.P

            tin = can_t * ones
            rhoin = can_rho * ones
            yin = can_y * ones

            cv = make_conserved(dim=2, mass=rhoin,
                    momentum=make_obj_array([zeros for _ in range(dim)]),
                    energy=rhoin*gas_model.eos.get_internal_energy(tin, yin),
                    species_mass=rhoin*yin)

            fluid_state = make_fluid_state(cv, gas_model, tin)

            # Making sure both pressure and temperature are correct
            err_p = inf_norm(fluid_state.dv.pressure - can_p)/can_p
            assert err_p < 1.0e-14

            err_t = inf_norm(fluid_state.dv.temperature - can_t)/can_t
            assert err_t < 1.0e-14

            # Viscosity
            mu = fluid_state.tv.viscosity
            mu_ct = cantera_soln.viscosity
            err_mu = inf_norm(mu - mu_ct)
            assert err_mu < 1.0e-12

            # Thermal conductivity
            kappa = fluid_state.tv.thermal_conductivity
            kappa_ct = cantera_soln.thermal_conductivity
            err_kappa = inf_norm(kappa - kappa_ct)
            assert err_kappa < 1.0e-12

            # Species diffusivities
            diff = fluid_state.tv.species_diffusivity
            diff_ct = cantera_soln.mix_diff_coeffs
            for i in range(nspecies):
                err_diff = inf_norm(diff[i] - diff_ct[i])
                assert err_diff < 1.0e-11
