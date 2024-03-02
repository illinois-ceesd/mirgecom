"""Test wall-model related functions."""

__copyright__ = """Copyright (C) 2023 University of Illinois Board of Trustees"""

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

import pytest
import cantera
import numpy as np
from pytools.obj_array import make_obj_array
from grudge import op
from meshmode.dof_array import DOFArray
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from mirgecom.simutil import get_box_mesh
from mirgecom.discretization import create_discretization_collection
from mirgecom.eos import PyrometheusMixture
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import make_fluid_state
from mirgecom.wall_model import PorousFlowModel, PorousWallTransport
from mirgecom.materials.initializer import PorousWallInitializer
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera


@pytest.mark.parametrize("order", [1, 4])
@pytest.mark.parametrize("my_material", ["fiber", "composite"])
def test_wall_eos(actx_factory, order, my_material):
    """Check the wall degradation model."""
    actx = actx_factory()

    dim = 2

    tol = 1e-8

    nelems = 20
    mesh = get_box_mesh(dim, -0.1, 0.1, nelems)
    dcoll = create_discretization_collection(
        actx, mesh, order=order, quadrature_order=2*order+1)

    nodes = actx.thaw(dcoll.nodes())
    zeros = actx.np.zeros_like(nodes[0])

    # {{{ EOS initialization

    mech_input = get_mechanism_input("uiuc_8sp_phenol")
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species

    x = make_obj_array([0.0 for i in range(nspecies)])
    x[cantera_soln.species_index("O2")] = 0.21
    x[cantera_soln.species_index("N2")] = 0.79

    cantera_soln.TPX = 900.0, 101325.0, x
    _, _, y = cantera_soln.TDY

    eos = PyrometheusMixture(pyro_obj, temperature_guess=900.0)

    # }}}

    # {{{ Initialize wall model

    if my_material == "fiber":
        import mirgecom.materials.carbon_fiber as material_sample
        material = material_sample.FiberEOS(
            dim=dim, anisotropic_direction=0, char_mass=0., virgin_mass=168.)
        material_densities = 0.12*1400.0

    if my_material == "composite":
        import mirgecom.materials.tacot as material_sample
        material = material_sample.TacotEOS(char_mass=220., virgin_mass=280.)
        material_densities = np.empty((3,), dtype=object)
        material_densities[0] = 30.0
        material_densities[1] = 90.0
        material_densities[2] = 160.

    # }}}

    # {{{ Gas model

    base_transport = SimpleTransport()
    solid_transport = PorousWallTransport(base_transport=base_transport)
    gas_model = PorousFlowModel(eos=eos, transport=solid_transport,
                                wall_eos=material)

    # }}}

    pressure = 101325.0 + zeros
    temperature = 900.0 + zeros

    sample_init = PorousWallInitializer(
        pressure=pressure, temperature=temperature, species=y,
        material_densities=material_densities)

    cv, solid_densities = sample_init(nodes, gas_model)

    solid_state = make_fluid_state(cv=cv, gas_model=gas_model,
        material_densities=solid_densities, temperature_seed=900.0)

    wv = solid_state.wv

    assert actx.to_numpy(op.norm(dcoll, wv.tau - 1.0, np.inf)) < tol

    assert isinstance(solid_state.wv.density, DOFArray)

    if my_material == "fiber":
        assert np.max(actx.to_numpy(wv.density - 168.0)) < tol
        assert np.max(actx.to_numpy(wv.void_fraction)) - 1.00 < tol

    if my_material == "composite":
        assert np.max(actx.to_numpy(wv.density - 280.0)) < tol
        assert np.max(actx.to_numpy(wv.void_fraction)) - 1.00 < tol

    assert actx.to_numpy(
        op.norm(dcoll, solid_state.pressure - 101325.0, np.inf)) < tol
    assert actx.to_numpy(
        op.norm(dcoll, solid_state.temperature - 900.0, np.inf)) < tol
