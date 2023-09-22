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
import grudge.op as op
from pytools.obj_array import make_obj_array
from grudge.dof_desc import DOFDesc, VolumeDomainTag, DISCR_TAG_BASE
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


import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("order", [1, 4])
@pytest.mark.parametrize("my_material", ["composite"])
def test_wallmodel_wall_only(actx_factory, order, my_material):
    """Check the wall degradation model."""
    actx = actx_factory()

    dim = 2

    nelems = 20
    global_mesh = get_box_mesh(dim, -0.1, 0.1, nelems)

    mgrp, = global_mesh.groups
    x = global_mesh.vertices[0, mgrp.vertex_indices]
    x_elem_avg = np.sum(x, axis=1)/x.shape[0]
    volume_to_elements = {
        "Fluid": np.where(x_elem_avg > 0)[0],
        "Solid": np.where(x_elem_avg <= 0)[0]}

    from meshmode.mesh.processing import partition_mesh
    volume_meshes = partition_mesh(global_mesh, volume_to_elements)

    dcoll = create_discretization_collection(
        actx, volume_meshes, order=order, quadrature_order=2*order+1)

    dd_vol_fluid = DOFDesc(VolumeDomainTag("Fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("Solid"), DISCR_TAG_BASE)

    fluid_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_solid))

    fluid_zeros = actx.np.zeros_like(fluid_nodes[0])
    solid_zeros = actx.np.zeros_like(solid_nodes[0])

    from grudge.discretization import filter_part_boundaries
    from grudge.reductions import integral
    solid_dd_list = filter_part_boundaries(dcoll, volume_dd=dd_vol_solid,
                                           neighbor_volume_dd=dd_vol_fluid)
    solid_normal = actx.thaw(dcoll.normal(solid_dd_list[0]))

    fluid_dd_list = filter_part_boundaries(dcoll, volume_dd=dd_vol_fluid,
                                           neighbor_volume_dd=dd_vol_solid)
    fluid_normal = actx.thaw(dcoll.normal(fluid_dd_list[0]))

    interface_zeros = actx.np.zeros_like(solid_normal[0])

    integral_volume = integral(dcoll, dd_vol_solid, fluid_zeros + 1.0)
    integral_surface = integral(dcoll, solid_dd_list[0], interface_zeros + 1.0)

    print(integral_volume)
    print(integral_surface)

    # {{{ Pyrometheus initialization

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
        material = material_sample.FiberEOS(char_mass=0., virgin_mass=160.)
        decomposition = material_sample.Y3_Oxidation_Model(wall_material=material)
        solid_density = 0.1*1600.0 + solid_zeros

    if my_material == "composite":
        import mirgecom.materials.tacot as material_sample
        material = material_sample.TacotEOS(char_mass=220., virgin_mass=280.)
        decomposition = material_sample.Pyrolysis()
        solid_density = np.empty((3,), dtype=object)
        solid_density[0] = 30.0 + solid_zeros
        solid_density[1] = 90.0 + solid_zeros
        solid_density[2] = 160. + solid_zeros

    # }}}

    # {{{ Gas model

    base_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.0,
                                     species_diffusivity=np.zeros(nspecies,))
    solid_transport = PorousWallTransport(base_transport=base_transport)
    gas_model_solid = PorousFlowModel(eos=eos, transport=solid_transport,
                                      wall_eos=material)

    # }}}

    # {{{ Solid cv

    temperature = 900.0 + solid_zeros
    species_mass_fractions = y + solid_zeros
    mass = eos.get_density(pressure=101325.0, temperature=temperature,
                           species_mass_fractions=species_mass_fractions)
    sample_init = PorousWallInitializer(density=mass, temperature=temperature,
        species=species_mass_fractions, material_densities=solid_density)

    wv = sample_init(2, solid_nodes, gas_model_solid)
    solid_state = make_fluid_state(cv=wv, gas_model=gas_model_solid,
        material_densities=solid_density, temperature_seed=900.0)

    # }}}

    def blowing_velocity(cv, source):

        # volume integral of the source terms
        integral_volume_source = integral(dcoll, dd_vol_solid, source)

        # restrict to coupled surface
        surface_density = op.project(dcoll, dd_vol_solid, solid_dd_list[0], cv.mass)

        # surface integral of the density
        integral_surface_density = integral(dcoll, solid_dd_list[0], surface_density)

        # boundary velocity
        bndry_velocity = \
            integral_volume_source/integral_surface_density + interface_zeros

        return bndry_velocity, fluid_normal

    # ~~~

    tol = 1e-8

    if my_material == "composite":
        solid_mass_rhs = decomposition.get_source_terms(
            temperature=solid_state.temperature,
            chi=solid_state.wv.material_densities)

        sample_source_gas = -sum(solid_mass_rhs)

        assert actx.to_numpy(
            op.norm(dcoll, solid_mass_rhs[0] + 26.76761185399653, np.inf)) < tol
        assert actx.to_numpy(
            op.norm(dcoll, solid_mass_rhs[1] + 2.03565420370596, np.inf)) < tol
        assert actx.to_numpy(
            op.norm(dcoll, sample_source_gas - 28.80326605770249, np.inf)) < tol

        ref_bnd_velocity = 9.216295153230408
        bnd_velocity, normal = blowing_velocity(solid_state.cv, -sample_source_gas)
        normal_velocity = bnd_velocity*normal[0]
        assert actx.to_numpy(
            op.norm(dcoll, normal_velocity - ref_bnd_velocity, np.inf)) < 1e-6
