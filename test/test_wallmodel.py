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
import grudge.op as op
from mirgecom.simutil import get_box_mesh
from grudge.dof_desc import DOFDesc, VolumeDomainTag, DISCR_TAG_BASE, DISCR_TAG_QUAD
from mirgecom.discretization import create_discretization_collection
from mirgecom.eos import PyrometheusMixture
from mirgecom.transport import SimpleTransport
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.boundary import AdiabaticNoslipWallBoundary
from mirgecom.navierstokes import (
    grad_t_operator, grad_cv_operator, ns_operator
)
from mirgecom.multiphysics.multiphysics_coupled_fluid_wall import (
    add_interface_boundaries_no_grad as add_multiphysics_interface_bdries_no_grad,
    add_interface_boundaries as add_multiphysics_interface_bdries
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from mirgecom.wall_model import PorousFlowModel, PorousWallTransport
from mirgecom.materials.initializer import PorousWallInitializer
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera


import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("use_overintegration", [False])
@pytest.mark.parametrize("my_material", ["composite"])
def test_wallmodel(
        actx_factory, order, use_overintegration, my_material, visualize=True):
    """Check the wall degradation model."""
    actx = actx_factory()

    dim = 2

    nelems = 20
    global_mesh = get_box_mesh(2, -0.1, 0.1, nelems)

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
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

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

    fluid_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=1.0,
                                      species_diffusivity=np.zeros(nspecies,))
    gas_model_fluid = GasModel(eos=eos, transport=fluid_transport)

    base_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.0,
                                     species_diffusivity=np.zeros(nspecies,))
    solid_transport = PorousWallTransport(base_transport=base_transport)
    gas_model_solid = PorousFlowModel(eos=eos, transport=solid_transport,
                                      wall_eos=material)

    # }}}

    # {{{ Fluid cv

    # here, fluid temperature is given by a parabolic function:
    a = -183708.0
    b = 36741.6
    c = 900.0
    temperature = a*fluid_nodes[0]**2 + b*fluid_nodes[0] + c
    species_mass_fractions = y + fluid_zeros
    mass = eos.get_density(pressure=101325.0, temperature=temperature,
                           species_mass_fractions=species_mass_fractions)
    energy = mass*eos.get_internal_energy(temperature, species_mass_fractions)
    momentum = make_obj_array([0.0*fluid_nodes[0], 0.0*fluid_nodes[0]])
    species_mass = mass*species_mass_fractions

    cv = make_conserved(dim=dim, mass=mass, energy=energy, momentum=momentum,
                        species_mass=species_mass)
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model_fluid,
                                   temperature_seed=temperature)

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

    # {{{ Setup boundaries and interface

    fluid_boundaries = {
        dd_vol_fluid.trace("-2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+1").domain_tag: AdiabaticNoslipWallBoundary(),
    }

    solid_boundaries = {
        dd_vol_solid.trace("-2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("+2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("-1").domain_tag: AdiabaticNoslipWallBoundary(),
    }

    fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
        add_multiphysics_interface_bdries_no_grad(
            dcoll, dd_vol_fluid, dd_vol_solid, gas_model_fluid, gas_model_solid,
            fluid_state, solid_state, fluid_boundaries, solid_boundaries,
            interface_noslip=True, interface_radiation=True)

    fluid_grad_cv = grad_cv_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid)

    fluid_grad_t = grad_t_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid)

    solid_grad_cv = grad_cv_operator(
        dcoll, gas_model_solid, solid_all_boundaries_no_grad, solid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid)

    solid_grad_t = grad_t_operator(
        dcoll, gas_model_solid, solid_all_boundaries_no_grad, solid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid)

    tol = 1e-8

    check_fluid_grad_t = fluid_grad_t[0] - (2.0*a*fluid_nodes[0] + b)
    assert actx.to_numpy(
        op.norm(dcoll, check_fluid_grad_t/fluid_state.temperature, np.inf)) < tol
    assert actx.to_numpy(
        op.norm(dcoll, fluid_grad_t[1]/fluid_state.temperature, np.inf)) < tol
    assert actx.to_numpy(
        op.norm(dcoll, solid_grad_t[0]/solid_state.temperature, np.inf)) < tol
    assert actx.to_numpy(
        op.norm(dcoll, solid_grad_t[1]/solid_state.temperature, np.inf)) < tol

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

    if visualize:
        from grudge.shortcuts import make_visualizer
        viz_fluid = make_visualizer(dcoll, 2*order+1, volume_dd=dd_vol_fluid)
        viz_solid = make_visualizer(dcoll, 2*order+1, volume_dd=dd_vol_solid)
        if use_overintegration:
            viz_suffix = f"over_{order}"
        else:
            viz_suffix = f"{order}"

        i_O2 = cantera_soln.species_index("O2")  # noqa N806
        i_N2 = cantera_soln.species_index("N2")  # noqa N806

        viz_fluid.write_vtk_file(
            f"multiphysics_coupled_species_{viz_suffix}_fluid.vtu", [
                ("rho", fluid_state.cv.mass),
                ("rhoE", fluid_state.cv.energy),
                ("Y_O2", fluid_state.cv.species_mass_fractions[i_O2]),
                ("Y_N2", fluid_state.cv.species_mass_fractions[i_N2]),
                ("dv", fluid_state.dv),
                ("grad_T", fluid_grad_t),
                ], overwrite=True)
        viz_solid.write_vtk_file(
            f"multiphysics_coupled_species_{viz_suffix}_solid.vtu", [
                ("eps_rho", solid_state.cv.mass),
                ("eps_rhoE", solid_state.cv.energy),
                ("rho", solid_state.cv.mass/solid_state.wv.void_fraction),
                ("Y_O2", solid_state.cv.species_mass_fractions[i_O2]),
                ("Y_N2", solid_state.cv.species_mass_fractions[i_N2]),
                ("dv", solid_state.dv),
                ("grad_T", solid_grad_t),
                ], overwrite=True)
