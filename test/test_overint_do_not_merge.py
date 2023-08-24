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
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.boundary import (
    AdiabaticNoslipWallBoundary,
    IsothermalWallBoundary
)
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
from mirgecom.materials.prescribed_porous_material import PrescribedMaterialEOS
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera


import logging
logger = logging.getLogger(__name__)


def enthalpy_func(temperature):
    return 1000.0*temperature


def heat_capacity_func(temperature):
    return 1000.0


def thermal_conductivity_func(temperature=None):
    return 0.1


def permeability_func(tau=None):
    return 0.0


def emissivity_func(tau=None):
    return 1.0


def tortuosity_func(tau=None):
    return 1.2


def volume_fraction_func(tau=None):
    return 0.1


def decomposition_progress_func(mass):
    return 1.0 + mass*0.0


@pytest.mark.parametrize("order", [1, 3, 5])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_my_sanity(
        actx_factory, order, use_overintegration, visualize=True):
    """Check the NS-coupled fluid/wall interface.

    Analytic solution prescribed as initial condition, then the RHS is assessed
    to ensure that it is nearly zero.
    """
    actx = actx_factory()

    dim = 2

    nelems = 15
    global_mesh = get_box_mesh(2, -1.0, 2.0, nelems)

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

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    dd_vol_fluid = DOFDesc(VolumeDomainTag("Fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("Solid"), DISCR_TAG_BASE)

    fluid_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_solid))

    # Use Cantera for initialization
    cantera_soln = cantera.Solution(name="gas",
                                    yaml=get_mechanism_input("air_3sp"))
    nspecies = cantera_soln.n_species
    y_species = np.zeros(nspecies)
    y_species[0] = 1.0

    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=500.0)

    # Gas model

    fluid_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.2,
        species_diffusivity=np.zeros(nspecies,))
    gas_model_fluid = GasModel(eos=eos, transport=fluid_transport)

    virgin_mass = 10.0

    my_material = PrescribedMaterialEOS(enthalpy_func, heat_capacity_func,
        thermal_conductivity_func, volume_fraction_func, permeability_func,
        emissivity_func, tortuosity_func, decomposition_progress_func)
    solid_density = virgin_mass + solid_nodes[0]*0.0

    base_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.1,
        species_diffusivity=np.zeros(nspecies,))
    solid_transport = PorousWallTransport(base_transport=base_transport)
    gas_model_solid = PorousFlowModel(eos=eos, transport=solid_transport,
                                      wall_eos=my_material)

    # Fluid cv

    grad_t_fluid_exact = (300.0 - 900.0)/(2.0)
    fluid_temperature = 900.0 + grad_t_fluid_exact*fluid_nodes[0]
    mass = 1.0 + 0.0*fluid_nodes[0]
    energy = mass*eos.get_internal_energy(fluid_temperature, y_species)
    mom_x = 0.0*fluid_nodes[0]
    mom_y = 0.0*fluid_nodes[0]
    momentum = make_obj_array([mom_x, mom_y])

    cv = make_conserved(dim=dim, mass=mass, energy=energy,
        momentum=momentum, species_mass=y_species*mass)
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model_fluid,
                                   temperature_seed=fluid_temperature)

    # Solid cv

    grad_t_solid_exact = (1500.0 - 900.0)/(-1.0)
    solid_temperature = 900.0 + grad_t_solid_exact*solid_nodes[0]

    sample_init = PorousWallInitializer(density=1.0,
        temperature=solid_temperature, species=y_species,
        material_densities=solid_density)

    wv = sample_init(2, solid_nodes, gas_model_solid)
    solid_state = make_fluid_state(cv=wv, gas_model=gas_model_solid,
        material_densities=solid_density, temperature_seed=solid_temperature)

    # Setup boundaries and interface

    base_fluid_temp = 300.0
    base_solid_temp = 1500.0

    fluid_boundaries = {
        dd_vol_fluid.trace("-2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+1").domain_tag:
            IsothermalWallBoundary(wall_temperature=base_fluid_temp),
    }

    solid_boundaries = {
        dd_vol_solid.trace("-2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("+2").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("-1").domain_tag:
            IsothermalWallBoundary(wall_temperature=base_solid_temp),
    }

    fluid_all_boundaries_no_grad, solid_all_boundaries_no_grad = \
        add_multiphysics_interface_bdries_no_grad(
            dcoll, dd_vol_fluid, dd_vol_solid,
            gas_model_fluid, gas_model_solid,
            fluid_state, solid_state,
            fluid_boundaries, solid_boundaries,
            interface_noslip=False, interface_radiation=False,
            use_kappa_weighted_grad_flux_in_fluid=False)

    from mirgecom.gas_model import make_operator_fluid_states
    fluid_operator_states_quad = make_operator_fluid_states(
        dcoll, fluid_state, gas_model_fluid, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=dd_vol_fluid)

    solid_operator_states_quad = make_operator_fluid_states(
        dcoll, solid_state, gas_model_solid, solid_all_boundaries_no_grad,
        quadrature_tag, dd=dd_vol_solid)

    fluid_grad_cv = grad_cv_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
        operator_states_quad=fluid_operator_states_quad)

    fluid_grad_temperature = grad_t_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
        operator_states_quad=fluid_operator_states_quad)

    solid_grad_cv = grad_cv_operator(
        dcoll, gas_model_solid, solid_all_boundaries_no_grad, solid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid,
        operator_states_quad=solid_operator_states_quad)

    solid_grad_temperature = grad_t_operator(
        dcoll, gas_model_solid, solid_all_boundaries_no_grad, solid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid,
        operator_states_quad=solid_operator_states_quad)

    fluid_all_boundaries, solid_all_boundaries = \
        add_multiphysics_interface_bdries(
            dcoll, dd_vol_fluid, dd_vol_solid,
            gas_model_fluid, gas_model_solid,
            fluid_state, solid_state,
            fluid_grad_cv, solid_grad_cv,
            fluid_grad_temperature, solid_grad_temperature,
            fluid_boundaries, solid_boundaries,
            interface_noslip=False, interface_radiation=False,
            use_kappa_weighted_grad_flux_in_fluid=False)

    fluid_rhs = ns_operator(
        dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
        grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
        operator_states_quad=fluid_operator_states_quad,
        inviscid_terms_on=False)

    solid_rhs = ns_operator(
        dcoll, gas_model_solid, solid_state, solid_all_boundaries,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid,
        grad_cv=solid_grad_cv, grad_t=solid_grad_temperature,
        operator_states_quad=solid_operator_states_quad,
        inviscid_terms_on=False)

    if visualize:
        from grudge.shortcuts import make_visualizer
        viz_fluid = make_visualizer(dcoll, 2*order+1, volume_dd=dd_vol_fluid)
        viz_solid = make_visualizer(dcoll, 2*order+1, volume_dd=dd_vol_solid)
        if use_overintegration:
            viz_suffix = f"over_{order}"
        else:
            viz_suffix = f"{order}"

        viz_fluid.write_vtk_file(
            f"multiphysics_coupled_temperature_{viz_suffix}_fluid.vtu", [
                ("cv", fluid_state.cv),
                ("dv", fluid_state.dv),
                ("grad_T", fluid_grad_temperature),
                ("rhs", fluid_rhs),
                ], overwrite=True)
        viz_solid.write_vtk_file(
            f"multiphysics_coupled_temperature_{viz_suffix}_solid.vtu", [
                ("cv", solid_state.cv),
                ("dv", solid_state.dv),
                ("grad_T", solid_grad_temperature),
                ("rhs", solid_rhs),
                ], overwrite=True)

    # Check gradients

    fluid_grad = \
        (fluid_grad_temperature[0] - grad_t_fluid_exact)/fluid_state.temperature
    solid_grad = \
        (solid_grad_temperature[0] - grad_t_solid_exact)/solid_state.temperature

    assert actx.to_numpy(op.norm(dcoll, fluid_grad, np.inf)) < 1e-10
    assert actx.to_numpy(op.norm(dcoll, solid_grad, np.inf)) < 1e-10

    # Check that steady-state solution has 0 RHS

    assert actx.to_numpy(op.norm(dcoll, fluid_rhs.energy/cv.energy, np.inf)) < 1e-6
    assert actx.to_numpy(op.norm(dcoll, solid_rhs.energy/wv.energy, np.inf)) < 1e-6
