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
from dataclasses import replace
from functools import partial
from pytools.obj_array import make_obj_array
import pymbolic as pmbl
import grudge.op as op
from mirgecom.symbolic import (
    grad as sym_grad,
    evaluate)
from mirgecom.simutil import max_component_norm
import mirgecom.math as mm
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from grudge.dof_desc import DOFDesc, VolumeDomainTag, DISCR_TAG_BASE, DISCR_TAG_QUAD
from mirgecom.discretization import create_discretization_collection
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.transport import SimpleTransport
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.boundary import (
    AdiabaticNoslipWallBoundary,
    IsothermalWallBoundary,
    DummyBoundary
)
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    basic_coupled_ns_heat_operator as coupled_ns_heat_operator,
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
from mirgecom.transport import PorousWallTransport
from mirgecom.wall_model import PorousFlowModel
from mirgecom.materials.initializer import PorousWallInitializer
from mirgecom.materials.prescribed_material import SolidProperties
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera


import logging
logger = logging.getLogger(__name__)


def get_box_mesh(dim, a, b, n):
    dim_names = ["x", "y", "z"]
    boundary_tag_to_face = {}
    for i in range(dim):
        boundary_tag_to_face["-"+str(i)] = ["-"+dim_names[i]]
        boundary_tag_to_face["+"+str(i)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh
    return generate_regular_rect_mesh(a=(a,)*dim, b=(b,)*dim,
        nelements_per_axis=(n,)*dim, boundary_tag_to_face=boundary_tag_to_face)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_independent_volumes(actx_factory, order, visualize=False):
    """Check multi-volume machinery by setting up two independent volumes."""
    actx = actx_factory()

    n = 8

    dim = 2

    dim_names = ["x", "y", "z"]
    boundary_tag_to_face = {}
    for i in range(dim):
        boundary_tag_to_face["-"+str(i)] = ["-"+dim_names[i]]
        boundary_tag_to_face["+"+str(i)] = ["+"+dim_names[i]]

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-1,)*dim, b=(1,)*dim,
        nelements_per_axis=(n,)*dim, boundary_tag_to_face=boundary_tag_to_face)

    volume_meshes = {
        "vol1": mesh,
        "vol2": mesh,
    }

    dcoll = create_discretization_collection(actx, volume_meshes, order)

    dd_vol1 = DOFDesc(VolumeDomainTag("vol1"), DISCR_TAG_BASE)
    dd_vol2 = DOFDesc(VolumeDomainTag("vol2"), DISCR_TAG_BASE)

    nodes1 = actx.thaw(dcoll.nodes(dd=dd_vol1))
    nodes2 = actx.thaw(dcoll.nodes(dd=dd_vol2))

    # Set solution to x for volume 1
    # Set solution to y for volume 2

    boundaries1 = {
        dd_vol1.trace("-0").domain_tag: DirichletDiffusionBoundary(-1.),
        dd_vol1.trace("+0").domain_tag: DirichletDiffusionBoundary(1.),
        dd_vol1.trace("-1").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol1.trace("+1").domain_tag: NeumannDiffusionBoundary(0.),
    }

    boundaries2 = {
        dd_vol2.trace("-0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol2.trace("+0").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol2.trace("-1").domain_tag: DirichletDiffusionBoundary(-1.),
        dd_vol2.trace("+1").domain_tag: DirichletDiffusionBoundary(1.),
    }

    u1 = nodes1[0]
    u2 = nodes2[1]

    u = make_obj_array([u1, u2])

    def get_rhs(t, u):
        return make_obj_array([
            diffusion_operator(
                dcoll, kappa=1, boundaries=boundaries1, u=u[0], dd=dd_vol1),
            diffusion_operator(
                dcoll, kappa=1, boundaries=boundaries2, u=u[1], dd=dd_vol2)])

    rhs = get_rhs(0, u)

    if visualize:
        from grudge.shortcuts import make_visualizer
        viz1 = make_visualizer(dcoll, order+3, volume_dd=dd_vol1)
        viz2 = make_visualizer(dcoll, order+3, volume_dd=dd_vol2)
        viz1.write_vtk_file(
            f"multiphysics_independent_volumes_{order}_1.vtu", [
                ("u", u[0]),
                ("rhs", rhs[0]),
                ])
        viz2.write_vtk_file(
            f"multiphysics_independent_volumes_{order}_2.vtu", [
                ("u", u[1]),
                ("rhs", rhs[1]),
                ])

    linf_err1 = actx.to_numpy(op.norm(dcoll, rhs[0], np.inf, dd=dd_vol1))
    linf_err2 = actx.to_numpy(op.norm(dcoll, rhs[1], np.inf, dd=dd_vol2))

    assert linf_err1 < 1e-9
    assert linf_err2 < 1e-9


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_thermally_coupled_fluid_wall(
        actx_factory, order, use_overintegration, visualize=False):
    """Check the thermally-coupled fluid/wall interface."""
    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc_rec_fluid = EOCRecorder()
    eoc_rec_wall = EOCRecorder()

    scales = [6, 8, 12]

    for n in scales:
        global_mesh = get_box_mesh(2, -1, 1, n)

        mgrp, = global_mesh.groups
        y = global_mesh.vertices[1, mgrp.vertex_indices]
        y_elem_avg = np.sum(y, axis=1)/y.shape[1]
        volume_to_elements = {
            "Fluid": np.where(y_elem_avg > 0)[0],
            "Wall": np.where(y_elem_avg < 0)[0]}

        from meshmode.mesh.processing import partition_mesh
        volume_meshes = partition_mesh(global_mesh, volume_to_elements)

        dcoll = create_discretization_collection(
            actx, volume_meshes, order=order, quadrature_order=2*order+1)

        if use_overintegration:
            quadrature_tag = DISCR_TAG_QUAD
        else:
            quadrature_tag = None

        dd_vol_fluid = DOFDesc(VolumeDomainTag("Fluid"), DISCR_TAG_BASE)
        dd_vol_wall = DOFDesc(VolumeDomainTag("Wall"), DISCR_TAG_BASE)

        if visualize:
            from grudge.shortcuts import make_visualizer
            viz_fluid = make_visualizer(dcoll, order+3, volume_dd=dd_vol_fluid)
            viz_wall = make_visualizer(dcoll, order+3, volume_dd=dd_vol_wall)
            if use_overintegration:
                viz_suffix = f"over_{order}_{n}"
            else:
                viz_suffix = f"{order}_{n}"

        fluid_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_fluid))
        wall_nodes = actx.thaw(dcoll.nodes(dd=dd_vol_wall))

        # Crank up the heat conduction so it's fast as possible within NS
        # timestep restriction
        heat_amplification_factor = 10000

        gamma = 1.4
        r = 285.71300152552493
        mu = 4.216360056e-05
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
        base_fluid_pressure = 4935.22
        base_fluid_temp = 300
        fluid_density = base_fluid_pressure/base_fluid_temp/r
        fluid_heat_capacity = eos.heat_capacity_cv()
        fluid_kappa = heat_amplification_factor * 0.05621788139856423
        transport = SimpleTransport(
            viscosity=mu,
            thermal_conductivity=fluid_kappa)
        gas_model = GasModel(eos=eos, transport=transport)

        # Made-up wall material
        wall_density = 10*fluid_density
        wall_heat_capacity = fluid_heat_capacity
        wall_kappa = 10*fluid_kappa

        base_wall_temp = 600

        fluid_boundaries = {
            dd_vol_fluid.trace("-0").domain_tag: AdiabaticNoslipWallBoundary(),
            dd_vol_fluid.trace("+0").domain_tag: AdiabaticNoslipWallBoundary(),
            dd_vol_fluid.trace("+1").domain_tag:
                IsothermalWallBoundary(wall_temperature=base_fluid_temp),
        }

        wall_boundaries = {
            dd_vol_wall.trace("-0").domain_tag: NeumannDiffusionBoundary(0.),
            dd_vol_wall.trace("+0").domain_tag: NeumannDiffusionBoundary(0.),
            dd_vol_wall.trace("-1").domain_tag:
                DirichletDiffusionBoundary(base_wall_temp),
        }

        interface_temp = (
            (fluid_kappa * base_fluid_temp + wall_kappa * base_wall_temp)
            / (fluid_kappa + wall_kappa))
        interface_flux = (
            -fluid_kappa * wall_kappa / (fluid_kappa + wall_kappa)
            * (base_fluid_temp - base_wall_temp))
        fluid_alpha = fluid_kappa/(fluid_density * fluid_heat_capacity)
        wall_alpha = wall_kappa/(wall_density * wall_heat_capacity)

        def steady_func(kappa, x, t):
            return interface_temp - interface_flux/kappa * x[1]

        fluid_steady_func = partial(steady_func, fluid_kappa)
        wall_steady_func = partial(steady_func, wall_kappa)

        def perturb_func(alpha, x, t):
            w = 1.5 * np.pi
            return 50 * mm.cos(w * x[1]) * mm.exp(-w**2 * alpha * t)

        # This perturbation function is nonzero at the interface, so the two alphas
        # need to be the same (otherwise the perturbations will decay at different
        # rates and a discontinuity will form)
        assert abs(fluid_alpha - wall_alpha) < 1e-12

        fluid_perturb_func = partial(perturb_func, fluid_alpha)
        wall_perturb_func = partial(perturb_func, wall_alpha)

        def fluid_func(x, t):
            return fluid_steady_func(x, t) + fluid_perturb_func(x, t)

        def wall_func(x, t):
            return wall_steady_func(x, t) + wall_perturb_func(x, t)

        if visualize:
            fluid_temp_steady = fluid_steady_func(fluid_nodes, 0)
            fluid_temp_perturb = fluid_perturb_func(fluid_nodes, 0)
            fluid_temp_perturb_later = fluid_perturb_func(fluid_nodes, 5)
            fluid_temp = fluid_func(fluid_nodes, 0)
            wall_temp_steady = wall_steady_func(wall_nodes, 0)
            wall_temp_perturb = wall_perturb_func(wall_nodes, 0)
            wall_temp_perturb_later = wall_perturb_func(wall_nodes, 5)
            wall_temp = wall_func(wall_nodes, 0)
            viz_fluid.write_vtk_file(
                f"thermally_coupled_init_{viz_suffix}_fluid.vtu", [
                    ("temp_steady", fluid_temp_steady),
                    ("temp_perturb", fluid_temp_perturb),
                    ("temp_perturb_later", fluid_temp_perturb_later),
                    ("temp", fluid_temp),
                    ])
            viz_wall.write_vtk_file(
                f"thermally_coupled_init_{viz_suffix}_wall.vtu", [
                    ("temp_steady", wall_temp_steady),
                    ("temp_perturb", wall_temp_perturb),
                    ("temp_perturb_later", wall_temp_perturb_later),
                    ("temp", wall_temp),
                    ])

        # Add a source term to the momentum equations to cancel out the pressure term
        sym_fluid_temp = fluid_func(pmbl.var("x"), pmbl.var("t"))
        sym_fluid_pressure = fluid_density * r * sym_fluid_temp
        sym_momentum_source = sym_grad(2, sym_fluid_pressure)

        def momentum_source_func(x, t):
            return evaluate(sym_momentum_source, x=x, t=t)

        def get_rhs(t, state):
            fluid_state = make_fluid_state(cv=state[0], gas_model=gas_model)
            wall_temperature = state[1]
            fluid_rhs, wall_energy_rhs = coupled_ns_heat_operator(
                dcoll,
                gas_model,
                dd_vol_fluid, dd_vol_wall,
                fluid_boundaries, wall_boundaries,
                fluid_state, wall_kappa, wall_temperature,
                time=t,
                quadrature_tag=quadrature_tag)
            fluid_rhs = replace(
                fluid_rhs,
                momentum=fluid_rhs.momentum + momentum_source_func(fluid_nodes, t))
            wall_rhs = wall_energy_rhs / (wall_density * wall_heat_capacity)
            return make_obj_array([fluid_rhs, wall_rhs])

        def cv_from_temp(temp):
            rho = fluid_density * (0*temp + 1)
            mom = make_obj_array([0*temp]*2)
            energy = (
                (rho * r * temp)/(gamma - 1.0)
                + np.dot(mom, mom)/(2.0*rho))
            return make_conserved(
                dim=2,
                mass=rho,
                momentum=mom,
                energy=energy)

        # Check that steady-state solution has 0 RHS

        t_large = 1e6
        fluid_temp = fluid_func(fluid_nodes, t_large)
        wall_temp = wall_func(wall_nodes, t_large)

        state = make_obj_array([cv_from_temp(fluid_temp), wall_temp])

        rhs = get_rhs(t_large, state)

        if visualize:
            fluid_state = make_fluid_state(state[0], gas_model)
            viz_fluid.write_vtk_file(
                f"thermally_coupled_steady_{viz_suffix}_fluid.vtu", [
                    ("cv", fluid_state.cv),
                    ("dv", fluid_state.dv),
                    ("rhs", rhs[0]),
                    ])
            viz_wall.write_vtk_file(
                f"thermally_coupled_steady_{viz_suffix}_wall.vtu", [
                    ("temp", state[1]),
                    ("rhs", rhs[1]),
                    ])

        fluid_cv = cv_from_temp(fluid_temp)
        linf_err_fluid = max_component_norm(
            dcoll,
            rhs[0]/replace(fluid_cv, momentum=0*fluid_cv.momentum+1),
            np.inf,
            dd=dd_vol_fluid)
        linf_err_wall = actx.to_numpy(
            op.norm(dcoll, rhs[1], np.inf, dd=dd_vol_wall)
            / op.norm(dcoll, wall_temp, np.inf, dd=dd_vol_wall))

        assert linf_err_fluid < 1e-6
        assert linf_err_wall < 1e-6

        # Now check accuracy/stability

        fluid_temp = fluid_func(fluid_nodes, 0)
        wall_temp = wall_func(wall_nodes, 0)

        state = make_obj_array([cv_from_temp(fluid_temp), wall_temp])

        from grudge.dt_utils import characteristic_lengthscales
        h_min_fluid = actx.to_numpy(
            op.nodal_min(
                dcoll, dd_vol_fluid,
                characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)))[()]
        h_min_wall = actx.to_numpy(
            op.nodal_min(
                dcoll, dd_vol_wall,
                characteristic_lengthscales(actx, dcoll, dd=dd_vol_wall)))[()]

        # Set dt once for all scales
        if n == scales[0]:
            dt = 0.00025 * min(h_min_fluid**2, h_min_wall**2)

        heat_cfl_fluid = fluid_alpha * dt/h_min_fluid**2
        heat_cfl_wall = wall_alpha * dt/h_min_wall**2

        print(f"{heat_cfl_fluid=}, {heat_cfl_wall=}")
        assert heat_cfl_fluid < 0.05
        assert heat_cfl_wall < 0.05

        from mirgecom.integrators import rk4_step

        t = 0
        for step in range(50):
            state = rk4_step(state, t, dt, get_rhs)
            t += dt
            if step % 5 == 0 and visualize:
                fluid_state = make_fluid_state(state[0], gas_model)
                expected_fluid_temp = fluid_func(fluid_nodes, t)
                expected_wall_temp = wall_func(wall_nodes, t)
                rhs = get_rhs(t, state)
                momentum_source = momentum_source_func(fluid_nodes, t)
                viz_fluid.write_vtk_file(
                    "thermally_coupled_accuracy_"
                    f"{viz_suffix}_fluid_{step}.vtu", [
                        ("cv", fluid_state.cv),
                        ("dv", fluid_state.dv),
                        ("expected_temp", expected_fluid_temp),
                        ("rhs", rhs[0]),
                        ("momentum_source", momentum_source),
                        ])
                viz_wall.write_vtk_file(
                    "thermally_coupled_accuracy_"
                    f"{viz_suffix}_wall_{step}.vtu", [
                        ("temp", state[1]),
                        ("expected_temp", expected_wall_temp),
                        ("rhs", rhs[1]),
                        ])

        fluid_state = make_fluid_state(state[0], gas_model)
        fluid_temp = fluid_state.dv.temperature
        wall_temp = state[1]
        expected_fluid_temp = fluid_func(fluid_nodes, t)
        expected_wall_temp = wall_func(wall_nodes, t)

        assert np.isfinite(
            actx.to_numpy(op.norm(dcoll, fluid_temp, 2, dd=dd_vol_fluid)))
        assert np.isfinite(
            actx.to_numpy(op.norm(dcoll, wall_temp, 2, dd=dd_vol_wall)))

        linf_err_fluid = actx.to_numpy(
            op.norm(dcoll, fluid_temp - expected_fluid_temp, np.inf, dd=dd_vol_fluid)
            / op.norm(dcoll, expected_fluid_temp, np.inf, dd=dd_vol_fluid))
        linf_err_wall = actx.to_numpy(
            op.norm(dcoll, wall_temp - expected_wall_temp, np.inf, dd=dd_vol_wall)
            / op.norm(dcoll, expected_wall_temp, np.inf, dd=dd_vol_wall))
        eoc_rec_fluid.add_data_point(1/n, linf_err_fluid)
        eoc_rec_wall.add_data_point(1/n, linf_err_wall)

    print("L^inf error (fluid):")
    print(eoc_rec_fluid)
    print("L^inf error (wall):")
    print(eoc_rec_wall)

    assert (
        eoc_rec_fluid.order_estimate() >= order - 0.5
        or eoc_rec_fluid.max_error() < 1e-11)
    assert (
        eoc_rec_wall.order_estimate() >= order - 0.5
        or eoc_rec_wall.max_error() < 1e-11)


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_thermally_coupled_fluid_wall_with_radiation(
        actx_factory, order, use_overintegration, visualize=False):
    """Check the thermally-coupled fluid/wall interface with radiation.

    Analytic solution prescribed as initial condition, then the RHS is assessed
    to ensure that it is nearly zero.
    """
    actx = actx_factory()

    dim = 2
    nelems = 48

    global_mesh = get_box_mesh(dim, -2, 1, nelems)

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

    eos = IdealSingleGas()
    transport = SimpleTransport(viscosity=0.0, thermal_conductivity=10.0)
    gas_model = GasModel(eos=eos, transport=transport)

    # Fluid cv

    mom_x = 0.0*fluid_nodes[0]
    mom_y = 0.0*fluid_nodes[0]
    momentum = make_obj_array([mom_x, mom_y])

    temperature = 2.0 - (2.0 - 1.33938)*(1.0 - fluid_nodes[0])/1.0
    mass = 1.0 + 0.0*fluid_nodes[0]
    energy = mass*eos.get_internal_energy(temperature)

    fluid_cv = make_conserved(dim=dim, mass=mass, energy=energy,
                              momentum=momentum)

    fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model)

    # Made-up wall material
    wall_rho = 1.0
    wall_cp = 1000.0
    wall_kappa = 1.0
    wall_emissivity = 1.0

    base_fluid_temp = 2.0
    base_solid_temp = 1.0

    fluid_boundaries = {
        dd_vol_fluid.trace("-1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+0").domain_tag:
            IsothermalWallBoundary(wall_temperature=base_fluid_temp),
    }

    solid_boundaries = {
        dd_vol_solid.trace("-1").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol_solid.trace("+1").domain_tag: NeumannDiffusionBoundary(0.),
        dd_vol_solid.trace("-0").domain_tag:
            DirichletDiffusionBoundary(base_solid_temp),
    }

    wall_temperature = 1.0 + (1.0 - 1.33938)*(2.0 + solid_nodes[0])/(-2.0)

    fluid_rhs, wall_energy_rhs = coupled_ns_heat_operator(
        dcoll, gas_model, dd_vol_fluid, dd_vol_solid, fluid_boundaries,
        solid_boundaries, fluid_state, wall_kappa, wall_temperature,
        time=0.0, quadrature_tag=quadrature_tag, interface_noslip=False,
        interface_radiation=True, sigma=2.0,
        ambient_temperature=0.0, wall_emissivity=wall_emissivity,
    )

    # Check that steady-state solution has 0 RHS

    fluid_rhs = fluid_rhs.energy
    solid_rhs = wall_energy_rhs/(wall_cp*wall_rho)

    assert actx.to_numpy(op.norm(dcoll, fluid_rhs, np.inf)) < 1e-4
    assert actx.to_numpy(op.norm(dcoll, solid_rhs, np.inf)) < 1e-4


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
    return 1.0


def volume_fraction_func(tau=None):
    return 0.1


def decomposition_progress_func(mass):
    return 1.0 + mass*0.0


@pytest.mark.parametrize("order", [1, 3, 5])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_multiphysics_coupled_fluid_wall_with_radiation(
        actx_factory, order, use_overintegration, visualize=False):
    """Check the thermally-coupled fluid/wall interface with radiation.

    Analytic solution prescribedas initial condition, then the RHS is assessed
    to ensure that it is nearly zero.
    """
    actx = actx_factory()

    dim = 2

    nelems = 15
    global_mesh = get_box_mesh(2, -0.1, 0.2, nelems)

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
                                    yaml=get_mechanism_input("inert"))
    nspecies = cantera_soln.n_species
    y_species = np.zeros(nspecies)
    y_species[cantera_soln.species_index("Ar")] = 1.0

    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=500.0)

    # Gas model

    fluid_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.2,
        species_diffusivity=np.zeros(nspecies,))
    gas_model_fluid = GasModel(eos=eos, transport=fluid_transport)

    virgin_mass = 10.0

    my_material = SolidProperties(enthalpy_func, heat_capacity_func,
        thermal_conductivity_func, volume_fraction_func, permeability_func,
        emissivity_func, tortuosity_func, decomposition_progress_func)
    wall_density = virgin_mass + solid_nodes[0]*0.0

    base_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.1,
        species_diffusivity=np.zeros(nspecies,))
    solid_transport = PorousWallTransport(base_transport=base_transport)
    gas_model_solid = PorousFlowModel(eos=eos, transport=solid_transport,
        wall_model=my_material)

    # Fluid cv

    delta_temperature = (300.0 - 400.4685405683)/(0.2)
    temperature = 400.4685405683 + delta_temperature*fluid_nodes[0]
    mass = 1.0 + 0.0*fluid_nodes[0]
    energy = mass*eos.get_internal_energy(temperature, y_species)
    mom_x = 0.0*fluid_nodes[0]
    mom_y = 0.0*fluid_nodes[0]
    momentum = make_obj_array([mom_x, mom_y])

    cv = make_conserved(dim=dim, mass=mass, energy=energy,
        momentum=momentum, species_mass=y_species*mass)
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model_fluid,
                                   temperature_seed=500.0)

    # Solid cv

    delta_temperature = (1500.0 - 400.4685405683)/(-0.1)
    wall_temperature = 400.4685405683 + delta_temperature*solid_nodes[0]

    sample_init = PorousWallInitializer(density=1.0,
        temperature=wall_temperature, species=y_species,
        material_densities=wall_density)

    wv = sample_init(2, solid_nodes, gas_model_solid)
    wall_state = make_fluid_state(cv=wv, gas_model=gas_model_solid,
        material_densities=wall_density, temperature_seed=wall_temperature)

    # Setup boundaries and interface

    base_fluid_temp = 300.0
    base_solid_temp = 1500.0

    fluid_boundaries = {
        dd_vol_fluid.trace("-1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+0").domain_tag:
            IsothermalWallBoundary(wall_temperature=base_fluid_temp),
    }

    solid_boundaries = {
        dd_vol_solid.trace("-1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("+1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("-0").domain_tag:
            IsothermalWallBoundary(wall_temperature=base_solid_temp),
    }

    fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad = \
        add_multiphysics_interface_bdries_no_grad(
            dcoll, dd_vol_fluid, dd_vol_solid,
            gas_model_fluid, gas_model_solid,
            fluid_state, wall_state,
            fluid_boundaries, solid_boundaries,
            interface_noslip=False, interface_radiation=True)

    fluid_grad_cv = grad_cv_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid)

    fluid_grad_temperature = grad_t_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid)

    wall_grad_cv = grad_cv_operator(
        dcoll, gas_model_solid, wall_all_boundaries_no_grad, wall_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid)

    wall_grad_temperature = grad_t_operator(
        dcoll, gas_model_solid, wall_all_boundaries_no_grad, wall_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid)

    fluid_all_boundaries, wall_all_boundaries = \
        add_multiphysics_interface_bdries(
            dcoll, dd_vol_fluid, dd_vol_solid,
            gas_model_fluid, gas_model_solid,
            fluid_state, wall_state,
            fluid_grad_cv, wall_grad_cv,
            fluid_grad_temperature, wall_grad_temperature,
            fluid_boundaries, solid_boundaries,
            interface_noslip=False, interface_radiation=True,
            wall_emissivity=1.0, sigma=5.67e-8, ambient_temperature=300.0)

    fluid_rhs = ns_operator(
        dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
        grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
        inviscid_terms_on=False)

    solid_rhs = ns_operator(
        dcoll, gas_model_solid, wall_state, wall_all_boundaries,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid,
        grad_cv=wall_grad_cv, grad_t=wall_grad_temperature,
        inviscid_terms_on=False)

    if visualize:
        from grudge.shortcuts import make_visualizer
        viz_fluid = make_visualizer(dcoll, order+3, volume_dd=dd_vol_fluid)
        viz_wall = make_visualizer(dcoll, order+3, volume_dd=dd_vol_solid)
        if use_overintegration:
            viz_suffix = f"over_{order}"
        else:
            viz_suffix = f"{order}"

        viz_fluid.write_vtk_file(
            f"multiphysics_coupled_radiation_{viz_suffix}_fluid.vtu", [
                ("cv", fluid_state.cv),
                ("dv", fluid_state.dv),
                ("grad_T", fluid_grad_temperature),
                ("rhs", fluid_rhs),
                ], overwrite=True)
        viz_wall.write_vtk_file(
            f"multiphysics_coupled_radiation_{viz_suffix}_wall.vtu", [
                ("cv", wall_state.cv),
                ("dv", wall_state.dv),
                ("rhs", solid_rhs),
                ], overwrite=True)

    # Check that steady-state solution has 0 RHS

    fluid_rhs = fluid_rhs.energy/cv.energy
    solid_rhs = solid_rhs.energy/wv.energy

    assert actx.to_numpy(op.norm(dcoll, fluid_rhs, np.inf)) < 1e-6
    assert actx.to_numpy(op.norm(dcoll, solid_rhs, np.inf)) < 1e-6


@pytest.mark.parametrize("order", [1, 3, 5])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_multiphysics_coupled_fluid_wall_for_species_diffusion(
        actx_factory, order, use_overintegration, visualize=False):

    """Check the thermally-coupled fluid/wall interface with radiation.

    Analytic solution prescribedas initial condition, then the RHS is assessed
    to ensure that it is nearly zero.
    """
    actx = actx_factory()

    dim = 2

    nelems = 30
    global_mesh = get_box_mesh(2, -1, 1, nelems)

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

    # Pyrometheus initialization
    import cantera
    from mirgecom.mechanisms import get_mechanism_input
    from mirgecom.thermochemistry import \
        get_pyrometheus_wrapper_class_from_cantera
    mech_input = get_mechanism_input("inert")
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species

    eos = PyrometheusMixture(pyro_obj, temperature_guess=300.0)

    # Gas model

    fluid_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.0,
        species_diffusivity=np.zeros(nspecies,) + 0.003)
    gas_model_fluid = GasModel(eos=eos, transport=fluid_transport)

    virgin_mass = 10.0

    my_material = SolidProperties(enthalpy_func, heat_capacity_func,
        thermal_conductivity_func, volume_fraction_func, permeability_func,
        emissivity_func, tortuosity_func, decomposition_progress_func)
    wall_density = virgin_mass + solid_nodes[0]*0.0

    base_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.0,
        species_diffusivity=np.zeros(nspecies,) + 0.001)
    solid_transport = PorousWallTransport(base_transport=base_transport)
    gas_model_solid = PorousFlowModel(eos=eos, transport=solid_transport,
        wall_model=my_material)

    fluid_diffusivity = 0.003
    solid_diffusivity = 0.001/tortuosity_func()

    diff_ratio = fluid_diffusivity/solid_diffusivity
    y_species_interface = 1.0/(1.0 + diff_ratio)

    # Fluid cv

    zeros = fluid_nodes[0]*0.0
    y = make_obj_array([zeros, zeros, zeros])
    y[0] = 1.0 - (y_species_interface)*(1.0 - fluid_nodes[0])
    y[1] = 1.0 - y[0]
    y[2] = y[0]*0.0

    temperature = 300.0 + zeros
    mass = 1.0 + zeros
    energy = mass*eos.get_internal_energy(temperature, y)
    momentum = make_obj_array([0.0*fluid_nodes[0], 0.0*fluid_nodes[0]])
    species_mass = mass*y

    cv = make_conserved(dim=dim, mass=mass, energy=energy, momentum=momentum,
        species_mass=species_mass)
    fluid_state = make_fluid_state(cv=cv, gas_model=gas_model_fluid,
                                   temperature_seed=temperature)

    # Solid cv

    zeros = solid_nodes[0]*0.0
    y = make_obj_array([zeros, zeros, zeros])
    y[0] = (1.0 - y_species_interface)*(solid_nodes[0] + 1.0)
    y[1] = 1.0 - y[0]
    y[2] = y[0]*0.0

    wall_density = 10.0 + zeros
    sample_init = PorousWallInitializer(density=1.0, temperature=300.0,
        species=y, material_densities=wall_density)

    wv = sample_init(2, solid_nodes, gas_model_solid)
    wall_state = make_fluid_state(cv=wv, gas_model=gas_model_solid,
        material_densities=wall_density, temperature_seed=300.0)

    # Setup boundaries and interface

    fluid_boundaries = {
        dd_vol_fluid.trace("-1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("+0").domain_tag: DummyBoundary(),
    }

    solid_boundaries = {
        dd_vol_solid.trace("-1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("+1").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_solid.trace("-0").domain_tag: DummyBoundary(),
    }

    fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad = \
        add_multiphysics_interface_bdries_no_grad(
            dcoll, dd_vol_fluid, dd_vol_solid,
            gas_model_fluid, gas_model_solid,
            fluid_state, wall_state,
            fluid_boundaries, solid_boundaries,
            interface_noslip=True, interface_radiation=False)

    fluid_grad_cv = grad_cv_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid)

    fluid_grad_temperature = grad_t_operator(
        dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid)

    wall_grad_cv = grad_cv_operator(
        dcoll, gas_model_solid, wall_all_boundaries_no_grad, wall_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid)

    wall_grad_temperature = grad_t_operator(
        dcoll, gas_model_solid, wall_all_boundaries_no_grad, wall_state,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid)

    fluid_all_boundaries, wall_all_boundaries = \
        add_multiphysics_interface_bdries(
            dcoll, dd_vol_fluid, dd_vol_solid,
            gas_model_fluid, gas_model_solid,
            fluid_state, wall_state,
            fluid_grad_cv, wall_grad_cv,
            fluid_grad_temperature, wall_grad_temperature,
            fluid_boundaries, solid_boundaries,
            interface_noslip=True, interface_radiation=False)

    fluid_rhs = ns_operator(
        dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries,
        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
        grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
        inviscid_terms_on=False)

    solid_rhs = ns_operator(
        dcoll, gas_model_solid, wall_state, wall_all_boundaries,
        quadrature_tag=quadrature_tag, dd=dd_vol_solid,
        grad_cv=wall_grad_cv, grad_t=wall_grad_temperature,
        inviscid_terms_on=False)

    if visualize:
        from grudge.shortcuts import make_visualizer
        viz_fluid = make_visualizer(dcoll, order+3, volume_dd=dd_vol_fluid)
        viz_wall = make_visualizer(dcoll, order+3, volume_dd=dd_vol_solid)
        if use_overintegration:
            viz_suffix = f"over_{order}"
        else:
            viz_suffix = f"{order}"

        viz_fluid.write_vtk_file(
            f"multiphysics_coupled_species_{viz_suffix}_fluid.vtu", [
                ("rho", fluid_state.cv.mass),
                ("Y_0", fluid_state.cv.species_mass_fractions[0]),
                ("Y_1", fluid_state.cv.species_mass_fractions[1]),
                ("Y_2", fluid_state.cv.species_mass_fractions[2]),
                ("dv", fluid_state.dv),
                ("rhs", fluid_rhs),
                ], overwrite=True)
        viz_wall.write_vtk_file(
            f"multiphysics_coupled_species_{viz_suffix}_wall.vtu", [
                ("eps_rho", wall_state.cv.mass),
                ("rho", wall_state.cv.mass/wall_state.wv.void_fraction),
                ("Y_0", wall_state.cv.species_mass_fractions[0]),
                ("Y_1", wall_state.cv.species_mass_fractions[1]),
                ("Y_2", wall_state.cv.species_mass_fractions[2]),
                ("dv", wall_state.dv),
                ("rhs", solid_rhs),
                ], overwrite=True)

    # Check that steady-state solution has 0 RHS

    for i in range(0, nspecies):
        assert actx.to_numpy(
            op.norm(dcoll, fluid_rhs.species_mass[i], np.inf)) < 1e-10
        assert actx.to_numpy(
            op.norm(dcoll, solid_rhs.species_mass[i], np.inf)) < 1e-10


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
