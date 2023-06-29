__copyright__ = """Copyright (C) 2022 University of Illinois Board of Trustees"""

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
from dataclasses import replace
from functools import partial
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath # noqa
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
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.boundary import (
    AdiabaticNoslipWallBoundary,
    IsothermalWallBoundary,
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
import pytest

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


def coupled_ns_heat_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        *,
        time=0.,
        quadrature_tag=DISCR_TAG_BASE):

    # Insert the interface boundaries for computing the gradient
    from mirgecom.multiphysics.thermally_coupled_fluid_wall import \
        add_interface_boundaries_no_grad
    fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad = \
        add_interface_boundaries_no_grad(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_kappa, wall_temperature,
            fluid_boundaries, wall_boundaries)

    # Get the operator fluid states
    from mirgecom.gas_model import make_operator_fluid_states
    fluid_operator_states_quad = make_operator_fluid_states(
        dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
        quadrature_tag, dd=fluid_dd)

    # Compute the temperature gradient for both subdomains
    from mirgecom.navierstokes import grad_t_operator as fluid_grad_t_operator
    from mirgecom.diffusion import grad_operator as wall_grad_t_operator
    fluid_grad_temperature = fluid_grad_t_operator(
        dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
        time=time, quadrature_tag=quadrature_tag,
        dd=fluid_dd, operator_states_quad=fluid_operator_states_quad)
    wall_grad_temperature = wall_grad_t_operator(
        dcoll, wall_kappa, wall_all_boundaries_no_grad, wall_temperature,
        quadrature_tag=quadrature_tag, dd=wall_dd)

    # Insert boundaries for the fluid-wall interface, now with the temperature
    # gradient
    from mirgecom.multiphysics.thermally_coupled_fluid_wall import \
        add_interface_boundaries
    fluid_all_boundaries, wall_all_boundaries = \
        add_interface_boundaries(
            dcoll,
            gas_model,
            fluid_dd, wall_dd,
            fluid_state, wall_kappa, wall_temperature,
            fluid_grad_temperature, wall_grad_temperature,
            fluid_boundaries, wall_boundaries)

    # Compute the subdomain NS/diffusion operators using the augmented boundaries
    from mirgecom.navierstokes import ns_operator
    from mirgecom.diffusion import diffusion_operator
    ns_result = ns_operator(
        dcoll, gas_model, fluid_state, fluid_all_boundaries,
        time=time, quadrature_tag=quadrature_tag, dd=fluid_dd,
        operator_states_quad=fluid_operator_states_quad,
        grad_t=fluid_grad_temperature)
    diffusion_result = diffusion_operator(
        dcoll, wall_kappa, wall_all_boundaries, wall_temperature,
        quadrature_tag=quadrature_tag, dd=wall_dd,
        grad_u=wall_grad_temperature)

    return ns_result, diffusion_result


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
                f"multiphysics_thermally_coupled_init_{viz_suffix}_fluid.vtu", [
                    ("temp_steady", fluid_temp_steady),
                    ("temp_perturb", fluid_temp_perturb),
                    ("temp_perturb_later", fluid_temp_perturb_later),
                    ("temp", fluid_temp),
                    ])
            viz_wall.write_vtk_file(
                f"multiphysics_thermally_coupled_init_{viz_suffix}_wall.vtu", [
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
                f"multiphysics_thermally_coupled_steady_{viz_suffix}_fluid.vtu", [
                    ("cv", fluid_state.cv),
                    ("dv", fluid_state.dv),
                    ("rhs", rhs[0]),
                    ])
            viz_wall.write_vtk_file(
                f"multiphysics_thermally_coupled_steady_{viz_suffix}_wall.vtu", [
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
                    "multiphysics_thermally_coupled_accuracy_"
                    f"{viz_suffix}_fluid_{step}.vtu", [
                        ("cv", fluid_state.cv),
                        ("dv", fluid_state.dv),
                        ("expected_temp", expected_fluid_temp),
                        ("rhs", rhs[0]),
                        ("momentum_source", momentum_source),
                        ])
                viz_wall.write_vtk_file(
                    "multiphysics_thermally_coupled_accuracy_"
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
