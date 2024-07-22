"""Demonstrate multiple coupled volumes."""

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
import os
from functools import partial

import numpy as np
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    DISCR_TAG_QUAD,
    DOFDesc,
    VolumeDomainTag,
)
from grudge.op import nodal_max, nodal_min
from grudge.shortcuts import make_visualizer
from logpyle import IntervalTimer, set_dt
from meshmode.discretization.connection import FACE_RESTR_ALL  # noqa
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from mirgecom.boundary import (
    IsothermalWallBoundary,
)
from mirgecom.diffusion import NeumannDiffusionBoundary
from mirgecom.discretization import create_discretization_collection
from mirgecom.eos import IdealSingleGas
from mirgecom.euler import extract_vars_for_logging
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
)
from mirgecom.integrators import rk4_step
from mirgecom.io import make_init_message
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    logmgr_add_many_discretization_quantities,
    set_sim_state,
)
from mirgecom.mpi import mpi_entry_point
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    basic_coupled_ns_heat_operator as coupled_ns_heat_operator,
)
from mirgecom.simutil import (
    get_sim_timestep,
)
from mirgecom.steppers import advance_state
from mirgecom.transport import SimpleTransport
from pytools.obj_array import make_obj_array


logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, use_esdg=False, use_overintegration=False,
         use_leap=False, casename=None, rst_filename=None, use_tpe=True):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import actx_class_is_profiling, initialize_actx
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=True,
                           use_einsum_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    current_dt = 1e-8
    nsteps = 20
    t_final = nsteps * current_dt
    current_cfl = 1.0
    current_t = 0
    constant_cfl = False

    final_time_error = t_final/current_dt - np.around(t_final/current_dt)
    assert np.abs(final_time_error) < 1e-10, final_time_error

    # some i/o frequencies
    nstatus = 1
    nrestart = 500
    nviz = 10
    nhealth = 1

    dim = 2
    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    local_path = os.path.dirname(os.path.abspath(__file__))
    grid_file_name = local_path + "/multivolume_quads.msh"

    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        volume_to_local_mesh = restart_data["volume_to_local_mesh"]
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_ranks"] == num_ranks
    else:  # generate the grid from scratch
        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            # pylint: disable=unpacking-non-sequence
            mesh, tag_to_elements = read_gmsh(
                grid_file_name, force_ambient_dim=2,
                return_tag_to_elements_map=True,
                mesh_construction_kwargs={
                    "skip_tests": True,
                    "force_positive_orientation": True})
            volume_to_tags = {
                "Fluid": ["Upper"],
                "Wall": ["Lower"]}
            return mesh, tag_to_elements, volume_to_tags

        def partition_generator_func(mesh, tag_to_elements, num_ranks):
            # assert num_ranks == 2
            # rank_per_element = np.empty(mesh.nelements)
            # rank_per_element[tag_to_elements["Lower"]] = 0
            # rank_per_element[tag_to_elements["Upper"]] = 1
            # return rank_per_element
            from meshmode.distributed import get_partition_by_pymetis
            return get_partition_by_pymetis(mesh, num_ranks)

        from mirgecom.simutil import distribute_mesh
        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data, partition_generator_func)
        volume_to_local_mesh = {
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()}

    local_fluid_mesh = volume_to_local_mesh["Fluid"]
    local_wall_mesh = volume_to_local_mesh["Wall"]

    local_nelements = local_fluid_mesh.nelements + local_wall_mesh.nelements

    order = 1
    dcoll = create_discretization_collection(
        actx, volume_to_local_mesh, order=order,
        quadrature_order=order+2,
        tensor_product_elements=use_tpe)

    dd_vol_fluid = DOFDesc(VolumeDomainTag("Fluid"), DISCR_TAG_BASE)
    dd_vol_wall = DOFDesc(VolumeDomainTag("Wall"), DISCR_TAG_BASE)

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    wall_nodes = actx.thaw(dcoll.nodes(dd_vol_wall))

    fluid_ones = 0*fluid_nodes[0] + 1
    wall_ones = 0*wall_nodes[0] + 1

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        def extract_fluid_vars(dim, state, eos):
            cv = state[0]
            name_to_field = extract_vars_for_logging(dim, cv, eos)
            return {
                name + "_Fluid": field
                for name, field in name_to_field.items()}

        def units(quantity):
            return ""

        logmgr_add_many_discretization_quantities(
            logmgr, dcoll, dim, extract_fluid_vars, units, dd=dd_vol_fluid)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure_Fluid", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure_Fluid", "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    x_scale = 1

    gamma = 1.4
    r = 285.71300152552493
    mu = 4.216360056e-05/x_scale
    fluid_kappa = 0.05621788139856423*x_scale
    eos = IdealSingleGas(gamma=gamma, gas_const=r)
    transport = SimpleTransport(
        viscosity=mu,
        thermal_conductivity=fluid_kappa)
    gas_model = GasModel(eos=eos, transport=transport)

    fluid_pressure = 4935.22/x_scale
    fluid_temperature = 300
    fluid_density = fluid_pressure/fluid_temperature/r
    wall_density = fluid_density
    wall_heat_capacity = 50*eos.heat_capacity_cp()
    wall_kappa = 10*fluid_kappa*wall_ones

    wall_time_scale = 200

    isothermal_wall_temp = 300

    def smooth_step(actx, x, epsilon=1e-12):
        y = actx.np.minimum(actx.np.maximum(x, 0*x), 0*x+1)
        return (1 - actx.np.cos(np.pi*y))/2

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        current_wall_temperature = restart_data["wall_temperature"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        pressure = 4935.22/x_scale
        temperature = isothermal_wall_temp * fluid_ones
        sigma = 500/x_scale
        offset = 0
        smoothing = (
            fluid_ones
            * smooth_step(actx, sigma*(fluid_nodes[1]+offset))
            * smooth_step(actx, sigma*(-(fluid_nodes[1]-0.02*x_scale)+offset))
            * smooth_step(actx, sigma*(fluid_nodes[0]+0.02*x_scale+offset))
            * smooth_step(actx, sigma*(-(fluid_nodes[0]-0.02*x_scale)+offset)))
        temperature = (
            isothermal_wall_temp
            + (temperature - isothermal_wall_temp) * smoothing)
        mass = pressure/temperature/r * fluid_ones
        mom = make_obj_array([0*mass]*dim)
        energy = (pressure/(gamma - 1.0)) + np.dot(mom, mom)/(2.0*mass)
        current_cv = make_conserved(
            dim=dim,
            mass=mass,
            momentum=mom,
            energy=energy)

        current_wall_temperature = isothermal_wall_temp * wall_ones

    current_state = make_obj_array([current_cv, current_wall_temperature])

    fluid_boundaries = {
        dd_vol_fluid.trace("Upper Sides").domain_tag:  # pylint: disable=no-member
        IsothermalWallBoundary(wall_temperature=isothermal_wall_temp)}
    wall_boundaries = {
        dd_vol_wall.trace("Lower Sides").domain_tag:  # pylint: disable=no-member
        NeumannDiffusionBoundary(0)}

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    wall_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_wall)

    from grudge.dt_utils import characteristic_lengthscales
    wall_lengthscales = characteristic_lengthscales(actx, dcoll, dd=dd_vol_wall)

    initname = "thermally-coupled"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_get_timestep(step, t, state):
        fluid_state = make_fluid_state(state[0], gas_model)
        fluid_dt = get_sim_timestep(
            dcoll, fluid_state, t, current_dt, current_cfl, t_final,
            constant_cfl, fluid_dd=dd_vol_fluid)
        if constant_cfl:
            wall_alpha = (
                wall_time_scale
                * wall_kappa/(wall_density * wall_heat_capacity))
            wall_dt = actx.to_numpy(
                nodal_min(
                    dcoll, dd_vol_wall,
                    wall_lengthscales**2 * current_cfl/wall_alpha))[()]
        else:
            wall_dt = current_dt
        return min(fluid_dt, wall_dt)

    def my_write_status(step, t, dt, fluid_state, wall_temperature):
        dv = fluid_state.dv
        p_min = actx.to_numpy(nodal_min(dcoll, dd_vol_fluid, dv.pressure))
        p_max = actx.to_numpy(nodal_max(dcoll, dd_vol_fluid, dv.pressure))
        fluid_t_min = actx.to_numpy(nodal_min(dcoll, dd_vol_fluid, dv.temperature))
        fluid_t_max = actx.to_numpy(nodal_max(dcoll, dd_vol_fluid, dv.temperature))
        wall_t_min = actx.to_numpy(nodal_min(dcoll, dd_vol_wall, wall_temperature))
        wall_t_max = actx.to_numpy(nodal_max(dcoll, dd_vol_wall, wall_temperature))
        if constant_cfl:
            fluid_cfl = current_cfl
            wall_cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            fluid_cfl = actx.to_numpy(
                nodal_max(
                    dcoll, dd_vol_fluid, get_viscous_cfl(
                        dcoll, dt, fluid_state, dd=dd_vol_fluid)))
            wall_alpha = (
                wall_time_scale
                * wall_kappa/(wall_density * wall_heat_capacity))
            wall_cfl = actx.to_numpy(
                nodal_max(
                    dcoll, dd_vol_wall,
                    wall_alpha * dt/wall_lengthscales**2))
        if rank == 0:
            logger.info(f"Step: {step}, T: {t}, DT: {dt}\n"
                        f"----- Fluid CFL: {fluid_cfl}, Wall CFL: {wall_cfl}\n"
                        f"----- Fluid Pressure({p_min}, {p_max})\n"
                        f"----- Fluid Temperature({fluid_t_min}, {fluid_t_max})\n"
                        f"----- Wall Temperature({wall_t_min}, {wall_t_max})\n")

    def _construct_fluid_state(cv):
        return make_fluid_state(cv, gas_model=gas_model)

    construct_fluid_state = actx.compile(_construct_fluid_state)

    def my_write_viz(step, t, state, fluid_state=None):
        cv = state[0]
        wall_temperature = state[1]
        if fluid_state is None:
            fluid_state = construct_fluid_state(cv)
        dv = fluid_state.dv

        (
            fluid_rhs, wall_rhs, grad_cv, fluid_grad_temperature,
            wall_grad_temperature) = construct_rhs_and_gradients(t, state)

        fluid_viz_fields = [
            ("cv", cv),
            ("dv", dv),
            ("grad_cv_mass", grad_cv.mass),
            ("grad_cv_energy", grad_cv.energy),
            ("grad_cv_momentum_x", grad_cv.momentum[0]),
            ("grad_cv_momentum_y", grad_cv.momentum[1]),
            ("grad_t", fluid_grad_temperature),
            ("rhs", fluid_rhs),
            ("kappa", fluid_state.thermal_conductivity),
        ]
        wall_viz_fields = [
            ("temperature", wall_temperature),
            ("grad_t", wall_grad_temperature),
            ("rhs", wall_rhs),
            ("kappa", wall_kappa),
        ]
        from mirgecom.simutil import write_visfile
        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer, vizname=casename+"-fluid",
            step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)
        write_visfile(
            dcoll, wall_viz_fields, wall_visualizer, vizname=casename+"-wall",
            step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "volume_to_local_mesh": volume_to_local_mesh,
                "cv": state[0],
                "wall_temperature": state[1],
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_ranks": num_ranks
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, dd_vol_fluid, pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        # default health status bounds
        health_pres_min = 1.0e-1/x_scale
        health_pres_max = 2.0e6/x_scale

        if global_reduce(check_range_local(dcoll, dd_vol_fluid, pressure,
                                     health_pres_min, health_pres_max),
                                     op="lor"):
            health_error = True
            p_min = actx.to_numpy(nodal_min(dcoll, dd_vol_fluid, pressure))
            p_max = actx.to_numpy(nodal_max(dcoll, dd_vol_fluid, pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        # FIXME: Check wall state

        return health_error

    def my_pre_step(step, t, dt, state):
        fluid_state = make_fluid_state(state[0], gas_model)
        wall_temperature = state[1]
        dv = fluid_state.dv

        try:
            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_status = check_step(step=step, interval=nstatus)
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_status:
                my_write_status(
                    step=step, t=t, dt=dt, fluid_state=fluid_state,
                    wall_temperature=wall_temperature)

            if do_health:
                health_errors = global_reduce(my_health_check(dv.pressure), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state, fluid_state=fluid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state, fluid_state=fluid_state)
            my_write_restart(step=step, t=t, state=state)
            raise

        dt = my_get_timestep(step=step, t=t, state=state)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))

    def my_rhs(t, state, return_gradients=False):
        fluid_state = make_fluid_state(cv=state[0], gas_model=gas_model)
        wall_temperature = state[1]

        ns_heat_result = coupled_ns_heat_operator(
            dcoll,
            gas_model,
            dd_vol_fluid, dd_vol_wall,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_kappa, wall_temperature,
            time=t,
            quadrature_tag=quadrature_tag,
            use_esdg=use_esdg,
            return_gradients=return_gradients)

        if return_gradients:
            (
                fluid_rhs, wall_energy_rhs, fluid_grad_cv, fluid_grad_temperature,
                wall_grad_temperature) = ns_heat_result
        else:
            fluid_rhs, wall_energy_rhs = ns_heat_result

        wall_rhs = (
            wall_time_scale
            * wall_energy_rhs/(wall_density * wall_heat_capacity))
        from dataclasses import replace
        fluid_rhs = replace(
            fluid_rhs,
            energy=fluid_rhs.energy + (
                1e9
                * actx.np.exp(
                    -(fluid_nodes[0]**2+(fluid_nodes[1]-0.005)**2)/0.004**2)
                * actx.np.exp(-t/5e-6)))

        if return_gradients:
            return make_obj_array([
                fluid_rhs, wall_rhs, fluid_grad_cv, fluid_grad_temperature,
                wall_grad_temperature])
        else:
            return make_obj_array([fluid_rhs, wall_rhs])

    def my_rhs_and_gradients(t, state):
        return my_rhs(t, state, return_gradients=True)

    construct_rhs_and_gradients = actx.compile(my_rhs_and_gradients)

    current_dt = my_get_timestep(step=current_step, t=current_t, state=current_state)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      istep=current_step, state=current_state, t=current_t,
                      t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    my_write_viz(step=current_step, t=current_t, state=current_state)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "thermally-coupled"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--esdg", action="store_true",
        help="use entropy-stable operator")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from warnings import warn

    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_esdg=args.esdg,
         use_overintegration=args.overintegration or args.esdg,
         use_leap=args.leap, casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
