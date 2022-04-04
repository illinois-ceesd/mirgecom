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
from mirgecom.mpi import mpi_entry_point
import numpy as np
from functools import partial
from pytools.obj_array import make_obj_array
import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.discretization.connection import FACE_RESTR_ALL  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.discretization import make_discretization_collection
from grudge.shortcuts import make_visualizer

from grudge.trace_pair import (
    inter_volume_trace_pairs,
)
from grudge.dof_desc import (
    VolumeDomainTag,
    DISCR_TAG_BASE,
    DISCR_TAG_QUAD,
    DOFDesc,
)
from mirgecom.navierstokes import (
    ns_operator,
    grad_t_operator as fluid_grad_t_operator
)
from mirgecom.diffusion import (
    diffusion_operator,
    NeumannDiffusionBoundary,
    InterfaceDiffusionBoundary,
    grad_operator as wall_grad_t_operator
)
from mirgecom.simutil import (
    get_sim_timestep,
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    IsothermalNoSlipBoundary,
    TemperatureCoupledNoSlipBoundary
)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    set_sim_state
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_overintegration=False,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=None, lazy=False):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    t_final = 1
    current_cfl = 1.0
#     current_dt = 1e-7  # Max dt for isothermal-only, tanh, 4x
#     current_dt = 1e-7  # Max dt for isothermal-only, tanh, 8x
#     current_dt = 5e-9  # Max dt for coupled, tanh, 4x
#     current_dt = 5e-8  # Max dt for coupled, bump, 8x
    current_dt = 5e-8
    current_t = 0
    constant_cfl = False

    final_time_error = t_final/current_dt - np.around(t_final/current_dt)
    assert np.abs(final_time_error) < 1e-10, final_time_error

    # some i/o frequencies
    nstatus = 1
    nrestart = 100
    nviz = 1
    nhealth = 1

    dim = 2
    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_fluid_mesh = restart_data["local_fluid_mesh"]
        local_wall_mesh = restart_data["local_fluid_mesh"]
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == num_parts
    else:  # generate the grid from scratch
        from meshmode.mesh.io import read_gmsh
        mesh = read_gmsh("multivolume.msh", force_ambient_dim=2)

        global_nelements = mesh.nelements

        volume_tags = ["Fluid", "Wall"]

        volume_tag_to_mesh_tags = {
            "Fluid": ["Upper"],
            "Wall": ["Lower"],
        }

        volume_index_per_element = np.empty(mesh.nelements, dtype=int)
        for vtag, mesh_tags in volume_tag_to_mesh_tags.items():
            vol_idx = volume_tags.index(vtag)
            for vgrp in mesh.volume_groups[0]:
                if vgrp.volume_tag in mesh_tags:
                    volume_index_per_element[vgrp.elements] = vol_idx

        from meshmode.distributed import get_partition_by_pymetis
        rank_per_element = get_partition_by_pymetis(mesh, num_parts)

        part_id_to_elements = {
            (rank, volume_tags[vol_idx]):
                np.where(
                    (volume_index_per_element == vol_idx)
                    & (rank_per_element == rank))[0]
            for vol_idx in range(len(volume_tags))
            for rank in range(num_parts)}

        from meshmode.mesh.processing import partition_mesh
        part_id_to_mesh = partition_mesh(mesh, part_id_to_elements)

        rank_to_meshes = {
            rank: (
                part_id_to_mesh[rank, "Fluid"],
                part_id_to_mesh[rank, "Wall"])
            for rank in range(num_parts)}

        from meshmode.distributed import mpi_distribute
        local_fluid_mesh, local_wall_mesh = mpi_distribute(comm, rank_to_meshes)

        del mesh

    local_nelements = local_fluid_mesh.nelements + local_wall_mesh.nelements

    from meshmode.discretization.poly_element import \
        default_simplex_group_factory, QuadratureSimplexGroupFactory

    order = 3
    discr = make_discretization_collection(
        actx,
        volumes={
            "Fluid": local_fluid_mesh,
            "Wall": local_wall_mesh
        },
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
        },
        _result_type=EagerDGDiscretization)

    dd_fluid_vol = DOFDesc(VolumeDomainTag("Fluid"), DISCR_TAG_BASE)
    dd_wall_vol = DOFDesc(VolumeDomainTag("Wall"), DISCR_TAG_BASE)

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(
            logmgr, discr, dim, extract_vars_for_logging, units_for_logging,
            volume_dd=dd_fluid_vol)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
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
#     wall_density = 1/x_scale**3
#     wall_heat_capacity = 1*x_scale**2
#     wall_kappa = 247.5/(1625*770)
    wall_density = 1625/x_scale**3
    wall_heat_capacity = 770.*x_scale**2
    wall_kappa = 247.5*x_scale
#     wall_kappa = 20*x_scale
#     wall_kappa = fluid_kappa







    wall_heat_capacity = wall_heat_capacity/500
    wall_kappa = wall_kappa/100








    fluid_alpha = fluid_kappa/(2.622e-2/x_scale**3 * eos.heat_capacity_cp())
    wall_alpha = wall_kappa/(wall_density * wall_heat_capacity)
    print(f"{fluid_alpha=}, {wall_alpha=}")
    from grudge.op import nodal_min
    from grudge.dt_utils import characteristic_lengthscales
    h = actx.np.minimum(
        nodal_min(discr, dd_fluid_vol,
            characteristic_lengthscales(actx, discr, dd=dd_fluid_vol)),
        nodal_min(discr, dd_wall_vol,
            characteristic_lengthscales(actx, discr, dd=dd_wall_vol)))
    print(f"{h=}")
    heat_cfl_fluid = actx.to_numpy(fluid_alpha * current_dt/h**2)[()]
    heat_cfl_wall = actx.to_numpy(wall_alpha * current_dt/h**2)[()]
    print(f"{heat_cfl_fluid=}, {heat_cfl_wall=}")
    isothermal_wall_temp = 300

    def smooth_step_tanh(actx, x, epsilon=1e-12):
        # return actx.np.tanh(actx.np.abs(x))
        return actx.np.where(
            actx.np.greater(x, 0),
            actx.np.tanh(x),
            0*x)

    def smooth_step_bump(actx, x, epsilon=1e-12):
        y = actx.np.minimum(actx.np.maximum(x, 0*x), 0*x+1)
        return actx.np.where(
            actx.np.greater(y, epsilon),
            actx.np.exp(1-1/(1-(y-1)**2)),
            0*y)

    def smooth_step_trig(actx, x, epsilon=1e-12):
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
        fluid_ones = discr.zeros(actx, dd=dd_fluid_vol) + 1
        wall_ones = discr.zeros(actx, dd=dd_wall_vol) + 1
        pressure = 4935.22/x_scale
        temperature = 658.7 * fluid_ones
#         temperature = isothermal_wall_temp
#         smooth_step = smooth_step_bump
#         sigma = 1200
#         offset = -0.0002
#         smooth_step = smooth_step_tanh
#         sigma = 2500/x_scale
#         offset = 0
# #         offset = -0.001*x_scale
        smooth_step = smooth_step_trig
        sigma = 1250/x_scale
        offset = 0
        fluid_nodes = thaw(discr.nodes(dd_fluid_vol), actx)
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

    current_fluid_state = make_fluid_state(current_cv, gas_model)
    current_state = make_obj_array([current_cv, current_wall_temperature])

    fluid_visualizer = make_visualizer(discr, volume_dd=dd_fluid_vol)
    wall_visualizer = make_visualizer(discr, volume_dd=dd_wall_vol)

    initname = "multivolume"
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

    def get_energies(state):
        from grudge.op import integral
        fluid_energy = integral(discr, dd_fluid_vol, state[0].energy)
        wall_energy = integral(
            discr, dd_wall_vol, wall_density * wall_heat_capacity * state[1])
        return fluid_energy, wall_energy, fluid_energy + wall_energy

    fluid_energy_init, wall_energy_init, total_energy_init = get_energies(
        current_state)

    def my_write_status(step, t, state):
        fluid_energy, wall_energy, total_energy = get_energies(state)
        fluid_diff = (fluid_energy - fluid_energy_init)/fluid_energy_init
        wall_diff = (wall_energy - wall_energy_init)/wall_energy_init
        total_diff = (total_energy - total_energy_init)/total_energy_init
        print(
            f"{fluid_energy=} ({fluid_diff/100}%), "
            f"{wall_energy=}, ({wall_diff/100}%), "
            f"{total_energy=}, ({total_diff/100}%)")

    def my_write_viz(step, t, state, dv=None, rhs=None):
        fluid_state = make_fluid_state(state[0], gas_model)
        wall_temperature = state[1]
        if dv is None:
            dv = fluid_state.dv
        if rhs is None:
            rhs = my_rhs(t, state)
        fluid_boundaries, wall_boundaries = get_boundaries(
            t, fluid_state, wall_temperature)
        fluid_grad_temperature = fluid_grad_t_operator(
            discr, gas_model, fluid_boundaries, fluid_state, time=t,
            quadrature_tag=quadrature_tag, volume_dd=dd_fluid_vol)
        wall_grad_temperature = wall_grad_t_operator(
            discr, wall_boundaries, wall_temperature,
            quadrature_tag=quadrature_tag, volume_dd=dd_wall_vol)
        fluid_viz_fields = [
            ("cv", fluid_state.cv),
            ("dv", dv),
            ("grad_t", fluid_grad_temperature),
            ("rhs", rhs[0]),
            ("kappa", fluid_state.thermal_conductivity),
        ]
        wall_viz_fields = [
            ("temperature", wall_temperature),
            ("energy", wall_density * wall_heat_capacity * wall_temperature),
            ("grad_t", wall_grad_temperature),
            ("rhs", rhs[1]),
            ("kappa", wall_kappa),
        ]
        from mirgecom.simutil import write_visfile
        write_visfile(
            discr, fluid_viz_fields, fluid_visualizer, vizname=casename+"-fluid",
            step=step, t=t, overwrite=True, vis_timer=vis_timer)
        write_visfile(
            discr, wall_viz_fields, wall_visualizer, vizname=casename+"-wall",
            step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_fluid_mesh": local_fluid_mesh,
                "local_wall_mesh": local_wall_mesh,
                "cv": state[0],
                "wall_temperature": state[1],
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure):
        health_error = False
#         from mirgecom.simutil import check_naninf_local, check_range_local
#         if check_naninf_local(discr, "vol", pressure) \
#            or check_range_local(discr, "vol", pressure, .8, 1.5):
#             health_error = True
#             logger.info(f"{rank=}: Invalid pressure data found.")
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, dd_fluid_vol, pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        # default health status bounds
        health_pres_min = 1.0e-1/x_scale
        health_pres_max = 2.0e6/x_scale

        if global_reduce(check_range_local(discr, dd_fluid_vol, pressure,
                                     health_pres_min, health_pres_max),
                                     op="lor"):
            health_error = True
            from grudge.op import nodal_min, nodal_max
            p_min = actx.to_numpy(nodal_min(discr, dd_fluid_vol, pressure))
            p_max = actx.to_numpy(nodal_max(discr, dd_fluid_vol, pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")
        return health_error

    def my_pre_step(step, t, dt, state):
        fluid_state = make_fluid_state(state[0], gas_model)
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
                my_write_status(step=step, t=t, state=state)

            if do_health:
                health_errors = global_reduce(my_health_check(dv.pressure), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        dt = get_sim_timestep(discr, fluid_state, t, dt, current_cfl, t_final,
                              constant_cfl, fluid_volume_dd=dd_fluid_vol)
        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state[0], eos)
            logmgr.tick_after()
        return state, dt

    def get_boundaries(t, fluid_state, wall_temperature):
        # Construct the BCs without temperature gradients
        pairwise_vol_data_no_grad = {
            (dd_fluid_vol, dd_wall_vol): (
                make_obj_array([
                    fluid_state.temperature,
                    fluid_kappa]),
                make_obj_array([
                    wall_temperature,
                    wall_kappa]))}
        inter_vol_tpairs_no_grad = inter_volume_trace_pairs(
            discr, pairwise_vol_data_no_grad)

        fluid_boundaries_no_grad = {
            dd_fluid_vol.trace("Upper Sides").domain_tag: IsothermalNoSlipBoundary(
                wall_temperature=isothermal_wall_temp)}
        fluid_tpairs_no_grad = inter_vol_tpairs_no_grad[dd_wall_vol, dd_fluid_vol]
        for tpair in fluid_tpairs_no_grad:
            bdtag = tpair.dd.domain_tag
            ext_temperature, ext_kappa = tpair.ext
            fluid_boundaries_no_grad[bdtag] = TemperatureCoupledNoSlipBoundary(
                ext_temperature,
                (0*ext_temperature,)*dim,
                ext_kappa)

        wall_boundaries_no_grad = {
            dd_wall_vol.trace("Lower Sides").domain_tag: NeumannDiffusionBoundary(0)}
        wall_tpairs_no_grad = inter_vol_tpairs_no_grad[dd_fluid_vol, dd_wall_vol]
        for tpair in wall_tpairs_no_grad:
            bdtag = tpair.dd.domain_tag
            ext_temperature, ext_kappa = tpair.ext
            wall_boundaries_no_grad[bdtag] = InterfaceDiffusionBoundary(
                ext_temperature,
                (0*ext_temperature,)*dim,
                ext_kappa)

        # Compute the temperature gradients
        fluid_grad_temperature = fluid_grad_t_operator(
            discr, gas_model, fluid_boundaries_no_grad, fluid_state, time=t,
            quadrature_tag=quadrature_tag, volume_dd=dd_fluid_vol)
        wall_grad_temperature = wall_grad_t_operator(
            discr, wall_boundaries_no_grad, wall_temperature,
            quadrature_tag=quadrature_tag, volume_dd=dd_wall_vol)

        # Construct the BCs again, now with temperature gradients
        pairwise_vol_data = {
            (dd_fluid_vol, dd_wall_vol): (
                make_obj_array([
                    fluid_state.temperature,
                    fluid_grad_temperature,
                    fluid_kappa]),
                make_obj_array([
                    wall_temperature,
                    wall_grad_temperature,
                    wall_kappa]))}
        inter_vol_tpairs = inter_volume_trace_pairs(discr, pairwise_vol_data)

        fluid_boundaries = {
            dd_fluid_vol.trace("Upper Sides").domain_tag: IsothermalNoSlipBoundary(
                wall_temperature=isothermal_wall_temp)}
        fluid_tpairs = inter_vol_tpairs[dd_wall_vol, dd_fluid_vol]
        for tpair in fluid_tpairs:
            bdtag = tpair.dd.domain_tag
            ext_temperature, ext_grad_temperature, ext_kappa = tpair.ext
            fluid_boundaries[bdtag] = TemperatureCoupledNoSlipBoundary(
                ext_temperature,
                ext_grad_temperature,
                ext_kappa)

        wall_boundaries = {
            dd_wall_vol.trace("Lower Sides").domain_tag: NeumannDiffusionBoundary(0)}
        wall_tpairs = inter_vol_tpairs[dd_fluid_vol, dd_wall_vol]
        for tpair in wall_tpairs:
            bdtag = tpair.dd.domain_tag
            ext_temperature, ext_grad_temperature, ext_kappa = tpair.ext
            wall_boundaries[bdtag] = InterfaceDiffusionBoundary(
                ext_temperature,
                ext_grad_temperature,
                ext_kappa)

        return fluid_boundaries, wall_boundaries

    def my_rhs(t, state):
        fluid_state = make_fluid_state(cv=state[0], gas_model=gas_model)
        wall_temperature = state[1]
        fluid_boundaries, wall_boundaries = get_boundaries(
            t, fluid_state, wall_temperature)
        rhs = make_obj_array([
            ns_operator(
                discr, state=fluid_state, boundaries=fluid_boundaries,
                gas_model=gas_model, time=t, quadrature_tag=quadrature_tag,
                volume_dd=dd_fluid_vol),
            1/(wall_density * wall_heat_capacity) * diffusion_operator(
                discr, quad_tag=quadrature_tag, alpha=wall_kappa,
                boundaries=wall_boundaries, u=wall_temperature,
                volume_dd=dd_wall_vol)])

        return rhs

    current_fluid_state = make_fluid_state(current_state[0], gas_model)
    current_dt = get_sim_timestep(discr, current_fluid_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl,
                                  fluid_volume_dd=dd_fluid_vol)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

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
    casename = "multivolume"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    if args.profiling:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(use_logmgr=args.log, use_overintegration=args.overintegration,
         use_leap=args.leap, use_profiling=args.profiling,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class,
         lazy=args.lazy)

# vim: foldmethod=marker
