"""Demonstrate scalar transport."""

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
from functools import partial

from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.dof_desc import DD_VOLUME_ALL
from grudge.shortcuts import make_visualizer

from logpyle import IntervalTimer, set_dt

from mirgecom.mpi import mpi_entry_point
from mirgecom.discretization import create_discretization_collection
from mirgecom.simutil import (
    # get_sim_timestep,
    distribute_mesh
)
from mirgecom.io import make_init_message
import grudge.op as op
import grudge.geometry as geo
from mirgecom.operators import div_operator, grad_operator

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.logging_quantities import (
    initialize_logmgr,
    # logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    # set_sim_state
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def diffusion_numerical_flux(dcoll, diffusivity, soln_trace_pair,
                             grad_soln_trace_pair):
    """Diffusive flux."""
    actx = soln_trace_pair.int.array_context  # = Q^-
    dd = soln_trace_pair.dd
    n_hat = geo.normal(actx, dcoll, dd)
    j_dot_n = diffusivity * np.dot(grad_soln_trace_pair.avg, n_hat)
    return j_dot_n


def gradient_numerical_flux(dcoll, soln_trace_pair):
    """Grad flux."""
    actx = soln_trace_pair.int.array_context  # = Q^-
    dd = soln_trace_pair.dd
    n_hat = geo.normal(actx, dcoll, dd)
    return soln_trace_pair.avg * n_hat


def advection_numerical_flux(dcoll, soln_trace_pair, velocity):
    """Advection flux."""
    actx = soln_trace_pair.int.array_context  # = Q^-
    dd = soln_trace_pair.dd
    n_hat = geo.normal(actx, dcoll, dd)
    v_normal = np.dot(velocity, n_hat)

    return soln_trace_pair.avg * v_normal


@mpi_entry_point
def main(actx_class, use_esdg=False,
         use_overintegration=False, use_leap=False,
         casename=None, rst_filename=None):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    n_step = 10000
    current_cfl = 1.0
    current_dt = .005
    t_final = n_step * current_dt
    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)

    current_t = 0
    constant_cfl = False

    # some i/o frequencies
    nstatus = 1
    nrestart = 5
    nviz = 100
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
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == num_parts
    else:  # generate the grid from scratch
        from meshmode.mesh.generation import generate_regular_rect_mesh
        box_ll = -1
        box_ur = 1
        nel_1d = 16
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(box_ll,)*dim, b=(box_ur,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            periodic=(True,)*dim
        )

        local_mesh, global_nelements = distribute_mesh(comm, generate_mesh)
        local_nelements = local_mesh.nelements

    order = 3
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    rank_field = 0*nodes[0] + rank

    # quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        # logmgr_add_many_discretization_quantities(logmgr, dcoll, dim,
        #                     extract_vars_for_logging, units_for_logging)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    velocity = np.zeros(shape=(dim,))
    velocity[0] = 0.1
    #velocity[1] = 0.05

    alpha = 5.0

    def gaussian_init(xyz_coords):
        x = xyz_coords[0]
        actx = x.array_context
        r2 = np.dot(xyz_coords, xyz_coords)
        return actx.np.exp(-alpha * r2)

    def initializer(xyz_coords):
        dim = len(xyz_coords)
        x = xyz_coords[0]
        actx = x.array_context
        A = 1.0
        #r2 = np.dot(xyz_coords, xyz_coords)
        #return actx.np.exp(-alpha * r2)

        return A*actx.np.cos(0.25*x)

    init_state = initializer(nodes)
    exact_state = initializer(nodes)
    # boundaries = {}

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_state = restart_data["state"]
    else:
        # Set the current state from time 0
        current_state = init_state

    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, current_step, current_t)

    visualizer = make_visualizer(dcoll)

    initname = "gauss"
    eosname = "none"
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_write_viz(step, t, state):
        state_ip = op.interior_trace_pairs(dcoll, state)
        grad_state = my_scalar_gradient(dcoll, state, state_ip)
        viz_fields = [("state", state),
                      ("grad_state", grad_state),
                      ("rank", rank_field),
                      ("exact", exact_state)]
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state):
        health_error = False
        from mirgecom.simutil import check_naninf_local
        if check_naninf_local(dcoll, "vol", state):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")
        return health_error

    def my_pre_step(step, t, dt, state):
        #attempt at updating the exact state
        xyz_coords = nodes
        dim = len(xyz_coords)
        x = xyz_coords[0]
        actx = x.array_context
        d = 1e-3
        exact_state = A*np.exp(-1*((0.25)**2)*d*t)*actx.np.cos(0.25*(x - 0.1*t))
        #

        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(my_health_check(state), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("State failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        # We need a DT computation for scalar transport
        # dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl, t_final,
        #                      constant_cfl)

        return state, dt

    def my_post_step(step, t, dt, state):
        
        
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_scalar_gradient(dcoll, state, state_trace_pairs):

        def flux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces",
                              gradient_numerical_flux(dcoll, tpair))

        surface_flux = sum(flux(tpair) for tpair in state_trace_pairs)

        return grad_operator(dcoll, dd_vol, dd_allfaces, state, surface_flux)

    def advection_rhs(state, state_trace_pairs):
        def flux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces",
                              advection_numerical_flux(dcoll, tpair, velocity))

        volume_flux = velocity * state
        surface_flux = sum(flux(tpair) for tpair in state_trace_pairs)
        return -div_operator(dcoll, dd_vol, dd_allfaces, volume_flux, surface_flux)

    def diffusion_rhs(diffusivity, state, grad_state, state_trace_pairs,
                      grad_state_trace_pairs):
        all_trace_pairs = zip(state_trace_pairs, grad_state_trace_pairs)

        def flux(diffusivity, state_tpair, grad_tpair):
            return op.project(dcoll, state_tpair.dd, "all_faces",
                              diffusion_numerical_flux(dcoll, diffusivity,
                                                       state_tpair, grad_tpair))

        surface_flux = sum(flux(diffusivity, state_tpair, grad_state_tpair)
                           for state_tpair, grad_state_tpair in all_trace_pairs)
        volume_flux = diffusivity*grad_state
        return div_operator(dcoll, dd_vol, dd_allfaces, volume_flux, surface_flux)

    def my_rhs(t, state):
        

        state_trace_pairs = op.interior_trace_pairs(dcoll, state)
        diffusivity = 1e-3
        grad_state = my_scalar_gradient(dcoll, state, state_trace_pairs)
        grad_state_trace_pairs = op.interior_trace_pairs(dcoll, grad_state)
        rhs = advection_rhs(state, state_trace_pairs)
        rhs = rhs + diffusion_rhs(diffusivity, state, grad_state, state_trace_pairs,
                                  grad_state_trace_pairs)
        return rhs

    # Need a DT calculation for scalar transport
    # current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
    #                             current_cfl, t_final, constant_cfl)

    current_step, current_t, final_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_write_viz(step=current_step, t=current_t, state=final_state)
    my_write_restart(step=current_step, t=current_t, state=final_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = current_dt
    print(f"{current_t=}, {t_final=}, {(current_t - t_final)=}")
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "scalar-transport"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--esdg", action="store_true",
        help="use entropy-stable dg for inviscid terms.")
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
         use_leap=args.leap,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
