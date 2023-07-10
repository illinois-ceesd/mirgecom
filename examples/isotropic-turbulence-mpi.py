"""Demonstrate the Isotropic turbulence."""

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
import pyopencl as cl
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer

from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    get_sim_timestep,
    check_step
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
# from mirgecom.boundary import PrescribedFluidBoundary
from mirgecom.initializers import IsotropicTurbulence
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state
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
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None, periodic=None):
    if periodic is None:
        periodic = (False,)*dim

    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t,
               periodic=periodic)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_overintegration=False, lazy=False,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=None, use_esdg=False):
    """Drive the example."""
    if actx_class is None:
        raise RuntimeError("Array context class missing.")
    print(actx_class)
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

    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    t_final = 5
    current_cfl = 1.0
    current_dt = .001
    current_t = 0
    constant_cfl = False

    # some i/o frequencies
    nrestart = 200
    nstatus = -1
    nviz = 20
    nhealth = -1

    # some geometry setup
    dim = 3
    if dim != 3:
        raise ValueError("This example must be run with dim = 3.")

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
    else:
        left_boundary_location = tuple([0. for _ in range(dim)])
        right_boundary_location = tuple([2*np.pi for _ in range(dim)])
        periodic = (True,)*dim

        n_refine = 1
        pts_per_axis = 16
        npts_axis = tuple([n_refine * pts_per_axis for _ in range(dim)])
        # npts_axis = (npts_x, npts_y)
        box_ll = left_boundary_location
        box_ur = right_boundary_location
        generate_mesh = partial(_get_box_mesh, dim=dim, a=box_ll, b=box_ur,
                                n=npts_axis, periodic=periodic)
        print(f"{left_boundary_location=}")
        print(f"{right_boundary_location=}")
        print(f"{npts_axis=}")
        from mirgecom.simutil import generate_and_distribute_mesh
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    # from meshmode.mesh.processing import rotate_mesh_around_axis
    # local_mesh = rotate_mesh_around_axis(local_mesh, theta=-np.pi/4)
    order = 3
    dcoll = \
            create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    from grudge.dof_desc import DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    vis_timer = None

    eos = IdealSingleGas()
    pathi = "./output_data"
    initializer = IsotropicTurbulence(coordinate_path=pathi+"/coordinate.txt",
                                      velocity_path=pathi+"/velocity.txt")
    gas_model = GasModel(eos=eos)

    boundaries = {}

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        init_order = 1
        init_dcoll = create_discretization_collection(actx, local_mesh,
                                                 order=init_order)
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)

    else:
        init_order = 1
        init_dcoll = create_discretization_collection(actx, local_mesh,
                                                 order=init_order)
        init_nodes = actx.thaw(init_dcoll.nodes())
        actx = (init_nodes[0]).array_context
        data_x = actx.to_numpy(init_nodes[0][0])
        data_y = actx.to_numpy(init_nodes[1][0])
        data_z = actx.to_numpy(init_nodes[2][0])

        np.savetxt(pathi+f"/x_rank{rank}.txt", data_x)
        np.savetxt(pathi+f"/y_rank{rank}.txt", data_y)
        np.savetxt(pathi+f"/z_rank{rank}.txt", data_z)

        # Set the current state from time 0
        low_order_cv = initializer(init_nodes, eos=eos)
        from meshmode.discretization.connection import make_same_mesh_connection
        connection = make_same_mesh_connection(actx, dcoll.discr_from_dd("vol"),
                                                init_dcoll.discr_from_dd("vol"))
        current_cv = connection(low_order_cv)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, dcoll, dim,
                             extract_vars_for_logging, units_for_logging)

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

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

    current_state = make_fluid_state(current_cv, gas_model)

    visualizer = make_visualizer(dcoll)

    initname = initializer.__class__.__name__
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

    def my_write_status(state, component_errors, cfl=None):
        if cfl is None:
            if constant_cfl:
                cfl = current_cfl
            else:
                from grudge.op import nodal_max
                from mirgecom.inviscid import get_inviscid_cfl
                cfl = actx.to_numpy(
                    nodal_max(
                        dcoll, "vol",
                        get_inviscid_cfl(dcoll, state, current_dt)))[()]
        if rank == 0:
            logger.info(
                f"------ {cfl=}\n"
                "------- errors="
                + ", ".join("%.3g" % en for en in component_errors))

    def my_write_low_order_data(t, state):
        from meshmode.discretization.connection import make_same_mesh_connection

        connection2 = make_same_mesh_connection(
            actx, init_dcoll.discr_from_dd("vol"), dcoll.discr_from_dd("vol"))

        currstate = connection2(state)
        mass = actx.to_numpy(currstate.mass)[0]
        energy = actx.to_numpy(currstate.energy)[0]
        momx = actx.to_numpy(currstate.momentum)[0][0]
        momy = actx.to_numpy(currstate.momentum)[1][0]
        momz = actx.to_numpy(currstate.momentum)[2][0]
        path = "./output_data/"

        np.savetxt(path+f"/mass_{t}_{rank}.txt", mass)
        np.savetxt(path+f"/energy_{t}_{rank}.txt", energy)
        np.savetxt(path+f"/momx_{t}_{rank}.txt", momx)
        np.savetxt(path+f"/momy_{t}_{rank}.txt", momy)
        np.savetxt(path+f"/momz_{t}_{rank}.txt", momz)

    def my_write_viz(step, t, state, dv=None, exact=None, resid=None):
        viz_fields = [("cv", state),
                      ("dv", dv)]
        my_write_low_order_data(t, state)
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure, component_errors):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, "vol", pressure) \
           or check_range_local(dcoll, "vol", pressure, .2, 1.02):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        exittol = .1
        if max(component_errors) > exittol:
            health_error = True
            if rank == 0:
                logger.info("Solution diverged from exact soln.")

        return health_error

    def my_pre_step(step, t, dt, state):
        fluid_state = make_fluid_state(state, gas_model)
        cv = fluid_state.cv
        dv = fluid_state.dv

        try:
            exact = None
            component_errors = None

            if logmgr:
                logmgr.tick_before()

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                exact = initializer(x_vec=nodes, eos=eos, time=t)
                from mirgecom.simutil import compare_fluid_solutions
                component_errors = compare_fluid_solutions(dcoll, cv, exact)
                health_errors = global_reduce(
                    my_health_check(dv.pressure, component_errors), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=cv)

            if do_status:
                if component_errors is None:
                    if exact is None:
                        exact = initializer(x_vec=nodes, eos=eos, time=t)
                    from mirgecom.simutil import compare_fluid_solutions
                    component_errors = compare_fluid_solutions(dcoll, cv, exact)
                my_write_status(fluid_state, component_errors)

            if do_viz:
                my_write_viz(step=step, t=t, state=cv, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=cv)
            my_write_restart(step=step, t=t, state=cv)
            raise

        dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl, t_final,
                              constant_cfl)
        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(state, gas_model)
        return euler_operator(dcoll, state=fluid_state, time=t,
                              boundaries=boundaries, gas_model=gas_model,
                              quadrature_tag=quadrature_tag, use_esdg=use_esdg)

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state.cv, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_state = make_fluid_state(current_cv, gas_model)
    final_dv = current_state.dv
    my_write_viz(step=current_step, t=current_t, state=current_state.cv, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "IsotropicTurbulence"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--esdg", action="store_true",
        help="use flux-differencing/entropy stable DG for inviscid computations.")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from warnings import warn
    if args.esdg:
        if not args.lazy:
            warn("ESDG requires lazy-evaluation, enabling --lazy.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    lazy = args.lazy or args.esdg
    if args.profiling:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
         use_overintegration=args.overintegration or args.esdg, lazy=lazy,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class,
         use_esdg=args.esdg)

# vim: foldmethod=marker
