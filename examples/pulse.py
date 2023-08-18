"""Demonstrate acoustic pulse, and adiabatic slip wall."""

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

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag, DISCR_TAG_QUAD

from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    LinearizedOutflowBoundary,
    RiemannInflowBoundary,
    PressureOutflowBoundary
)
from mirgecom.initializers import (
    Uniform,
    AcousticPulse
)
from mirgecom.eos import IdealSingleGas
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
    t_final = 0.1
    current_cfl = 1.0
    current_dt = .005
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
            boundary_tag_to_face={
                "outlet_L": ["+y"],
                "outlet_R": ["-y", "+x"],
                "inlet": ["-x"]})
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    order = 1
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    vis_timer = None

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

    eos = IdealSingleGas(gamma=1.4, gas_const=1.0)
    gas_model = GasModel(eos=eos)
    velocity = np.zeros(shape=(dim,))
    velocity[0] = 0.1
    orig = np.zeros(shape=(dim,))
    initializer = Uniform(dim=dim, velocity=velocity)
    uniform_state = initializer(nodes, eos=eos)

    # Riemann inflow
    from mirgecom.initializers import initialize_flow_solution
    from mirgecom.utils import force_evaluation

    free_stream_cv = initialize_flow_solution(
        actx, dcoll, gas_model, dd_bdry=BoundaryDomainTag("inlet"),
        pressure=1.0, temperature=1.0, velocity=velocity)

    inflow_freestream_state = make_fluid_state(cv=free_stream_cv,
                                               gas_model=gas_model)
    inflow_freestream_state = force_evaluation(actx, inflow_freestream_state)

    def _inflow_bnd_state_func(dcoll, btag, gas_model, state_minus, **kwargs):
        return inflow_freestream_state

    riemann_inflow_bnd = RiemannInflowBoundary(
        free_stream_state_func=_inflow_bnd_state_func)

    # Linearized outflow
    linear_outflow_bnd = LinearizedOutflowBoundary(free_stream_density=1.0,
        free_stream_velocity=velocity, free_stream_pressure=1.0)

    # Pressure prescribed outflow boundary
    pressure_outflow_bnd = PressureOutflowBoundary(boundary_pressure=1.0)

    # boundaries
    boundaries = {BoundaryDomainTag("inlet"): riemann_inflow_bnd,
                  BoundaryDomainTag("outlet_L"): linear_outflow_bnd,
                  BoundaryDomainTag("outlet_R"): pressure_outflow_bnd}

    acoustic_pulse = AcousticPulse(dim=dim, amplitude=0.5, width=.1, center=orig)

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_cv = acoustic_pulse(x_vec=nodes, cv=uniform_state, eos=eos)

    current_state = make_fluid_state(current_cv, gas_model)

    visualizer = make_visualizer(dcoll)

    initname = "pulse"
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

    def my_write_viz(step, t, state, dv=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        viz_fields = [("cv", state),
                      ("dv", dv)]
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

    def my_health_check(pressure):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, "vol", pressure) \
           or check_range_local(dcoll, "vol", pressure, .8, 1.6):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")
        return health_error

    def my_pre_step(step, t, dt, state):
        fluid_state = make_fluid_state(state, gas_model)
        dv = fluid_state.dv

        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

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
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)
        return euler_operator(dcoll, state=fluid_state, time=t,
                              boundaries=boundaries,
                              gas_model=gas_model, use_esdg=use_esdg,
                              quadrature_tag=quadrature_tag)

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_cv, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_state = make_fluid_state(current_cv, gas_model)
    final_dv = final_state.dv

    my_write_viz(step=current_step, t=current_t, state=current_cv, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "pulse"
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
