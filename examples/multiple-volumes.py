"""
Demonstrate multiple non-interacting volumes.

Runs several acoustic pulse simulations with different pulse amplitudes
simultaneously.
"""

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

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import VolumeDomainTag, DISCR_TAG_BASE, DISCR_TAG_QUAD, DOFDesc

from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import (
    euler_operator,
    extract_vars_for_logging
)
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import (
    Lump,
    AcousticPulse
)
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from logpyle import IntervalTimer, set_dt
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

    # Run simulations with several different pulse amplitudes simultaneously
    pulse_amplitudes = [0.01, 0.1, 1.0]
    nvolumes = len(pulse_amplitudes)

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_prototype_mesh = restart_data["local_prototype_mesh"]
        global_prototype_nelements = restart_data["global_prototype_nelements"]
        assert restart_data["num_parts"] == num_parts
    else:  # generate the grids from scratch
        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                a=(-1,)*dim, b=(1,)*dim, nelements_per_axis=(16,)*dim)
        local_prototype_mesh, global_prototype_nelements = \
            generate_and_distribute_mesh(comm, generate_mesh)

    volume_to_local_mesh = {i: local_prototype_mesh for i in range(nvolumes)}

    local_nelements = local_prototype_mesh.nelements * nvolumes
    global_nelements = global_prototype_nelements * nvolumes

    order = 3
    dcoll = create_discretization_collection(actx, volume_to_local_mesh, order=order)

    volume_dds = [
        DOFDesc(VolumeDomainTag(i), DISCR_TAG_BASE)
        for i in range(nvolumes)]

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        def extract_vars(i, dim, cvs, eos):
            name_to_field = extract_vars_for_logging(dim, cvs[i], eos)
            return {
                name + f"_{i}": field
                for name, field in name_to_field.items()}

        def units(quantity):
            return ""

        for i in range(nvolumes):
            logmgr_add_many_discretization_quantities(
                logmgr, dcoll, dim, partial(extract_vars, i), units,
                dd=volume_dds[i])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s\n")
        ])

        for i in range(nvolumes):
            logmgr.add_watches([
                (f"min_pressure_{i}", "------- P (vol. " + str(i)
                    + ") (min, max) (Pa) = ({value:1.9e}, "),
                (f"max_pressure_{i}",    "{value:1.9e})\n"),
            ])

    eos = IdealSingleGas()
    gas_model = GasModel(eos=eos)
    wall = AdiabaticSlipBoundary()
    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cvs = restart_data["cvs"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        def init(nodes, pulse_amplitude):
            vel = np.zeros(shape=(dim,))
            orig = np.zeros(shape=(dim,))
            background = Lump(
                dim=dim, center=orig, velocity=vel, rhoamp=0.0)(nodes)
            return AcousticPulse(
                dim=dim,
                amplitude=pulse_amplitude,
                width=0.1,
                center=orig)(x_vec=nodes, cv=background, eos=eos)
        current_cvs = make_obj_array([
            init(actx.thaw(dcoll.nodes(dd)), pulse_amplitude)
            for dd, pulse_amplitude in zip(volume_dds, pulse_amplitudes)])

    current_fluid_states = [make_fluid_state(cv, gas_model) for cv in current_cvs]

    visualizers = [make_visualizer(dcoll, volume_dd=dd) for dd in volume_dds]

    initname = "multiple-volumes"
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

    def my_get_timestep(step, t, dt, fluid_states):
        return min([
            get_sim_timestep(
                dcoll, fluid_state, t, dt, current_cfl, t_final, constant_cfl)
            for fluid_state in fluid_states])

    def my_write_viz(step, t, cvs, dvs=None):
        if dvs is None:
            dvs = [eos.dependent_vars(cv) for cv in cvs]
        for i in range(nvolumes):
            viz_fields = [
                ("cv", cvs[i]),
                ("dv", dvs[i])]
            from mirgecom.simutil import write_visfile
            write_visfile(
                dcoll, viz_fields, visualizers[i], vizname=casename + f"-{i}",
                step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)

    def my_write_restart(step, t, cvs):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_prototype_mesh": local_prototype_mesh,
                "cvs": cvs,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressures):
        health_error = False
        for dd, pressure in zip(volume_dds, pressures):
            from mirgecom.simutil import check_naninf_local, check_range_local
            if check_naninf_local(dcoll, dd, pressure) \
               or check_range_local(dcoll, dd, pressure, 1e-2, 10):
                health_error = True
                logger.info(f"{rank=}: Invalid pressure data found.")
                break
        return health_error

    def my_pre_step(step, t, dt, state):
        cvs = state
        fluid_states = [make_fluid_state(cv, gas_model) for cv in cvs]
        dvs = [fluid_state.dv for fluid_state in fluid_states]

        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                pressures = [dv.pressure for dv in dvs]
                health_errors = global_reduce(my_health_check(pressures), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cvs=cvs)

            if do_viz:
                my_write_viz(step=step, t=t, cvs=cvs, dvs=dvs)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, cvs=cvs)
            my_write_restart(step=step, t=t, cvs=cvs)
            raise

        dt = my_get_timestep(step=step, t=t, dt=dt, fluid_states=fluid_states)

        return cvs, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        cvs = state
        fluid_states = [make_fluid_state(cv, gas_model) for cv in cvs]
        return make_obj_array([
            euler_operator(
                dcoll, state=fluid_state, time=t,
                boundaries={dd.trace(BTAG_ALL).domain_tag: wall},
                gas_model=gas_model, quadrature_tag=quadrature_tag,
                dd=dd, comm_tag=dd, use_esdg=use_esdg)
            for dd, fluid_state in zip(volume_dds, fluid_states)])

    current_dt = my_get_timestep(
        current_step, current_t, current_dt, current_fluid_states)

    current_step, current_t, current_cvs = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_cvs, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_fluid_states = [make_fluid_state(cv, gas_model) for cv in current_cvs]
    final_dvs = [fluid_state.dv for fluid_state in final_fluid_states]

    my_write_viz(step=current_step, t=current_t, cvs=current_cvs, dvs=final_dvs)
    my_write_restart(step=current_step, t=current_t, cvs=current_cvs)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "multiple-volumes"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--esdg", action="store_true",
        help="use entropy-stable DG for inviscid terms")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    parser.add_argument("--cupy", action="store_true",
        help="use cupy-based eager actx.")
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
        lazy=args.lazy, distributed=True, profiling=args.profiling,
        numpy=args.numpy, cupy=args.cupy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class,
         use_leap=args.leap, use_esdg=args.esdg,
         use_overintegration=args.overintegration or args.esdg,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
