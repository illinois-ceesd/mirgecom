"""Drive a :class:`mirgecom.simutil.SimulationApplication` time advancement."""

__copyright__ = """
dCopyright (C) 2020 University of Illinois Board of Trustees
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
import pyopencl as cl
import pyopencl.tools as cl_tools
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.mpi import mpi_entry_point

from mirgecom.steppers import advance_state
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, simulation_application_class, casename,
         ctx_factory=cl.create_some_context, use_logmgr=True,
         lazy=False, use_profiling=False, rst_filename=None, cfg_object=None):
    """Drive advancement of a :class:`mirgecom.simutil.SimulationApplication`."""
    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    sim_app = simulation_application_class(
        actx=actx, comm=comm, log_manager=logmgr, casename=casename,
        config_object=cfg_object, restart_file_root=rst_filename)

    if sim_app.status:
        raise RuntimeError("Simulation application failed to initialize.")

    current_advancer_state = sim_app.get_initial_advancer_state()
    current_step, current_time = sim_app.get_initial_stepper_position()
    max_step, final_time = sim_app.get_final_stepper_position()
    max_dt = sim_app.get_timestep_dt(current_advancer_state, current_time)
    timestepper = sim_app.get_timestepper_method()

    if rank == 0:
        logger.info(f"Running MIRGE-Com simulation application: {sim_app.name=}\n"
                    f"Starting at: {current_step=}, {current_time=}, {max_dt=}\n"
                    f"Finishing by: {max_step=}, {final_time=}\n"
                    f"Using timestepper: {timestepper=}")

    current_step, current_time, current_advancer_state = \
        advance_state(rhs=sim_app.rhs, timestepper=timestepper,
                      pre_step_callback=sim_app.pre_step_callback,
                      post_step_callback=sim_app.post_step_callback, dt=max_dt,
                      state=current_advancer_state, t=current_time,
                      t_final=final_time)

    if rank == 0:
        logger.info(f"Stepping finished at: {current_step=}, {current_time=}")

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    if sim_app.finalize(step=current_step, t=current_time,
                        state=current_advancer_state):
        raise RuntimeError("Simulation application has experienced an error.")


if __name__ == "__main__":
    import argparse
    casename = "mirgecom"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--simulation-module",
                        help="path/filename of simulation application module.")
    parser.add_argument("--config-filename",
                        help="path/filename of application configuration module.")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    lazy = args.lazy

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

    cfg_object = None
    if args.config_filename:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config_module", args.config_filename)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        cfg_object = cfg_module.Config()

    from mirgecom.simutil import DummySimulation
    app_class = DummySimulation
    if args.simulation_module:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "module.name", args.simulation_module)
        application_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(application_module)
        app_class = application_module.SimulationApplication

    main(actx_class, app_class, casename, cfg_object=cfg_object, lazy=lazy,
         use_profiling=args.profiling, use_logmgr=args.log,
         rst_filename=rst_filename)

# vim: foldmethod=marker
