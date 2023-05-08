"""mirgecom driver for the Y2 prediction."""

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
import sys
import argparse
import importlib
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    logmgr_add_mempool_usage
)

from mirgecom.mpi import mpi_entry_point


class DriverCommandLineError(Exception):
    pass


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def mirge_driver(ctx_factory=cl.create_some_context,
           restart_filename=None, target_filename=None,
           use_profiling=False, use_logmgr=True, user_input_file=None,
           use_overintegration=False, actx_class=None, casename=None,
           lazy=False, log_path="log_data",
           simulation_module=None):

    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    simulation_module_name = simulation_module.__name__

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # logging and profiling
    logname = log_path + "/" + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    comm.Barrier()

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    monitor_memory = True
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        if monitor_memory:
            logmgr_add_device_memory_usage(logmgr, queue)
            logmgr_add_mempool_usage(logmgr, alloc)

            logmgr.add_watches([
                ("memory_usage_python.max",
                 "| Memory:\n| \t python memory: {value:7g} Mb\n")
            ])

            try:
                logmgr.add_watches([
                    ("memory_usage_gpu.max",
                     "| \t gpu memory: {value:7g} Mb\n")
                ])
            except KeyError:
                pass

            logmgr.add_watches([
                ("memory_usage_hwm.max",
                 "| \t memory hwm: {value:7g} Mb\n"),
                ("memory_usage_mempool_managed.max",
                 "| \t mempool total: {value:7g} Mb\n"),
                ("memory_usage_mempool_active.max",
                 "| \t mempool active: {value:7g} Mb")
            ])

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

    comm.Barrier()

    if rank == 0:
        logger.info(f"MIRGE-Com Running Simulation module: {simulation_module_name}")

    simulation_module.driver(comm=comm, actx=actx, logmgr=logmgr, logger=logger,
                             restart_filename=restart_filename,
                             target_filename=target_filename,
                             user_input_file=user_input_file, casename=casename,
                             use_overintegration=use_overintegration)
    comm.Barrier()

    if rank == 0:
        logger.info("MIRGE-Com Simulation module done!")

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())


def main(args=None):

    program_name = sys.argv[0]

    if args is None:
        args = sys.argv[1:]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="MIRGE-Com Simulation Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("-g", "--logpath", type=ascii, dest="log_path", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("-s", "--simulation-module", dest="sim_module", nargs="?",
                        action="store", help="path to user simulation module file")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    args = parser.parse_args()

    module_name = args.sim_module or "mirgecom.drivers.example_driver"
    if module_name.endswith(".py"):
        module_name = module_name[:-3].replace("/", ".")
    try:
        simulation_module = importlib.import_module(module_name)
        print(f"Loaded simulation module: {module_name}")
    except ModuleNotFoundError:
        raise DriverCommandLineError(f"Simulation module not found: {module_name}.")

    # for writing output
    casename = "mirge-com"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    lazy = args.lazy
    if args.profile:
        if lazy:
            raise DriverCommandLineError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    target_filename = None
    if args.target_file:
        target_filename = (args.target_file).replace("'", "")
        print(f"Target file specified: {target_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    log_path = "log_data"
    if args.log_path:
        log_path = args.log_path.replace("'", "")

    print(f"MIRGE-Com: Running {program_name}.\n")

    mirge_driver(restart_filename=restart_filename,
                 target_filename=target_filename,
                 user_input_file=input_file, log_path=log_path,
                 use_profiling=args.profile, use_logmgr=True,
                 use_overintegration=args.overintegration, lazy=lazy,
                 actx_class=actx_class, casename=casename,
                 simulation_module=simulation_module)
