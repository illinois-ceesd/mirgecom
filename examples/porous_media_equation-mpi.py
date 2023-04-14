"""Demonstrate diffusion boundary conditions example."""

__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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

from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from mirgecom.discretization import create_discretization_collection

from mirgecom.integrators.ssprk import ssprk43_step

from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary
)

from mirgecom.simutil import (
    check_naninf_local,
    generate_and_distribute_mesh
)

from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage)

from logpyle import IntervalTimer, set_dt

from mirgecom.simutil import write_visfile

#########################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_file=None):
    """Run the example."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename="pme.sqlite", mode="wo", mpi_comm=comm)

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

    viz_path = "viz_data/"
    vizname = viz_path+casename

    nviz = 50
    ngarbage = 100
    nrestart = 1000

    t_final = 0.8

    order = 2
    dt = 1.5e-5

####################################

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    )
    if restart_file:  # read the grid from restart data
        rst_filename = f"{restart_file}"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == num_parts

    else:  # generate the grid from scratch
        from functools import partial
        box_ll = (-.0025, 0.0)
        box_ur = (+.0025, .05)
        num_elements = (6+1, 60+1)

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                                a=box_ll,
                                b=box_ur,
                                n=num_elements,
                                periodic=(True, False),
                                boundary_tag_to_face={"dirichlet": ["+y"],
                                                      "neumann": ["-y"]})
        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    nodes = actx.thaw(dcoll.nodes())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # arguments are available
    def my_presc_u_func(**kwargs):
        u_minus = kwargs["u_minus"]  # noqa 841
        time = kwargs["time"]  # noqa 841
        return 101325.0

    # arguments are available
    def my_presc_grad_u_func(**kwargs):
        u_minus = kwargs["u_minus"]  # noqa 841
        grad_u_minus = kwargs["grad_u_minus"]  # noqa 841
        time = kwargs["time"]  # noqa 841
        return 0.0

    # if a more complex boundary condition is to be applied, then an external
    # function can be used to prescribe either the property (Dirichlet) or
    # its gradient (Neumann)
    boundaries = {
        BoundaryDomainTag("dirichlet"):
            DirichletDiffusionBoundary(function=my_presc_u_func),
        BoundaryDomainTag("neumann"):
            NeumannDiffusionBoundary(function=my_presc_grad_u_func)
    }

    # alternatively, one can specifiy "value" and "boundary_gradient" for
    # DirichletDiffusionBoundary and NeumannDiffusionBoundary, respectively
    # boundaries = {
    #     BoundaryDomainTag("dirichlet"):
    #         DirichletDiffusionBoundary(value=101325.0),
    #     BoundaryDomainTag("neumann"):
    #         NeumannDiffusionBoundary(boundary_gradient=0.0)
    # }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if restart_file:
        t = restart_data["t"]
        istep = restart_data["step"]
        pressure = restart_data["pressure"]
    else:
        t = 0
        istep = 0
        # since the diffusivity depends on pressure, the BCs alone are not
        # enough to enforce "RHS != 0.0", so I have to poke only at the very
        # first grid point...
        pressure = actx.np.where(actx.np.greater(nodes[1], 0.0499),
            101325.0,
            0.0)

    pressure = force_evaluation(actx, pressure)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, istep, t)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.3e} s, "),
            ("t_sim.max", "sim time: {value:7.3f} s, "),
            ("t_step.max", "step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_field(u):
        return bound_preserving_limiter(dcoll, u, mmin=0.0)

    limit_field = actx.compile(_limit_field)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mu = 1e-4
    permeability = 1.0e-12
    epsilon = 1.0

    def _rhs(t, pressure):

        # limit the field before passing it to the RHS function
        pres_limited = _limit_field(pressure)

        # using the limited pressure
        diffusivity = pres_limited*permeability/(mu*epsilon)

        return diffusion_operator(dcoll, kappa=diffusivity,
            boundaries=boundaries, u=pres_limited, time=t)

    compiled_rhs = actx.compile(_rhs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    visualizer = make_visualizer(dcoll)

    def my_write_viz(step, t, pressure):
        diffusivity = pressure*permeability/(mu*epsilon)
        rhs, grad = diffusion_operator(dcoll, kappa=diffusivity,
            boundaries=boundaries, u=pressure, time=t, return_grad_u=True)

        viz_fields = [("pressure", pressure),
                      ("gradient", grad),
                      ("rhs", rhs)]

        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
            step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)

    def my_write_restart(step, t, pressure):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "pressure": pressure,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    u = pressure*1.0
    my_write_viz(step=istep, t=t, pressure=u)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from warnings import warn
    warn("Running gc.collect() to work around memory growth issue ")
    import gc
    gc.collect()

    while t < t_final:

        if logmgr:
            logmgr.tick_before()

        u = ssprk43_step(u, t, dt, compiled_rhs)
        u = force_evaluation(actx, u)

        t += dt
        istep += 1

        try:

            if check_naninf_local(dcoll, "vol", u):
                if rank == 0:
                    logger.info("Fluid solution failed health check.")
                raise MyRuntimeError("Failed simulation health check.")

            if istep % nviz == 0:
                # since limiting is only applied to the field down inside the
                # rhs, it may be non-positive when writing the viz file
                # thus, this extra limiting here can remove the "spurious"
                # negative values that come after the timestepper
                u = limit_field(u)
                my_write_viz(step=istep, t=t, pressure=u)

            if istep % ngarbage == 0:
                gc.collect()

            if istep % nrestart == 0:
                my_write_restart(step=istep, t=t, pressure=u)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=istep, t=t, pressure=u)
            raise

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    if logmgr:
        logmgr.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":

    import argparse
    casename = "pme"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
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
        rst_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {rst_filename}")

    main(actx_class, use_logmgr=args.log, use_leap=args.leap, lazy=lazy,
         use_profiling=args.profiling, casename=casename,
         restart_file=rst_filename)

# vim: foldmethod=marker
