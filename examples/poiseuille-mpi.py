"""Demonstrate a planar Poiseuille flow example."""

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
import pyopencl.tools as cl_tools
from pytools.obj_array import make_obj_array
from functools import partial

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DTAG_BOUNDARY

from mirgecom.fluid import make_conserved
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import get_sim_timestep

from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedViscousBoundary,
    IsothermalNoSlipBoundary
)
from mirgecom.transport import SimpleTransport
from mirgecom.eos import IdealSingleGas

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state
)


logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None):
    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_leap=False,
         use_profiling=False, rst_step=None, rst_name=None,
         casename="poiseuille", use_logmgr=True):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    order = 1
    t_final = 1e-6
    current_cfl = 0.1
    current_dt = 1e-8
    current_t = 0
    casename = "poiseuille"
    constant_cfl = True
    nstatus = 1
    nviz = 1
    nrestart = 100
    nhealth = 1
    current_step = 0
    timestepper = rk4_step
    left_boundary_location = 0
    right_boundary_location = 0.1
    npts_axis = (50, 30)
    rank = comm.Get_rank()

    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_step:  # read the grid from restart data
        rst_fname = rst_pattern.format(cname=rst_name, step=rst_step, rank=rank)

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_fname)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        box_ll = (left_boundary_location, 0.0)
        box_ur = (right_boundary_location, 0.02)
        generate_mesh = partial(_get_box_mesh, 2, a=box_ll, b=box_ur, n=npts_axis)
        from mirgecom.simutil import generate_and_distribute_mesh
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K) = ({value:1.9e}, "),
            ("max_temperature",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    base_pressure = 100000.0
    pressure_ratio = 1.001

    def poiseuille_soln(nodes, eos, cv=None, **kwargs):
        dim = len(nodes)
        x0 = left_boundary_location
        xmax = right_boundary_location
        xlen = xmax - x0
        p0 = base_pressure
        p1 = pressure_ratio*p0
        p_x = p1 + p0*(1 - pressure_ratio)*(nodes[0] - x0)/xlen
        ke = 0
        mass = nodes[0] + 1.0 - nodes[0]
        momentum = make_obj_array([0*mass for i in range(dim)])
        if cv is not None:
            mass = cv.mass
            momentum = cv.momentum
            ke = .5*np.dot(cv.momentum, cv.momentum)/cv.mass
        energy_bc = p_x / (eos.gamma() - 1) + ke
        return make_conserved(dim, mass=mass, energy=energy_bc,
                              momentum=momentum)

    initializer = poiseuille_soln
    boundaries = {DTAG_BOUNDARY("-1"): PrescribedViscousBoundary(q_func=initializer),
                  DTAG_BOUNDARY("+1"): PrescribedViscousBoundary(q_func=initializer),
                  DTAG_BOUNDARY("-2"): IsothermalNoSlipBoundary(),
                  DTAG_BOUNDARY("+2"): IsothermalNoSlipBoundary()}
    eos = IdealSingleGas(transport_model=SimpleTransport(viscosity=1.0))

    if rst_step:
        current_t = restart_data["t"]
        current_step = rst_step
        current_state = restart_data["state"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_state = initializer(nodes=nodes, eos=eos)

    vis_timer = None

    visualizer = make_visualizer(discr, order)

    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_write_status(step, t, dt, dv, state):
        from grudge.op import nodal_min, nodal_max
        p_min = nodal_min(discr, "vol", dv.pressure)
        p_max = nodal_max(discr, "vol", dv.pressure)
        t_min = nodal_min(discr, "vol", dv.temperature)
        t_max = nodal_max(discr, "vol", dv.temperature)
        if constant_cfl:
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            cfl = nodal_max(discr, "vol",
                            get_viscous_cfl(discr, eos, dt, state))
        if rank == 0:
            logger.info(f"Step: {step}, T: {t}, DT: {dt}, CFL: {cfl}\n"
                        f"----- Pressure({p_min}, {p_max})\n"
                        f"----- Temperature({t_min}, {t_max})\n")

    def my_write_viz(step, t, state, dv=None, exact=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        viz_fields = [("cv", state),
                      ("dv", dv)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        rst_data = {
            "local_mesh": local_mesh,
            "state": state,
            "t": t,
            "step": step,
            "order": order,
            "global_nelements": global_nelements,
            "num_parts": nparts
        }
        from mirgecom.restart import write_restart_file
        write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state, dv):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        from mirgecom.simutil import allsync
        if allsync(check_range_local(discr, "vol", dv.pressure, 9.999e4, 1.00101e5),
                   comm, op=MPI.LOR):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            p_min = nodal_min(discr, "vol", dv.pressure)
            p_max = nodal_max(discr, "vol", dv.pressure)
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if check_naninf_local(discr, "vol", dv.temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/INFs in temperature data.")

        if allsync(check_range_local(discr, "vol", dv.temperature, 348, 350),
                   comm, op=MPI.LOR):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            t_min = nodal_min(discr, "vol", dv.temperature)
            t_max = nodal_max(discr, "vol", dv.temperature)
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        return health_error

    def my_pre_step(step, t, dt, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                dv = eos.dependent_vars(state)
                from mirgecom.simutil import allsync
                health_errors = allsync(my_health_check(state, dv), comm,
                                        op=MPI.LOR)
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if step == rst_step:  # don't do viz or restart @ restart
                do_viz = False
                do_restart = False

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                my_write_viz(step=step, t=t, state=state, dv=dv)

            dt = get_sim_timestep(discr, state, t, dt, current_cfl, eos,
                                  t_final, constant_cfl)

            if do_status:  # needed because logging fails to make output
                if dv is None:
                    dv = eos.dependent_vars(state)
                my_write_status(step=step, t=t, dt=dt, dv=dv, state=state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

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
        return ns_operator(discr, eos=eos, boundaries=boundaries, cv=state, t=t)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = eos.dependent_vars(current_state)
    final_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                current_cfl, eos, t_final, constant_cfl)
    my_write_viz(step=current_step, t=current_t, state=current_state, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state)
    my_write_status(step=current_step, t=current_t, dt=final_dt, dv=final_dv,
                    state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    use_profiling = False
    use_logging = False

    main(use_profiling=use_profiling, use_logmgr=use_logging)

# vim: foldmethod=marker
