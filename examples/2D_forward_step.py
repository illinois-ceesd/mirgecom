"""Demonstrate the forward step example."""

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
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer

from mirgecom.discretization import create_discretization_collection
from mirgecom.navierstokes import ns_operator
from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from grudge.shortcuts import compiled_lsrk45_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    DummyBoundary,
    AdiabaticSlipBoundary,
)
from mirgecom.initializers import Uniform
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state, make_operator_fluid_states
from logpyle import IntervalTimer, set_dt

from mirgecom.fluid import make_conserved

from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    logmgr_set_time,
    set_sim_state
)
logger = logging.getLogger(__name__)

class MyRuntimeError(RuntimeError):
    pass


class _FluidOpStatesTag:
    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_profiling=False, casename=None, lazy=False, rst_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=False,
                           use_einsum_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    mesh_filename = "mesh_v3-v2.msh"

    # timestepping control
    current_step = 0
    t_final = 60.0
    current_cfl = 0.2
    current_dt = 1e-4
    current_t = 0
    integrator = "compiled_lsrk45"

    order = 2

    constant_cfl = True

    # some i/o frequencies
    nviz = 1000
    nrestart = 10000
    nstatus = 1
    nhealth = -1

    niter = 200001

    from grudge.shortcuts import compiled_lsrk45_step
    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    force_eval = True
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval = False

###################################################

    dim = 2
    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    restart_file = rst_filename

    restart_step = None
    if restart_file is None:        
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            tag_to_elements = None
            volume_to_tags = None
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

        local_mesh = volume_to_local_mesh_data
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["nparts"]

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    from grudge.dof_desc import DISCR_TAG_BASE
    quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    nodes = actx.thaw(dcoll.nodes())

##############################################################################

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.6e} s, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

    # soln setup and init
    eos = IdealSingleGas(gamma=1.4, gas_const=1.0)

    from mirgecom.transport import SimpleTransport
    kappa = 0.0
    mu = 1e-3
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)

    gas_model = GasModel(eos=eos, transport=transport_model)

###############################

    def _get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model)
    get_fluid_state = actx.compile(_get_fluid_state)

###############################

    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[0] = np.sqrt(1.4*1.0*0.5)
    init_flow = Uniform(rho=0.1, pressure=0.1*1.0*0.5, velocity=vel)

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = init_flow(x_vec=nodes, eos=eos)
    else:
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t))

        if restart_order != order:
            from grudge.eager import EagerDGDiscretization
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol"))

            current_cv = connection(restart_data["cv"])
        else:
            current_cv = restart_data["cv"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    current_cv = force_evaluation(actx, current_cv)

    current_state = get_fluid_state(current_cv)

#####################################################################################

    zeros = force_evaluation(actx, nodes[0]*0.0)
    ones = force_evaluation(actx, nodes[0]*0.0 + 1.0)

#####################################################################################

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
        inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
        inflow_cv_cond = init_flow(x_vec=inflow_nodes, eos=eos)
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model)

    inflow_bnd = PrescribedFluidBoundary(boundary_state_func=inlet_bnd_state_func)
    outflow_bnd = DummyBoundary()
    wall_bnd = AdiabaticSlipBoundary()

    from grudge.dof_desc import DTAG_BOUNDARY
    boundaries = {DTAG_BOUNDARY("inlet"): inflow_bnd,
                  DTAG_BOUNDARY("outlet"): outflow_bnd,
                  DTAG_BOUNDARY("wall"): wall_bnd}

#############################################################################

    current_state = make_fluid_state(current_cv, gas_model)

    visualizer = make_visualizer(dcoll)

    initname = init_flow.__class__.__name__
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

    def my_write_viz(step, t, state):
                     
        viz_fields = [("CV", state.cv),
                      ("DV_U", state.cv.velocity),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      # ("TV_mu", state.tv.viscosity),
                      # ("TV_kappa", state.tv.thermal_conductivity),
                      ]
                      
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True) 

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
                "nparts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure):
        health_error = False
        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        return health_error

#########################################################################

    def my_pre_step(step, t, dt, state):
        if logmgr:
            logmgr.tick_before()

        cv = force_evaluation(actx, state)
    
        fluid_state = get_fluid_state(cv=cv)
        fluid_state = force_evaluation(actx, fluid_state)

        cv = fluid_state.cv
        dv = fluid_state.dv

        if constant_cfl:
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                  t_final, constant_cfl)

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(
                    my_health_check(dv.pressure), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=cv)

            if do_viz:               
                my_write_viz(step=step, t=t, state=fluid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state)
            my_write_restart(step=step, t=t, state=cv)
            raise

        return cv, dt

    def my_rhs(t, state):
        cv = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)

        operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries, quadrature_tag,
            comm_tag=_FluidOpStatesTag)
     
        return ns_operator(dcoll, gas_model, fluid_state, boundaries,
            time=t, operator_states_quad=operator_states_quad,
            quadrature_tag=quadrature_tag)

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

#################################################################

    dt = 1.0*current_dt
    t = 1.0*current_t

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=dt,
                      state=current_state.cv, t=current_t, t_final=t_final,
                      istep=current_step)

#    # Dump the final data
#    if rank == 0:
#        logger.info("Checkpointing final state ...")

#    current_state = make_fluid_state(current_cv, gas_model)
#    final_dv = current_state.dv
#    final_exact = initializer(x_vec=nodes, eos=eos, time=current_t)
#    final_resid = current_state.cv - final_exact
#    my_write_viz(step=current_step, t=current_t, state=current_state.cv, dv=final_dv,
#                 exact=final_exact, resid=final_resid)
#    my_write_restart(step=current_step, t=current_t, state=current_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    casename = "step"

    import argparse
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")

    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    lazy = args.lazy

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    if args.casename:
        casename = args.casename

    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_logmgr=args.log, lazy=lazy,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
