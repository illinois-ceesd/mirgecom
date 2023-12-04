""" Fri 10 Mar 2023 02:22:21 PM CST """

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
import numpy as np
import pyopencl as cl
import pyopencl.array as cla  # noqa
import pyopencl.tools as cl_tools
from functools import partial
from dataclasses import dataclass, fields

from arraycontext import (
    dataclass_array_container, with_container_arithmetic,
    get_container_context_recursively
)

from meshmode.dof_array import DOFArray
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, BoundaryDomainTag, VolumeDomainTag
)

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step, get_sim_timestep, distribute_mesh, write_visfile,
    check_naninf_local, check_range_local, global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.steppers import advance_state
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)

from mirgecom.diffusion import (
    diffusion_operator,
    grad_operator,
    NeumannDiffusionBoundary,
    DirichletDiffusionBoundary
)

from logpyle import IntervalTimer, set_dt
from pytools.obj_array import make_obj_array
from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)


#########################################################################

class _SolidGradTempTag:
    pass

class _SolidOperatorTag:
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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_file=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    mesh_filename = "_heat_conduction_45deg-v2.msh"

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 5000
    nrestart = 5000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 2.5e-5/2.0
    t_final = 4.0
    niter = 40000
    local_dt = False
    constant_cfl = False
    current_cfl = 0.2
    
    # discretization and model control
    order = 2
    use_overintegration = False

    temp_wall = 50.0
    wall_penalty_amount = 1.0

##########################################################################
    
    dim = 2

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    if integrator == "compiled_lsrk45":
        from grudge.shortcuts import compiled_lsrk45_step
        timestepper = _compiled_stepper_wrapper
        force_eval = False

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        if constant_cfl is False:
            print(f"\tt_final = {t_final}")
        else:
            print(f"\tconstant_cfl = {constant_cfl}")
            print(f"\tcurrent_cfl = {current_cfl}")
            print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

##############################################################################

    restart_step = None
    if restart_file is None:

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0

        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {"solid": ["ort", "iso"]}
            return mesh, tag_to_elements, volume_to_tags

        local_mesh, global_nelements = distribute_mesh(comm, get_mesh_data)

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        first_step = restart_step+0

        assert comm.Get_size() == restart_data["num_parts"]

    local_nelements = local_mesh["solid"][0].nelements

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in local_mesh.items()},
        order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_solid = DOFDesc(VolumeDomainTag("solid"), DISCR_TAG_BASE)

    from mirgecom.utils import mask_from_elements
    wall_vol_discr = dcoll.discr_from_dd(dd_vol_solid)
    wall_tag_to_elements = local_mesh["solid"][1]
    wall_iso_mask = mask_from_elements(
        dcoll, dd_vol_solid, actx, wall_tag_to_elements["iso"])
    wall_ort_mask = mask_from_elements(
        dcoll, dd_vol_solid, actx, wall_tag_to_elements["ort"])

    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))

    solid_zeros = force_evaluation(actx, solid_nodes[0]*0.0)

    #~~~~~~~~~~

    from grudge.dt_utils import characteristic_lengthscales
    char_length_solid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid)

#####################################################################################

    # {{{ Initialize wall model

    from mirgecom.wall_model import SolidWallModel, SolidWallState, SolidWallConservedVars

    wall_iso_rho = 1.0
    wall_iso_cp = 1.0
    wall_iso_kappa = 1.0

    wall_ort_rho = 1.0
    wall_ort_cp = 1.0
    wall_ort_kappa = np.zeros(2,)
    wall_ort_kappa[0] = 0.1/10
    wall_ort_kappa[1] = 1.0/10

#    wall_iso_rho = 1.0
#    wall_iso_cp = 1.0
#    wall_iso_kappa = 0.01

#    wall_ort_rho = 1.0
#    wall_ort_cp = 1.0
#    wall_ort_kappa = np.zeros(2,)
#    wall_ort_kappa[0] = 1.0
#    wall_ort_kappa[1] = 0.1

#    wall_ort_kappa = np.zeros((2,2))
#    angle = np.deg2rad(45.0)
#    k11 = 0.1
#    k22 = 0.01
#    kxx = +k11*np.cos(angle)**2 + k22*np.sin(angle)**2
#    kxy = -k11*np.sin(angle)*np.cos(angle) + k22*np.sin(angle)*np.cos(angle)
#    kyy = +k11*np.sin(angle)**2 + k22*np.cos(angle)**2
#    wall_ort_kappa[0,0] = kxx
#    wall_ort_kappa[0,1] = kxy
#    wall_ort_kappa[1,0] = kxy
#    wall_ort_kappa[1,1] = kyy
#    
#    print(wall_ort_kappa)

#    print(np.linalg.eig(wall_ort_kappa))

#    n1 = np.array([np.cos(-angle), np.sin(-angle)])
#    n2 = np.array([-np.sin(-angle), np.cos(-angle)])
#    print(n1)
#    print(n2)
#    print(np.dot(np.dot(n1.T,wall_ort_kappa), n1))
#    print(np.dot(np.dot(n2.T,wall_ort_kappa), n2))

#    print(np.sqrt(kxx**2 + kyy**2))
#    print()

#    sys.exit()

    wall_densities = wall_ort_rho*wall_ort_mask + wall_iso_rho*wall_iso_mask

    def _get_solid_enthalpy(temperature, tau):
        wall_iso_h = wall_iso_cp * temperature
        wall_ort_h = wall_ort_cp * temperature
        return wall_ort_h*wall_ort_mask + wall_iso_h*wall_iso_mask

    def _get_solid_heat_capacity(temperature, tau):
        return wall_ort_cp*wall_ort_mask + wall_iso_cp*wall_iso_mask

    def _get_solid_thermal_conductivity(temperature, tau):
        return wall_ort_kappa*wall_ort_mask + wall_iso_kappa*wall_iso_mask

    solid_wall_model = SolidWallModel(
        enthalpy_func=_get_solid_enthalpy,
        heat_capacity_func=_get_solid_heat_capacity,
        thermal_conductivity_func=_get_solid_thermal_conductivity)

    # }}}

    def _get_solid_state(wv):
        wdv = solid_wall_model.dependent_vars(wv=wv)
        return SolidWallState(cv=wv, dv=wdv)

    get_solid_state = actx.compile(_get_solid_state)

#####################################################################################

    def vol_min(dd_vol, x):
        return actx.to_numpy(nodal_min(dcoll, dd_vol, x, initial=np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(nodal_max(dcoll, dd_vol, x, initial=-np.inf))[()]

#########################################################################

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                               
    vis_timer = None
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.5e} s, "),
            ("t_sim.max", "sim time: {value:1.5e} s, "),
            ("t_step.max", "--- step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max",
                                "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")

        from mirgecom.materials.initializer import SolidWallInitializer
        init = SolidWallInitializer(temperature=temp_wall, material_densities=wall_densities)
        current_wv = init(solid_nodes, solid_wall_model)

    else:
        current_step = restart_step
        current_t = restart_data["t"]

        if rank == 0:
            logger.info("Restarting soln.")

        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=restart_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            wall_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_solid),
                restart_dcoll.discr_from_dd(dd_vol_solid))
            current_wv = wall_connection(restart_data["wv"])
        else:
            current_wv = restart_data["wv"]

    if logmgr:
        logmgr_set_time(logmgr, current_step, current_t)

    current_wv = force_evaluation(actx, current_wv)

##############################################################################

    solid_boundaries = {
        dd_vol_solid.trace("left").domain_tag: DirichletDiffusionBoundary(100.0),
        dd_vol_solid.trace("right").domain_tag: DirichletDiffusionBoundary(0.0),
        dd_vol_solid.trace("adiabatic").domain_tag: NeumannDiffusionBoundary(0.0)
    }

##############################################################################

    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

    initname = original_casename
    eosname = None
    init_message = make_init_message(dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=current_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=current_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

#########################################################################

    def my_write_viz(step, t, dt, wv, wdv):
        if rank == 0:
            print('Writing solution file...')

        solid_state = get_solid_state(wv)
        wv = solid_state.cv
#        wdv = solid_state.dv

        rhs, grad = diffusion_operator(
            dcoll, wdv.thermal_conductivity, solid_boundaries,
            wdv.temperature,
            penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag,
            dd=dd_vol_solid, return_grad_u=True)

        solid_viz_fields = [
            ("wv", wv),
            ("cfl", solid_zeros), #FIXME
            ("wall_kappa", wdv.thermal_conductivity),
            ("wall_alpha", solid_wall_model.thermal_diffusivity(solid_state)),
            ("wall_temperature", wdv.temperature),
            ("grad_t", grad),
            ("rhs", rhs),
        ]

        write_visfile(dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        if rank == 0:
            print('Writing restart file...')

        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_file:
            restart_data = {
                "local_mesh": local_mesh,
                "wv": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

#########################################################################

    def my_health_check(cv, dv):
        health_error = False
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(dcoll, "vol", temperature),
                         op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

##############################################################################

    from grudge.dof_desc import DD_VOLUME_ALL
    def my_get_wall_timestep(wv, wdv):
        wall_diffusivity = solid_wall_model.thermal_diffusivity(wv.mass,
            wdv.temperature, wdv.thermal_conductivity)
        return char_length_solid**2/(actx.np.maximum(wall_diffusivity, wdv.oxygen_diffusivity))

    def _my_get_timestep_wall(wv, wdv, t, dt):

        if not constant_cfl:
            return dt

        actx = wv.mass.array_context
        if local_dt:
            mydt = current_cfl*my_get_wall_timestep(wv, wdv)
        else:
            if constant_cfl:
                ts_field = current_cfl*my_get_wall_timestep(wv, wdv)
                mydt = actx.to_numpy(
                    nodal_min(dcoll, dd_vol_solid, ts_field, initial=np.inf))[()]

        return mydt

    my_get_timestep_wall = actx.compile(_my_get_timestep_wall)

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        solid_state = get_solid_state(force_evaluation(actx, state))
        wv = solid_state.cv
        wdv = solid_state.dv

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            ngarbage = 50
            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            if do_health:
                health_errors = global_reduce(my_health_check(wv, wdv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, wv=wv, wdv=wdv)

            if do_restart:
                my_write_restart(step, t, wv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, wv=wv, wdv=wdv)
            raise

        return wv, dt


    def my_rhs(t, state):

        solid_state = _get_solid_state(state)
        wv = solid_state.cv
        wdv = solid_state.dv

        solid_energy_rhs = diffusion_operator(
            dcoll, wdv.thermal_conductivity, solid_boundaries,
            wdv.temperature,
            penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag,
            dd=dd_vol_solid,
            #grad_u=solid_grad_temperature,
            comm_tag=_SolidOperatorTag)

        return SolidWallConservedVars(mass=solid_zeros, energy=solid_energy_rhs)

    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer.start_sub_timer():
                import gc
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections")
                gc.freeze()

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = current_wv
    dt = 1.0*current_dt
    t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt, t=t, t_final=t_final,
                      max_steps=niter, local_dt=local_dt,
                      force_eval=force_eval, state=stepper_state)

    # 
    final_state = _get_solid_state(stepper_state)
    final_wv = final_state.cv
    final_wdv = final_state.dv

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_write_restart(step=final_step, t=final_t, state=stepper_state)

    my_write_viz(step=final_step, t=final_t, dt=current_dt,
                 wv=final_wv, wdv=final_wdv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profiling", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--esdg", action="store_true",
        help="use flux-differencing/entropy stable DG for inviscid computations.")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")

    args = parser.parse_args()

    # for writing output
    casename = "burner_mix"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

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

    main(actx_class, use_logmgr=args.log, casename=casename, 
         restart_file=restart_file)
