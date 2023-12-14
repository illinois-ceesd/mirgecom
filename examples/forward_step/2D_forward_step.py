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
from mirgecom.initializers import PlanarDiscontinuity, Uniform
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state, make_operator_fluid_states
from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging

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
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):
    """Drive the example."""
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

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    mesh_filename = "mesh_v3-v2.msh"

    # timestepping control
    current_step = 0
    t_final = 60.0
    current_cfl = 0.2
    current_dt = 1e-2
    current_t = 0
    integrator = "compiled_lsrk45"

    order = 2

    use_AV = False
    use_overintegration = False
    local_dt = True
    constant_cfl = True

    # some i/o frequencies
    nrestart = 1000
    nstatus = 1
    nviz = 100
    nhealth = 1

    niter = 134001

    dim = 2
    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")

    from grudge.shortcuts import compiled_lsrk45_step
    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    force_eval = True
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval = False

###################################################

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

        assert comm.Get_size() == restart_data["num_parts"]

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    nodes = actx.thaw(dcoll.nodes())

##############################################################################

    if use_AV:
        s0 = np.log10(1.0e-4 / np.power(order, 4))
        alpha = 1.0e-3
        kappa_av = 0.5
        av_species = 0.0
    else:
        s0 = np.log10(1.0e-4 / np.power(order, 4))
        alpha = 0.0
        kappa_av = 0.5
        av_species = 0.0

##############################################################################

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
            ("dt.max", "dt: {value:1.6e} s, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")
            ])

        #logmgr_add_device_memory_usage(logmgr, queue)
        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

    # soln setup and init
    eos = IdealSingleGas(gamma=1.4, gas_const=1.0)

    from mirgecom.transport import SimpleTransport#, ArtificialViscosityTransport
    kappa = 0.0
    mu = 0.0
    physical_transport = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)

    if use_AV:
        s0 = np.log10(1.0e-4 / np.power(order, 4))
        alpha = 1.0e-3
        kappa_av = 0.5

        transport_model = ArtificialViscosityTransport(
            physical_transport=physical_transport, av_mu=alpha, av_prandtl=0.71
        )

        gas_model = GasModel(eos=eos, transport=transport_model)
    else:
        gas_model = GasModel(eos=eos, transport=physical_transport)

###############################

    from grudge.dof_desc import DD_VOLUME_ALL
    from mirgecom.limiter import bound_preserving_limiter
    def _limit_fluid_cv(cv, pressure=None, temperature=None, tseed=None, dd=DD_VOLUME_ALL):

        mass_lim = bound_preserving_limiter(dcoll, cv.mass, mmin=1e-5, mmax=3.0, modify_average=True, dd=dd)
        pressure = bound_preserving_limiter(dcoll, pressure, mmin=1e-5, mmax=3.0, modify_average=True, dd=dd)

        velocity = cv.velocity
        energy_lim = pressure/(eos.gamma()-1.0) + 0.5*mass_lim*np.dot(velocity, velocity)

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                              momentum=mass_lim*velocity)


    from mirgecom.artificial_viscosity import smoothness_indicator
    def _get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                limiter_func=_limit_fluid_cv)
    get_fluid_state = actx.compile(_get_fluid_state)

###############################

    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[0] = 3.0*np.sqrt(1.4*1.0*0.5)
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
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
            limiter_func=_limit_fluid_cv, limiter_dd=dd_bdry
        )

    inflow_bnd = PrescribedFluidBoundary(boundary_state_func=inlet_bnd_state_func)
    outflow_bnd = DummyBoundary()
    wall_bnd = AdiabaticSlipBoundary()

    from grudge.dof_desc import DTAG_BOUNDARY
    boundaries = {DTAG_BOUNDARY("inlet"): inflow_bnd,
                  DTAG_BOUNDARY("outlet"): outflow_bnd,
                  DTAG_BOUNDARY("wall"): wall_bnd}

#############################################################################

    current_state = make_fluid_state(current_cv, gas_model,
            limiter_func=_limit_fluid_cv)

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

    def my_write_viz(step, t, state, smoothness=None):
                     
        viz_fields = [("CV", state.cv),
                      ("DV_U", state.cv.velocity),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      ("TV_mu", state.tv.viscosity),
                      ("TV_kappa", state.tv.thermal_conductivity),
                      ("smoothness", smoothness)
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
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        return health_error


    from grudge.dof_desc import DD_VOLUME_ALL
    from grudge.discretization import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_MODAL
    from meshmode.transform_metadata import FirstAxisIsElementsTag
    from meshmode.dof_array import DOFArray
    def drop_order(dcoll, field, theta, positivity_preserving=False, dd=None):
        # Compute cell averages of the state
        def cancel_polynomials(grp):
            return actx.from_numpy(
                np.asarray([1 if sum(mode_id) == 0
                            else 0 for mode_id in grp.mode_ids()]))

        # map from nodal to modal
        if dd is None:
            dd = DD_VOLUME_ALL

        dd_nodal = dd
        dd_modal = dd_nodal.with_discr_tag(DISCR_TAG_MODAL)

        modal_map = dcoll.connection_from_dds(dd_nodal, dd_modal)
        nodal_map = dcoll.connection_from_dds(dd_modal, dd_nodal)

        modal_discr = dcoll.discr_from_dd(dd_modal)
        modal_field = modal_map(field)

        # cancel the ``high-order"" polynomials p > 0 and keep the average
        filtered_modal_field = DOFArray(
            actx,
            tuple(actx.einsum("ej,j->ej",
                              vec_i,
                              cancel_polynomials(grp),
                              arg_names=("vec", "filter"),
                              tagged=(FirstAxisIsElementsTag(),))
                  for grp, vec_i in zip(modal_discr.groups, modal_field))
        )

        # convert back to nodal to have the average at all points
        cell_avgs = nodal_map(filtered_modal_field)

        if positivity_preserving:
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, 1e-5),
                                                      cell_avgs, 1e-5)    

        return theta*(field - cell_avgs) + cell_avgs

    from pytools.obj_array import make_obj_array
    def _drop_order_cv(cv, flipped_smoothness, theta_factor, dd=None):

        smoothness = 1.0 - theta_factor*flipped_smoothness

        density_lim = drop_order(dcoll, cv.mass, smoothness)
        momentum_lim = make_obj_array([
            drop_order(dcoll, cv.momentum[0], smoothness),
            drop_order(dcoll, cv.momentum[1], smoothness)])
        energy_lim = drop_order(dcoll, cv.energy, smoothness)


        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=density_lim, energy=energy_lim,
                              momentum=momentum_lim, species_mass=cv.species_mass)

    drop_order_cv = actx.compile(_drop_order_cv)

#########################################################################

    def my_pre_step(step, t, dt, state):
        if logmgr:
            logmgr.tick_before()

        cv = force_evaluation(actx, state)

        smoothness = smoothness_indicator(dcoll, cv.mass,
                                          kappa=kappa_av, s0=s0)
    
        cv = drop_order_cv(cv, smoothness, 0.05)

        fluid_state = get_fluid_state(cv=cv)
        fluid_state = force_evaluation(actx, fluid_state)

        cv = fluid_state.cv
        dv = fluid_state.dv

        if local_dt:
            t = force_evaluation(actx, t)
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                 gas_model, constant_cfl=constant_cfl, local_dt=local_dt)
            dt = force_evaluation(actx, actx.np.minimum(dt, current_dt))
        else:
            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                      t_final, constant_cfl, local_dt)

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
                my_write_viz(step=step, t=t, state=fluid_state,
                             smoothness=smoothness)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state)
            my_write_restart(step=step, t=t, state=cv)
            raise

        return cv, dt

    def my_rhs(t, state):
        smoothness = smoothness_indicator(dcoll, state.mass,
                                          kappa=kappa_av, s0=s0)
        cv = _drop_order_cv(state, smoothness, 0.05)

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
            limiter_func=_limit_fluid_cv)

        operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries, quadrature_tag,
            comm_tag=_FluidOpStatesTag, limiter_func=_limit_fluid_cv)
     
        return ns_operator(dcoll, gas_model, fluid_state, boundaries,
            time=t, operator_states_quad=operator_states_quad,
            quadrature_tag=quadrature_tag, limiter_func=_limit_fluid_cv)

    def my_post_step(step, t, dt, state):
        min_dt = np.min(actx.to_numpy(dt))
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()
        return state, dt

#################################################################

    if local_dt == True:
        dt = (#force_evaluation(actx, 
             get_sim_timestep(dcoll, current_state, current_t,
                     current_dt, current_cfl, gas_model,
                     constant_cfl=constant_cfl, local_dt=local_dt)
        )
        dt = force_evaluation(actx, actx.np.minimum(current_dt, dt))

        t = force_evaluation(actx, current_t + zeros)
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=dt,
                      max_steps=niter, local_dt=local_dt,
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

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "step"
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

    main(actx_class, use_logmgr=args.log, use_leap=args.leap, lazy=lazy,
         use_profiling=args.profiling, casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
