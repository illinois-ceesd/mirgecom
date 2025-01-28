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
import numpy as np
import time 
from functools import partial

from meshmode.mesh import BTAG_ALL
from meshmode.dof_array import DOFArray
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_QUAD

from logpyle import IntervalTimer, set_dt

from mirgecom.mpi import mpi_entry_point
from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    get_sim_timestep,
    distribute_mesh
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import (
    Uniform,
    AcousticPulse
)
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)

from mirgecom.fluid import (
    make_conserved
)
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    set_sim_state
)
from mirgecom.utils import (
    force_evaluation
)


logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass

from pytato.transform.parameter_study import ParameterStudyAxisTag
class NicksVelocityStudy(ParameterStudyAxisTag):
    pass


@mpi_entry_point
def main(actx_class, use_esdg=False,
         use_overintegration=False, use_leap=False,
         casename=None, rst_filename=None, nel_1d:int = 16, order:int=1,
         num_uq=None):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(False,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    from arraycontext.parameter_study import pack_for_parameter_study



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
    t_final = 1.0
    current_cfl = 1.0
    current_dt = .005
    current_t = 0
    constant_cfl = False

    # some i/o frequencies
    nstatus = 1
    nrestart = -5
    nviz = -1 
    nhealth = -1

    dim = 2
    rst_path = "restart_data/"
    rst_pattern = None
    if num_uq is not None:
        rst_pattern = rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    else:
        rst_path + "{cname}-{step:04d}-{rank:04d}-{num_uq:04d}.pkl"
    
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}-{num_uq:04d}.pkl"
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
        if nel_1d is None:
            nel_1d = 16
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(box_ll,)*dim, b=(box_ur,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            periodic=(True,)*dim
        )

        local_mesh, global_nelements = distribute_mesh(comm, generate_mesh)
        local_nelements = local_mesh.nelements

    if order is None:
        order = 1
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

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

    velocity = np.zeros(shape=(4,dim,))
    if num_uq is None:
        velocity[0] = np.zeros(shape=(dim,))
        velocity[1] = np.ones(shape=(dim,))
        velocity[2] = -np.ones(shape=(dim,))
        velocity[3] = np.array([np.sqrt(3)/2, 1/2])
    else:
        velocity = np.random.random(size=(num_uq, dim,))

    uncertain_velocity = pack_for_parameter_study(actx, NicksVelocityStudy,
                              *[actx.from_numpy(velocity[i]) for i in range(len(velocity))])


    assert uncertain_velocity.shape == velocity.shape[::-1]
    assert uncertain_velocity[:,-1].shape == velocity[-1].shape
    def compiled_setup(nodes_used, pressure, rho, velocity):
        initializer = Uniform(velocity=velocity, pressure=1.0, rho=1.0)
        # Pass eos through context but not the func args.
        # This way we do not need to handle splitting it.
        uniform_state = initializer(nodes_used, eos=eos)

        acoustic_pulse = AcousticPulse(dim=dim, amplitude=0.5, width=0.1, center=orig)

        initial_cv = None
        if rst_filename:
            current_t = restart_data["t"]
            current_step = restart_data["step"]
            initial_cv = restart_data["cv"]
        else:
            # Set the current state from time 0
            initial_cv = acoustic_pulse(x_vec=nodes, cv=uniform_state, eos=eos)

        return initial_cv 

    orig = np.zeros(shape=(dim,))
    my_func = actx.compile(compiled_setup)
    current_cv = my_func(nodes, 1.0, 1.0, uncertain_velocity)

    single_cv = [compiled_setup(nodes, 1.0, 1.0, velocity[i]) for i in range(len(velocity))]

    def get_initial_state(initial_cv):
        return make_fluid_state(initial_cv, gas_model)
    convert_cv_to_state = actx.compile(get_initial_state)

    current_state = convert_cv_to_state(current_cv)
    single_states = [convert_cv_to_state(single_cv[i]) for i in range(len(single_cv))]

    from arraycontext.parameter_study import unpack_parameter_study

    boundaries = {BTAG_ALL: AdiabaticSlipBoundary()}
    boundaries = {}


    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, current_step, current_t)

    visualizer = make_visualizer(dcoll)

    UNCERTAIN = True
    def my_write_viz(step, t, state, dv=None): 
        if dv is None:
            dv = eos.dependent_vars(state)
        from arraycontext.parameter_study import unpack_parameter_study
        from mirgecom.simutil import write_visfile
        if UNCERTAIN:
            # Evaluate first.
            # We need to rebuild the state and then write out the visualization file UQ times.
            mass_vals = unpack_parameter_study(state.mass, NicksVelocityStudy)
            energy_vals = unpack_parameter_study(state.energy, NicksVelocityStudy)
            momentum_vals = unpack_parameter_study(state.momentum, NicksVelocityStudy)

            num_uq = len(mass_vals[0]) # Index in to avoid the DoFArray.
            actx = state.array_context
            for i in range(num_uq):
                q = np.empty(2+state.dim, dtype="O")
                q[0] = DOFArray(actx, (mass_vals[0][i],))
                q[1] = DOFArray(actx, (energy_vals[0][i],))
                for k in range(state.dim):
                    q[2+k] = DOFArray(actx, (momentum_vals[k][0][i],))

                # Build the system as a q list.
                tmp_state = make_conserved(dim=state.dim, q=q)
                dv = eos.dependent_vars(tmp_state)
                viz_fields = [("cv", tmp_state),
                              ("dv", dv)]

                write_visfile(dcoll, viz_fields, visualizer, vizname=casename + "_uq_" + str(i),
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)
        else:
            write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                  step=step, t=t, overwrite=True, vis_timer=vis_timer,
                  comm=comm)


    #initname = "pulse"
    #eosname = eos.__class__.__name__
    """
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)




    def my_health_check(pressure):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: NAN or INF Pressure found.")
        if check_range_local(dcoll, "vol", pressure, 0.8, 1.6):
            health_error = True
            if rank ==0: 
                print("OUTSIDE RANGE!!!!!!!!")
                print(f"Max pressure {actx.max(pressure)}")
            logger.info(f"{rank=}: Pressure outside range.")
        return health_error
    """
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
                #health_errors = global_reduce(my_health_check(dv.pressure), op="lor")
                health_errors = None
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
            #my_write_viz(step=step, t=t, state=state)
            #my_write_restart(step=step, t=t, state=state)
            raise

        dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl, t_final,
                              constant_cfl)
        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        """
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        """
        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)
        return euler_operator(dcoll, state=fluid_state, time=t,
                              boundaries=boundaries,
                              gas_model=gas_model, use_esdg=use_esdg,
                              quadrature_tag=quadrature_tag)

    # This is just for the first timestep.
    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    start_clock = time.monotonic_ns()
    # State needs to be the current_cv so that we can build the fluid state from it.
    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_cv, t=current_t, t_final=t_final)
    end_clock = time.monotonic_ns()

    def assert_arrays_close(arr1, arr2):

        mask1 = np.isfinite(arr1)
        mask2 = np.isfinite(arr2)

        assert np.allclose(mask1, mask2) # Both NANs in the same locations.
        assert np.allclose(arr1[mask1], arr2[mask2])


    finish_tol = 1e-16
    time_data = []
    for i in range(len(velocity)): # These are going to be the singles.
        single_t = 0
        single_step = 0
        current_cfl = 1.0
        current_dt = .005
        start_single = time.monotonic_ns()

        single_step, single_t, single_cv[i] = \
                advance_state(rhs=my_rhs, timestepper=timestepper,
                              pre_step_callback=my_pre_step,
                              post_step_callback=my_post_step, dt=current_dt,
                              state=single_cv[i], t=single_t, t_final=t_final)

        end_single = time.monotonic_ns()

        time_data.append(end_single - start_single)
        if rank == 0:
            print(f"Advancing the state 1 by 1 took: {end_single - start_single} (ns) for {i+1} of {len(velocity)} states.")

        assert single_step == current_step # We made the same number of steps.
        assert np.abs(single_t - t_final) < finish_tol

        for j in range(dim):
            saved_momentum = actx.to_numpy(current_cv.momentum[j][0][...,i])
            correct_momentum = actx.to_numpy(single_cv[i].momentum[j][0])
            assert_arrays_close(saved_momentum, correct_momentum)
        assert np.allclose(actx.to_numpy(current_cv.nspecies),
                     actx.to_numpy(single_cv[i].nspecies))
        # Index into the DoFArray
        saved_mass = actx.to_numpy(current_cv.mass[0])[..., i]
        correct_mass = actx.to_numpy(single_cv[i].mass[0])
        assert_arrays_close(saved_mass, correct_mass)
        
        saved_energy = actx.to_numpy(current_cv.energy[0])[...,i]
        correct_energy = actx.to_numpy(single_cv[i].energy[0])
        assert_arrays_close(saved_energy, correct_energy)
        
        saved_speed = actx.to_numpy(current_cv.speed[0])[..., i]
        correct_speed = actx.to_numpy(single_cv[i].speed[0])
        assert_arrays_close(saved_speed, correct_speed)


    # Dump the final data
    if rank == 0:
        print(f"Advancing the states together took: {end_clock - start_clock} (ns)")
        print(f"Average cost for 1 by 1: {np.mean(time_data)} (ns)")
        logger.info("Checkpointing final state ...")
    final_state = make_fluid_state(current_cv, gas_model)
    final_dv = final_state.dv

    #my_write_viz(step=current_step, t=current_t, state=current_cv, dv=final_dv)
    #my_write_restart(step=current_step, t=current_t, state=current_cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    assert np.abs(current_t - t_final) < finish_tol

    if rank == 0:
        nicks_timing_data_file = "nicks_timing_data_file.txt"
        import os
        if os.path.isfile(nicks_timing_data_file):
            # Then, we know we can just add the next data line.
            # Num 1d elem, Order, Num Uq, Num Timesteps, All Together time (ns),
            # Avg 1-by-1 time (ns)
            with open(nicks_timing_data_file, "a+") as my_file:
                my_file.write(str(nel_1d))
                my_file.write(",")
                my_file.write(str(order))
                my_file.write(",")
                my_file.write(str(len(velocity)))
                my_file.write(",")
                my_file.write(str(current_step))
                my_file.write(",")
                my_file.write(str(end_clock - start_clock))
                my_file.write(",")
                my_file.write(str(np.mean(time_data)))
                my_file.write("\r\n")


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
    parser.add_argument("--elms",type=int, help="Number of 1d elements.")
    parser.add_argument("--order",type=int, help="Order of the elements.")
    parser.add_argument("--uncertain",type=int, help="Number of uncertain trails.")
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

    logging.basicConfig(format="%(message)s", level=logging.ERROR)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_esdg=args.esdg,
         use_overintegration=args.overintegration or args.esdg,
         use_leap=args.leap,
         casename=casename, rst_filename=rst_filename,
         nel_1d=args.elms, order=args.order, num_uq=args.uncertain)

# vim: foldmethod=marker
