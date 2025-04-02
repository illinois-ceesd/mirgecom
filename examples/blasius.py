"""mirgecom driver for laminar flat plate flow (Blasius)."""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
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

import os
import logging
import numpy as np

from logpyle import IntervalTimer, set_dt

from grudge.shortcuts import make_visualizer, compiled_lsrk45_step
from grudge.dof_desc import DISCR_TAG_QUAD, DD_VOLUME_ALL

from mirgecom.discretization import create_discretization_collection
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    write_visfile,
    check_naninf_local,
    distribute_mesh,
    global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PressureOutflowBoundary,
    AdiabaticSlipBoundary,
    IsothermalWallBoundary,
    LinearizedInflowBoundary,
    LinearizedOutflowBoundary,
)
from mirgecom.utils import force_evaluation
from mirgecom.fluid import make_conserved
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time,
)
from arraycontext import get_container_context_recursively

logger = logging.getLogger(__name__)


class Initializer:
    """Initial condition with uniform velocity and smooth temperature profile.

    The smoothing is only applied around the region with no-slip wall to avoid
    diverging calculations.
    """

    def __init__(self, dim, velocity):
        self._dim = dim
        self._velocity = velocity

    def __call__(self, x_vec, eos):
        actx = get_container_context_recursively(x_vec)
        zeros = actx.np.zeros_like(x_vec[0])
        temp_y = 1.0 + actx.np.tanh(1.0/0.01*x_vec[1])
        temp_x = 1.0 + 0.5*(1.0 - actx.np.tanh(1.0/0.01*(x_vec[0]+0.02)))
        temperature = actx.np.maximum(temp_y, temp_x)
        pressure = 1.0 + zeros
        mass = pressure / (eos.gas_const()*temperature)
        velocity = self._velocity
        energy = pressure/(eos.gamma() - 1.0) + 0.5*mass*np.dot(velocity, velocity)
        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mass*velocity)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, use_overintegration, casename, rst_filename, use_esdg):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(True, filename=(f"{casename}.sqlite"),
                               mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    restart_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    snapshot_pattern = restart_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # discretization and model control
    order = 2

    # default i/o frequencies
    nviz = 1
    nrestart = 1000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"

    use_overintegration = False
    local_dt = True
    constant_cfl = True

    niter = 10
    current_dt = 1e-9
    t_final = 2e-8
    if local_dt:
        current_dt = 1e-1
        t_final = 0

    dim = 2
    current_cfl = 0.08
    current_t = 0
    current_step = 0

############################################################

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    timestepper = None
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        # force_eval = False

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

    eos = IdealSingleGas(gamma=1.4, gas_const=1.0)

    from mirgecom.transport import SimpleTransport
    mu = 1.02e-4
    Pr = 0.71  # noqa N806
    kappa = mu*eos.heat_capacity_cp()/Pr
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    gas_model = GasModel(eos=eos, transport=transport_model)

    if rank == 0:
        print("\n#### Simulation fluid properties: ####")
        print(f"\tmu = {mu}")
        if kappa is not None:
            print(f"\tkappa = {kappa}")
        if Pr is not None:
            print(f"\tPrandtl Number  = {Pr}")

############################################################

    local_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    mesh_path = local_path + "blasius.msh"
    omesh_path = local_path + "blasius-v1.msh"
    geo_path = local_path + "blasius.geo"

    restart_step = None
    if rst_filename is None:

        if rank == 0:
            os.system(f"rm -rf {omesh_path} {mesh_path}")
            os.system(f"gmsh {geo_path} -2 -o {omesh_path}")
            os.system(f"gmsh {omesh_path} -save -format msh2 -o {mesh_path}")

            print(f"Reading mesh from {mesh_path}")

        comm.Barrier()

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            # pylint: disable=unpacking-non-sequence
            mesh, tag_to_elements = read_gmsh(
                mesh_path, force_ambient_dim=dim,
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
        restart_data = read_restart_data(actx, rst_filename)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]

        assert comm.Get_size() == restart_data["num_parts"]

############################################################

    if rank == 0:
        logging.info("Making discretization")

    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = nodes[0]*0.0
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None
    dd_vol = DD_VOLUME_ALL

############################################################

    velocity = np.zeros(shape=(dim,))
    velocity[0] = 0.3*np.sqrt(1.4*1.0*2.0)
    flow_init = Initializer(dim=2, velocity=velocity)

    linear_outflow_bnd = LinearizedOutflowBoundary(
        free_stream_density=0.5, free_stream_pressure=1.0,
        free_stream_velocity=velocity)

    linear_inflow_bnd = LinearizedInflowBoundary(
        free_stream_density=0.5, free_stream_pressure=1.0,
        free_stream_velocity=velocity)

    boundaries = {
        dd_vol.trace("inlet").domain_tag:
            linear_inflow_bnd,
        dd_vol.trace("farfield").domain_tag:
            linear_outflow_bnd,
        dd_vol.trace("outlet").domain_tag:
            PressureOutflowBoundary(boundary_pressure=1.0),
        dd_vol.trace("slip").domain_tag:
            AdiabaticSlipBoundary(),
        dd_vol.trace("wall").domain_tag:
            IsothermalWallBoundary(wall_temperature=1.0)
    }

#####################################################################

    def _get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model)

    get_fluid_state = actx.compile(_get_fluid_state)

#####################################################################

    if rst_filename is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = flow_init(x_vec=nodes, eos=eos)
    else:
        current_t = restart_data["t"]
        current_step = restart_step
        if np.isscalar(current_t) is False:
            current_t = actx.to_numpy(actx.np.min(current_t))

        current_cv = restart_data["state"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model)

#####################################################################

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(dcoll)

    initname = "Blasius"
    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

#####################################################################

    def my_write_viz(step, t, dt, state):
        cv = state.cv
        dv = state.dv
        viz_fields = [("CV", cv),
                      ("dt", dt),
                      ("U", cv.velocity[0]),
                      ("V", cv.velocity[1]),
                      ("P", dv.pressure),
                      ("T", dv.temperature),
                      ("Mach", cv.velocity[0]/dv.speed_of_sound)]

        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv):
        rst_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

#####################################################################

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_naninf_local(dcoll, "vol", temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

#####################################################################

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        state = force_evaluation(actx, state)

        fluid_state = get_fluid_state(state)
        dv = fluid_state.dv
        cv = fluid_state.cv

        if local_dt:
            t = force_evaluation(actx, t)
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                constant_cfl=constant_cfl, local_dt=local_dt)
            dt = force_evaluation(actx, actx.np.minimum(dt, current_dt))
        else:
            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                      constant_cfl=constant_cfl, local_dt=local_dt)

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv)

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, state=fluid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, state=fluid_state)
            raise

        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)
        return ns_operator(dcoll, state=fluid_state, time=t,
                           boundaries=boundaries, gas_model=gas_model,
                           quadrature_tag=quadrature_tag)

    def my_post_step(step, t, dt, state):
        min_dt = actx.to_numpy(actx.np.min(dt)) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

#####################################################################

    if local_dt:
        dt = force_evaluation(actx, actx.np.minimum(
            current_dt,
            get_sim_timestep(dcoll, current_state, current_t,
                             current_dt, current_cfl,
                             constant_cfl=constant_cfl, local_dt=local_dt)))
        t = force_evaluation(actx, current_t + zeros)
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_cv) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=current_state.cv,
                      dt=dt, t_final=t_final, t=t,
                      max_steps=niter, local_dt=local_dt,
                      istep=current_step)

    final_state = make_fluid_state(cv=current_cv, gas_model=gas_model)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    dt = get_sim_timestep(dcoll, final_state, current_t, current_dt,
                          current_cfl, constant_cfl=constant_cfl, local_dt=local_dt)
    dt = force_evaluation(actx, actx.np.minimum(dt, current_dt))

    my_write_viz(step=current_step, t=current_t, dt=dt, state=final_state)
    my_write_restart(step=current_step, t=current_t, cv=final_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":

    casename = "blasius"

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
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
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

    from warnings import warn
    warn("Automatically turning off DV logging. MIRGE-Com Issue(578)")

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

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Reading user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    main(actx_class, use_overintegration=args.overintegration or args.esdg,
         casename=casename, rst_filename=rst_filename, use_esdg=args.esdg)

# vim: foldmethod=marker
