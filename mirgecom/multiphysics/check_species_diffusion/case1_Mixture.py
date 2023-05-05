"""."""

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
#import yaml
import logging
import sys
import numpy as np
import pyopencl as cl
#import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce,
#    force_evaluation
)
from pytools.obj_array import make_obj_array
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from grudge.shortcuts import compiled_lsrk45_step
from mirgecom.steppers import advance_state
from mirgecom.fluid import make_conserved
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    PressureOutflowBoundary
)
from mirgecom.diffusion import (
    diffusion_operator,
    NeumannDiffusionBoundary
)
from mirgecom.transport import SimpleTransport, MixtureAveragedTransport
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
import cantera

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time, #LogUserQuantity,
    set_sim_state
)

from grudge.dof_desc import DD_VOLUME_ALL

from pytools.obj_array import make_obj_array

from mirgecom.utils import force_evaluation

############################################################################

"Ihme's paper, case #1"
class case1:

    def __init__(
            self, *, dim=2, sigma=1.0, nspecies, velocity,
                     species_mass_right, species_mass_left):
                   
        self._dim = dim
        self._nspecies = nspecies 
        self._sigma = sigma
        self._vel = velocity
        self._yu = species_mass_right
        self._yb = species_mass_left
        self._disc = 2.5

    def __call__(self, x_vec, eos, *, time=0.0):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        x = x_vec[0]
        actx = x.array_context
        
        u_x = x*0.0 + self._vel[0]
        velocity = make_obj_array([u_x])
        
#        aux = 0.5*(1.0 - actx.np.tanh( 1.0/(self._sigma)*(x_vec[0]) ))
#        y1 = self._yu*aux
#        y2 = self._yb*(1.0-aux)
#        
#        y = y1+y2
        
        aux1 = - actx.np.tanh( 1.0/(self._sigma)*(x_vec[0] - (self._disc + 10.0*time) ) )
        aux2 = + actx.np.tanh( 1.0/(self._sigma)*(x_vec[0] + (self._disc - 10.0*time) ) )
        aux = 0.5*(aux1 + aux2)
        y1 = self._yu*aux
        y2 = self._yb*(1.0-aux)
        
        y = y1+y2

#        aux1 = - actx.np.tanh( 1.0/(self._sigma)*(x_vec[0] - (self._disc + 10.0*time) ) )
#        aux2 = + actx.np.tanh( 1.0/(self._sigma)*(x_vec[0] + (self._disc - 10.0*time) ) )
#        aux = 0.5*(aux1 + aux2)
#        theta = 7.0
#        temperature = 1.0 + aux*theta
#        temperature = temperature*300.0
        temperature = 300.0 + x*0.0
        
        pressure = 101325.0 + x*0.0

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        specmass = mass * y
        momentum = velocity*mass        
        internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass*(internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum, species_mass=specmass)

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


#h1 = logging.StreamHandler(sys.stdout)
#f1 = SingleLevelFilter(logging.INFO, False)
#h1.addFilter(f1)
#root_logger = logging.getLogger()
#root_logger.addHandler(h1)
#h2 = logging.StreamHandler(sys.stderr)
#f2 = SingleLevelFilter(logging.INFO, True)
#h2.addFilter(f2)
#root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, casename="mixtransp",
         restart_filename=None, use_profiling=False, use_logmgr=False, lazy=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()

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

    # ~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 250
    nrestart = 1000
    nhealth = 1
    nstatus = 1

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 1e-3
    t_final = 100.0

    current_cfl = 0.4
    constant_cfl = False
    local_dt = False

    # discretization and model control
    order = 4
    fuel = "Ar"

    dim = 1
    
    niter = int(t_final/current_dt)

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval = False

##############################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.

    if fuel == "Ar":
        mechanism_file = "inert"
    if fuel == "C2H4":
        mechanism_file = "uiuc"
    elif fuel == "H2":
        mechanism_file = "sanDiego"

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

#    cantera_soln()

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    
    y_burned = np.zeros(nspecies)
    y_unburned = np.zeros(nspecies)

    i_Ar = cantera_soln.species_index("Ar")
    i_He = cantera_soln.species_index("He")
    y_burned[i_Ar] = 1.0 #-1e-7
    y_burned[i_He] = 0.0 #1e-7
    y_unburned[i_Ar] = 0.0 #1e-7
    y_unburned[i_He] = 1.0 #-1e-7

    one_atm = 101325.0
    pres_unburned = one_atm

    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPY = temp_unburned, pres_unburned, y_unburned
    can_t, rho_unburned, y_unburned = cantera_soln.TDY

    cantera_soln.TPY = temp_unburned, pres_unburned, y_burned
    can_t, rho_burned, y_burned = cantera_soln.TDY

###########################

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = \
        get_pyrometheus_wrapper_class_from_cantera(cantera_soln, temperature_niter=3)(actx.np)

    species_names = pyrometheus_mechanism.species_names

    transport_model = MixtureAveragedTransport(pyrometheus_mechanism)
#    transport_model = SimpleTransport(viscosity=1e-4,
#        thermal_conductivity=1e-4, species_diffusivity=1e-4*np.ones((nspecies,)))

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=temp_unburned)
    gas_model = GasModel(eos=eos, transport=transport_model)

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"(Y) = ({y_unburned}")
    print(f"(Y) = ({y_burned}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter
    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=False, dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=DD_VOLUME_ALL)

    get_fluid_state = actx.compile(_get_fluid_state)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    velocity = np.zeros(shape=(dim,))
    velocity[0] = 0.0

    # use the burned conditions with a lower temperature
    flow_init = case1(dim=dim, sigma=0.1, nspecies=nspecies,
                      velocity=velocity,
                      species_mass_right=y_unburned,
                      species_mass_left=y_burned)

    restart_step = None
    if restart_file is None:

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0

        nel_1d = 201

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(-5.0,)*dim, b=(+5.0,)*dim,
            periodic=(False,),
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={"inlet": ["-x"], "outlet": ["+x"]}
        )

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        first_step = restart_step+0

        assert comm.Get_size() == restart_data["num_parts"]

    if rank == 0:
        print('Making discretization')
        logging.info("Making discretization")
    
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order)
    nodes = actx.thaw(dcoll.nodes())

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    use_overintegration = False
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    dd_vol = DD_VOLUME_ALL

#################################################################

    if restart_file is None:
        current_t = 0.0
        current_step = 0
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = flow_init(x_vec=nodes, eos=eos, time=0.)
        temperature_seed = can_t + nodes[0]*0.0
    else:
        current_t = restart_data["t"]
        current_step = restart_step

        if restart_order != order:
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

            current_cv = connection(restart_data["state"])
            temperature_seed = connection(restart_data["temperature_seed"])
        else:
            current_cv = restart_data["state"]
            temperature_seed = restart_data["temperature_seed"]

#    if logmgr:
#        logmgr_set_time(logmgr, current_step, current_t)

    current_cv = force_evaluation(actx, current_cv)
    temperature_seed = force_evaluation(actx, temperature_seed)

    current_state = get_fluid_state(current_cv, temperature_seed)

#################################################################

    vis_timer = None
    #log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.5e} s, "),
            ("t_sim.max", "sim time: {value:1.5e} s, "),
            ("t_step.max", "--- step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)
        
#################################################################

    bnd_nodes = force_evaluation(actx, dcoll.nodes(dd_vol.trace('inlet')))
    bnd_cv = flow_init(x_vec=bnd_nodes, eos=eos, time=0.)
    bnd_cv = force_evaluation(actx, bnd_cv)
    def _inflow_boundary_state(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return make_fluid_state(bnd_cv, gas_model, temp_unburned)

    boundaries = {dd_vol.trace("inlet").domain_tag:
                    PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state),
                  dd_vol.trace("outlet").domain_tag: 
                    PressureOutflowBoundary()}

####################################################################

    visualizer = make_visualizer(dcoll)

    initname = "mixtransp"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     t_initial=current_t, cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

####################################################################

    def my_write_viz(step, t, state, species_rhs=None, species_grad=None):       
          
        cv = state.cv
        dv = state.dv
        tv = state.tv

        mmw = pyrometheus_mechanism.get_mix_molecular_weight(cv.species_mass_fractions)

        #reaction_rates, = compute_production_rates(cv, dv.temperature)
        viz_fields = [("CV_rho", cv.mass),
                      ("CV_rhoU", cv.momentum[0]),
                      ("CV_rhoE", cv.energy),
                      ("DV_U", cv.velocity[0]),
                      ("DV_P", dv.pressure),
                      ("DV_T", dv.temperature),
                      ("TV_mu", tv.viscosity),
                      ("TV_kappa", tv.thermal_conductivity),
                      ("mmw", mmw),
                      ]

        #species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], cv.species_mass_fractions[i])
            for i in range(nspecies))

        viz_fields.extend(
            ("TV_Y_"+species_names[i], tv.species_diffusivity[i])
            for i in range(nspecies))

        # depending on the version, paraview may complain without this
        viz_fields.append(("x", nodes[0]))

        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, cv, temperature_seed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                "temperature_seed": temperature_seed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(dcoll, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed = state
        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)

        fluid_state = get_fluid_state(cv, tseed)

        cv = fluid_state.cv

        try:
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                  t_final, constant_cfl, local_dt, dd_vol)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
#                health_errors = global_reduce(
#                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                health_errors = my_health_check(fluid_state.cv, fluid_state.dv)
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_viz:

                rhs, grad_cv, grad_t = ns_operator(
                    dcoll, gas_model, fluid_state, boundaries,
                    return_gradients=True, inviscid_terms_on=False)

                species_rhs = rhs.species_mass_fractions
                species_grad = grad_cv.species_mass_fractions

#                diff_boundaries = {
#                    BoundaryDomainTag("inlet"):
#                        DirichletDiffusionBoundary(1.0),
#                    BoundaryDomainTag("outlet"):
#                        DirichletDiffusionBoundary(0.0)
#                }

#                species_rhs, species_grad = diffusion_operator(
#                    dcoll, fluid_state.tv.species_diffusivity, diff_boundaries,
#                    fluid_state.cv.species_mass_fractions,
#                    return_grad_u=True,
#                    penalty_amount=1.0, quadrature_tag=DISCR_TAG_BASE,
#                    dd=DD_VOLUME_ALL)

                my_write_viz(step=step, t=t, state=fluid_state,
                                 species_rhs=species_rhs, species_grad=species_grad)

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv, temperature_seed=tseed)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state)
            raise

        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt

    def my_rhs(t, state):
        cv, tseed = state
        
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol)

        cv_rhs = ns_operator(dcoll, state=fluid_state, time=t,
                             boundaries=boundaries, gas_model=gas_model,
                             inviscid_terms_on=False)
        
        return make_obj_array([cv_rhs, 0*tseed])

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl, local_dt,
                                  dd_vol)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=make_obj_array([current_state.cv, temperature_seed]),
                      dt=current_dt, t_final=t_final, t=current_t,
                      istep=current_step)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    current_cv, tseed = stepper_state
    current_state = make_fluid_state(current_cv, gas_model, tseed)
    final_dv = current_state.dv
    ts_field, cfl, dt = my_get_timestep(current_t, current_dt, current_state)
    my_write_viz(step=current_step, t=current_t, state=current_state)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

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
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "mixtransp"
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

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile, casename=casename,
         lazy=args.lazy, restart_filename=restart_file)
