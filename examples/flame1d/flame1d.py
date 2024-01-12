"""mirgecom driver for the 1D flame demonstration."""

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
import yaml
import logging
import sys     
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
import cantera
from functools import partial
from pytools.obj_array import make_obj_array

from arraycontext import thaw, freeze

from grudge.dof_desc import DOFDesc, BoundaryDomainTag
from grudge.dof_desc import DD_VOLUME_ALL
from grudge.shortcuts import compiled_lsrk45_step
from grudge.shortcuts import make_visualizer
from grudge import op

from logpyle import IntervalTimer, set_dt
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator, grad_cv_operator, grad_t_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce
)
from mirgecom.utils import force_evaluation
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.fluid import make_conserved
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state, make_operator_fluid_states
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_add_cl_device_info, logmgr_set_time,
    logmgr_add_device_memory_usage)

#######################################################################################

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
    pass


class _FluidOpStatesTag:
    pass


class _FluidGradCVTag:
    pass


class _FluidGradTempTag:
    pass


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""
    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x_min=None, x_max=None, y_min=None, y_max=None, x_thickness=None, y_thickness=None, amplitude):
        r"""Initialize the sponge parameters.
        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_thickness = x_thickness
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.
        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = 0*xpos

        sponge = xpos*0.0

        if (self._x_max is not None):
          x0 = (self._x_max - self._x_thickness)
          dx = +((xpos - x0)/self._x_thickness)
          sponge = sponge + self._amplitude * actx.np.where(
              actx.np.greater(xpos, x0),
                  actx.np.where(actx.np.greater(xpos, self._x_max),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
                  0.0
          )

        if (self._x_min is not None):
          x0 = (self._x_min + self._x_thickness)
          dx = -((xpos - x0)/self._x_thickness)
          sponge = sponge + self._amplitude * actx.np.where(
              actx.np.less(xpos, x0),
                  actx.np.where(actx.np.less(xpos, self._x_min),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
              0.0
          )

        return sponge


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_file=None, use_overintegration=False,
         use_tensor_product_els=False):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    # actx = initialize_actx(actx_class, comm)
    # use_profiling = actx_class_is_profiling(actx_class)
    use_profiling = False
    from grudge.array_context import (
        TensorProductMPIFusionContractorArrayContext,
        TensorProductMPIPyOpenCLArrayContext
    )
    from mirgecom.array_context import initialize_actx

    if use_tensor_product_els:
        # For lazy:
        # actx = initialize_actx(TensorProductMPIFusionContractorArrayContext,
        #                       comm)
        # For eager:
        actx = initialize_actx(TensorProductMPIPyOpenCLArrayContext, comm)
    else:
        actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)

    # ~~~~~~~~~~~~~~~~~~

    import time
    t_start = time.time()
    t_shutdown = 720*60

    # ~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    ngarbage = 10
    nviz = 25000
    nrestart = 25000 
    nhealth = 1
    nstatus = 100

    order = 4
    my_mechanism = "uiuc_7sp"
    cantera_file = "adiabatic_flame_uiuc_7sp_phi1.00_p1.00_E:H0.0_Y.csv"    
    original_rst_file = "flame_1D_025um_C2H4_p=4_uiuc_7sp-000000-0000.pkl"
#    transport = "power-law"
#    transport = "mix-lewis"
    transport = "mix"

    # default timestepping control
    integrator = "compiled_lsrk45"
    current_dt = 2.0e-9
    t_final = 1.0

    niter = 2000000

######################################################

    local_dt = False
    constant_cfl = True
    current_cfl = 0.4

    dim = 2

##############################################################################

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    force_eval_stepper = True
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval_stepper = False

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if (constant_cfl == False):
            print(f"\tcurrent_dt = {current_dt}")
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

        xx = np.loadtxt('x_025um.dat')
        yy = np.loadtxt('y_025um.dat')

        coords = tuple((xx,yy))

        from meshmode.mesh.generation import generate_box_mesh
        from meshmode.mesh import TensorProductElementGroup
        group_cls = TensorProductElementGroup if use_tensor_product_els else None
        generate_mesh = partial(generate_box_mesh,
                                axis_coords=coords,
                                periodic=(False, True),
                                group_cls=group_cls,
                                boundary_tag_to_face={
                                    "inlet": ["-x"],
                                    "outlet": ["+x"]})

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

        assert comm.Get_size() == restart_data["num_parts"]

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx, local_mesh, order,
        use_tensor_product_elements=use_tensor_product_els,
        mpi_communicator=comm)

    nodes = actx.thaw(dcoll.nodes())

    dd_vol = DD_VOLUME_ALL

    zeros = nodes[0]*0.0
    ones = nodes[0]*0.0 + 1.0

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else DISCR_TAG_BASE

##########################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    print("Using Cantera", cantera.__version__)

    # import os
    # current_path = os.path.abspath(os.getcwd()) + "/"
    # mechanism_file = current_path + my_mechanism
    mechanism_file = "uiuc_7sp"

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    cantera_soln.set_equivalence_ratio(phi=1.0, fuel="C2H4:1,H2:1",
                                       oxidizer={"O2": 1.0, "N2": 3.76})

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    if False:
        cantera_data = np.genfromtxt(cantera_file, skip_header=1, delimiter=',')

        temp_unburned = cantera_data[0,2]
        temp_burned = cantera_data[-1,2]

        y_unburned = cantera_data[0,4:]
        y_burned = cantera_data[-1,4:]

        vel_unburned = cantera_data[0,1]
        vel_burned = cantera_data[-1,1]

        mass_unburned = cantera_data[0,3]
        mass_burned = cantera_data[-1,3]

    if True:
        temp_unburned = 300.0

        cantera_soln.TP = temp_unburned, 101325.0
        y_unburned = cantera_soln.Y
        mass_unburned = cantera_soln.density

        cantera_soln.equilibrate("HP")
        temp_burned = cantera_soln.T
        y_burned = cantera_soln.Y
        mass_burned = cantera_soln.density

        vel_unburned = 0.5
        vel_burned = vel_unburned*mass_unburned/mass_burned

    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm
    pres_unburned = one_atm
    pres_burned = one_atm

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = \
        get_pyrometheus_wrapper_class_from_cantera(cantera_soln)(actx.np)
    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=1350.0)

    species_names = pyrometheus_mechanism.species_names

    # }}}    

    if transport == "power-law":
        from mirgecom.transport import PowerLawTransport
        transport_model = PowerLawTransport(lewis=np.ones((nspecies,)), beta=4.093e-7)

    if transport == "mix-lewis":
        from mirgecom.transport import MixtureAveragedTransport
        transport_model = MixtureAveragedTransport(pyrometheus_mechanism,
                                                   lewis=np.ones(nspecies,))
    if transport == "mix":
        from mirgecom.transport import MixtureAveragedTransport
        transport_model = MixtureAveragedTransport(pyrometheus_mechanism)

    gas_model = GasModel(eos=eos, transport=transport_model)

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned:")
    print(f"T = {temp_unburned}")
    print(f"D = {mass_unburned}")
    print(f"Y = {y_unburned}")
    print(f"U = {vel_unburned}")
    print(f" ")
    print(f"Burned:")
    print(f"T = {temp_burned}")
    print(f"D = {mass_burned}")
    print(f"Y = {y_burned}")
    print(f"U = {vel_burned}")
    print(f" ")

###############################################################################

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(
            gas_model.eos.get_internal_energy(temperature,
                                              species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity))

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed,
            limiter_func=_limit_fluid_cv
        )

    get_fluid_state = actx.compile(_get_fluid_state)
    
##############################################################################

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                               
    vis_timer = None
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.6e} s, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")
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

        current_t = 0.0
        current_step = 0
        first_step = 0

        flame_start_loc = 0.0

        tseed = 1234.56789

        from mirgecom.initializers import PlanarDiscontinuity
        # use the burned conditions with a lower temperature
        bulk_init = PlanarDiscontinuity(dim=dim, disc_location=flame_start_loc,
            sigma=0.0005, nspecies=nspecies,
            temperature_right=temp_burned, temperature_left=temp_unburned,
            pressure_right=pres_burned, pressure_left=pres_unburned,
            velocity_right=make_obj_array([vel_burned, 0.0]),
            velocity_left=make_obj_array([vel_unburned, 0.0]),
            species_mass_right=y_burned, species_mass_left=y_unburned)

        if rank == 0:
            logging.info("Initializing soln.")
        # for Discontinuity initial conditions
        current_cv = bulk_init(x_vec=nodes, eos=eos, time=0.)

    else:

        if local_dt:
            current_t = restart_data["step"]
        else:
            current_t = restart_data["t"]
        current_step = restart_step
        first_step = restart_step + 0

        if restart_order != order:
            print('Wrong order...')
            sys.exit()
        else:
            current_cv = restart_data["cv"]
            tseed = restart_data["temperature_seed"]

    if logmgr:
        logmgr_set_time(logmgr, current_step, current_t)

    current_state = get_fluid_state(current_cv, tseed)
    current_state = force_evaluation(actx, current_state)

##############################################################################

    # initialize the sponge field
    sponge_x_thickness = 0.065
    sponge_amp = 10000.0

    xMaxLoc = +0.100
    xMinLoc = -0.100
        
    sponge_init = InitSponge(x_max=xMaxLoc,
                             x_min=xMinLoc,
                             x_thickness=sponge_x_thickness,
                             amplitude=sponge_amp)

    sponge_sigma = sponge_init(x_vec=nodes)

    if False:
        cantera_data = read_restart_data(actx, original_rst_file)
        ref_cv = force_evaluation(actx, cantera_data["cv"])
    if True:
        ref_cv = force_evaluation(actx, bulk_init(x_vec=nodes, eos=eos, time=0.))

#################################################################

    from mirgecom.boundary import (
        PrescribedFluidBoundary,PressureOutflowBoundary,
        LinearizedOutflow2DBoundary, LinearizedInflow2DBoundary)

    inflow_cv_cond = op.project(dcoll, dd_vol, dd_vol.trace("inlet"), ref_cv)
    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
                                temperature_seed=300.0)

    inflow_bnd = PrescribedFluidBoundary(
                      boundary_state_func=inlet_bnd_state_func)

#    inflow_bnd = LinearizedInflow2DBoundary(
#        free_stream_density=mass_unburned, free_stream_pressure=101325.0,
#        free_stream_velocity=make_obj_array([vel_unburned, 0.0]),
#        free_stream_species_mass_fractions=y_unburned)

    outflow_bnd = LinearizedOutflow2DBoundary(
        free_stream_density=mass_burned, free_stream_pressure=101325.0,
        free_stream_velocity=make_obj_array([vel_burned, 0.0]),
        free_stream_species_mass_fractions=y_burned)
#    outflow_bnd = PressureOutflowBoundary(boundary_pressure=101325.0)

    boundaries = {BoundaryDomainTag("inlet"): inflow_bnd,
                  BoundaryDomainTag("outlet"): outflow_bnd}

####################################################################################

    visualizer = make_visualizer(dcoll)

    initname = "flame1D"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        t_initial=current_t, dt=current_dt, t_final=t_final, nstatus=nstatus,
        nviz=nviz, cfl=current_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

##################################################################

#    def get_production_rates(cv, temperature):
#        return make_obj_array([eos.get_production_rates(cv, temperature)])
#    compute_production_rates = actx.compile(get_production_rates)

    def my_write_viz(step, t, dt, state):

        y = state.cv.species_mass_fractions
        gas_const = gas_model.eos.gas_const(cv=state.cv)

        # reaction_rates, = compute_production_rates(state.cv, state.temperature)
        viz_fields = [("CV_rho", state.cv.mass),
                      ("CV_rhoU", state.cv.momentum),
                      ("CV_rhoE", state.cv.energy),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      ("DV_U", state.velocity[0]),
                      ("DV_V", state.velocity[1]),
                      # ("reaction_rates", reaction_rates),
                      ("sponge", sponge_sigma),
                      ("R", gas_const),
                      ("gamma", eos.gamma(state.cv, state.temperature)),
                      ("dt", dt),
                      ("mu", state.tv.viscosity),
                      ("kappa", state.tv.thermal_conductivity),
                      ]

        # species mass fractions
        viz_fields.extend(("Y_"+species_names[i], y[i]) for i in range(nspecies))

        # species diffusivity
        viz_fields.extend(
            ("diff_"+species_names[i], state.tv.species_diffusivity[i])
                for i in range(nspecies))

        print('Writing solution file...')
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, cv, tseed):
        restart_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "temperature_seed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }

            write_restart_file(actx, rst_data, restart_fname, comm)

##################################################################

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

##############################################################################

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed = state
        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)

        fluid_state = get_fluid_state(cv, tseed)

        if not use_tensor_product_els:
            if local_dt:
                t = force_evaluation(actx, t)
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                      gas_model, constant_cfl=constant_cfl,
                                      local_dt=local_dt)
                dt = force_evaluation(actx, dt)
                # dt = force_evaluation(actx, actx.np.minimum(dt, current_dt))
            elif constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                      t_final, constant_cfl, local_dt)

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_garbage = check_step(step=step, interval=ngarbage)

            t_elapsed = time.time() - t_start
            if t_shutdown - t_elapsed < 300.0:
                my_write_restart(step=step, t=t, cv=fluid_state.cv,
                                 tseed=tseed)
                sys.exit()

            if do_garbage:
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            if do_health:
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=fluid_state.cv,
                                 tseed=tseed)

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, state=fluid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, state=fluid_state)
            raise

        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt

    def my_rhs(t, state):
        cv, tseed = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=tseed, limiter_func=_limit_fluid_cv)

        operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries, quadrature_tag,
            comm_tag=_FluidOpStatesTag, limiter_func=_limit_fluid_cv)

        grad_cv = grad_cv_operator(
            dcoll, gas_model, boundaries, fluid_state, time=t,
            quadrature_tag=quadrature_tag, comm_tag=_FluidGradCVTag,
            limiter_func=_limit_fluid_cv,
            operator_states_quad=operator_states_quad)

        grad_t = grad_t_operator(
            dcoll, gas_model, boundaries, fluid_state, time=t,
            quadrature_tag=quadrature_tag, comm_tag=_FluidGradTempTag,
            limiter_func=_limit_fluid_cv,
            operator_states_quad=operator_states_quad)

        ns_rhs = ns_operator(dcoll, gas_model, fluid_state, boundaries, time=t,
            quadrature_tag=quadrature_tag, grad_cv=grad_cv, grad_t=grad_t,
            operator_states_quad=operator_states_quad)

        chem_rhs = eos.get_species_source_terms(fluid_state.cv,
                                                fluid_state.temperature)

        #sponge_rhs = sponge_func(cv=fluid_state.cv, cv_ref=ref_cv,
        #                         sigma=sponge_sigma)
        rhs = ns_rhs + chem_rhs #+ sponge_rhs

        return make_obj_array([rhs, zeros])

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

        min_dt = np.min(actx.to_numpy(dt)) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

    if not use_tensor_product_els:
        current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                      current_cfl, t_final, constant_cfl)
        current_dt = force_evaluation(actx, current_dt)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt, t=current_t,
                      t_final=t_final, max_steps=niter, local_dt=local_dt,
                      force_eval=force_eval_stepper,
                      state=make_obj_array([current_state.cv, tseed]),
                      # compile_rhs=False
                      )
    current_cv, tseed = stepper_state
    current_state = make_fluid_state(current_cv, gas_model, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_write_viz(step=current_step, t=current_t, dt=current_dt,
                 state=current_state)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
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
    parser.add_argument("--overintegration", action="store_true", default=False,
                        help="enable overintegration [OFF]")
    parser.add_argument("--esdg", action="store_true",
                        help="use entropy stable DG for inviscid computations.")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--numpy", action="store_true",
                        help="use numpy-based eager actx.")
    parser.add_argument("--use_tensor_product_elements", action="store_true")

    args = parser.parse_args()

    # for writing output
    casename = "flame1D"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if args.input_file:
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
        lazy=args.lazy, distributed=True, profiling=args.profiling,
        numpy=args.numpy)

    main(actx_class, use_logmgr=args.log, casename=casename,
         restart_file=restart_file, use_overintegration=args.overintegration,
         use_tensor_product_els=args.use_tensor_product_elements)

# vim: foldmethod=marker
