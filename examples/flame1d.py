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
import logging
import sys
import gc
from warnings import warn
from functools import partial
import numpy as np
import cantera
from pytools.obj_array import make_obj_array

from grudge.dof_desc import BoundaryDomainTag
from grudge.shortcuts import compiled_lsrk45_step
from mirgecom.integrators import rk4_step, euler_step
from grudge.shortcuts import make_visualizer
from grudge import op

from logpyle import IntervalTimer, set_dt
from mirgecom.navierstokes import ns_operator, grad_cv_operator, grad_t_operator
from mirgecom.simutil import (
    check_step, check_naninf_local, check_range_local,
    get_sim_timestep,
    distribute_mesh,
    write_visfile,
)
from mirgecom.utils import force_evaluation
from mirgecom.restart import write_restart_file, read_restart_data
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.fluid import make_conserved
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import (
    GasModel, make_fluid_state, make_operator_fluid_states)
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_add_cl_device_info, logmgr_set_time,
    logmgr_add_device_memory_usage)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return record.levelno != self.passlevel
        return record.levelno == self.passlevel


# h1 = logging.StreamHandler(sys.stdout)
# f1 = SingleLevelFilter(logging.INFO, False)
# h1.addFilter(f1)
# root_logger = logging.getLogger()
# root_logger.addHandler(h1)
# h2 = logging.StreamHandler(sys.stderr)
# f2 = SingleLevelFilter(logging.INFO, True)
# h2.addFilter(f2)
# root_logger.addHandler(h2)

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
    """Apply the sponge."""
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""
    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, x_min, x_max, x_thickness, amplitude):
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
        self._x_thickness = x_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        """
        xpos = x_vec[0]
        actx = xpos.array_context

        sponge = xpos*0.0

        x0 = self._x_max - self._x_thickness
        dx = +(xpos - x0)/self._x_thickness
        sponge = sponge + self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            actx.np.where(
                actx.np.greater(xpos, self._x_max), 1., 3.*dx**2 - 2.*dx**3),
            0.)

        x0 = self._x_min + self._x_thickness
        dx = -(xpos - x0)/self._x_thickness
        sponge = sponge + self._amplitude * actx.np.where(
            actx.np.less(xpos, x0),
            actx.np.where(
                actx.np.less(xpos, self._x_min), 1., 3.*dx**2 - 2.*dx**3),
            0.)

        return sponge


@mpi_entry_point
def main(actx_class, use_esdg=False, use_tpe=False, use_overintegration=False,
         use_leap=False, casename=None, rst_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(True, filename=(f"{casename}.sqlite"),
                               mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # ~~~~~~~~~~~~~~~~~~

    import time
    t_start = time.time()
    t_shutdown = 720*60

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    ngarbage = 10
    nviz = 1000
    nrestart = 25000
    nhealth = 1
    nstatus = 100

    mechanism_file = "uiuc_7sp"

    order = 2

    transport = "power-law"
    # transport = "mix-lewis"
    # transport = "mix"

    # default timestepping control
    integrator = "rk4"
    constant_cfl = False
    current_cfl = 0.4
    current_dt = 1.0e-9
    niter = 1000000
    t_final = current_dt * niter

    use_sponge = False

    # use Cantera's 1D flame solution to prescribe the BC and an
    # approximated initial condition with hyperbolic tangent profile
    use_flame_from_cantera = True

# ############################################################################

    dim = 2

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    force_eval_stepper = True
    timestepper = rk4_step
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval_stepper = False
    if integrator == "euler":
        timestepper = euler_step

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if constant_cfl:
            print(f"\tcurrent_cfl = {current_cfl}")
        else:
            print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

# ############################################################################

    restart_step = 0
    if rst_filename is None:

        import os
        path = os.path.dirname(os.path.abspath(__file__))
        xx = np.loadtxt(f"{path}/flame1d_x_050um.dat")
        yy = np.loadtxt(f"{path}/flame1d_y_050um.dat")

        from meshmode.mesh import TensorProductElementGroup
        grp_cls = TensorProductElementGroup if use_tpe else None

        from meshmode.mesh.generation import generate_box_mesh
        generate_mesh = partial(generate_box_mesh,
                                axis_coords=(xx, yy),
                                periodic=(False, True),
                                boundary_tag_to_face={"inlet": ["-x"],
                                                      "outlet": ["+x"]},
                                group_cls=grp_cls)

        local_mesh, global_nelements = distribute_mesh(comm, generate_mesh)
        local_nelements = local_mesh.nelements

    else:
        restart_file = f"{rst_filename}-{rank:04d}.pkl"
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert restart_order == order
        assert comm.Get_size() == restart_data["num_parts"]

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order)

    nodes = actx.thaw(dcoll.nodes())
    zeros = actx.np.zeros_like(nodes[0])

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else DISCR_TAG_BASE

# ############################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    if rank == 0:
        logging.info("\nUsing Cantera " + cantera.__version__)

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    cantera_soln.set_equivalence_ratio(phi=1.0, fuel="C2H4:1,H2:1",
                                       oxidizer={"O2": 1.0, "N2": 3.76})

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    if use_flame_from_cantera:
        # Set up flame object
        f = cantera.FreeFlame(cantera_soln, width=0.02)

        # Solve with mixture-averaged transport model
        f.transport_model = "mixture-averaged"
        f.set_refine_criteria(ratio=2, slope=0.15, curve=0.15)
        f.solve(loglevel=0, refine_grid=True, auto=True)

        temp_unburned = f.T[0]
        temp_burned = f.T[-1]

        y_unburned = f.Y[:, 0]
        y_burned = f.Y[:, -1]

        vel_unburned = f.velocity[0]
        vel_burned = f.velocity[-1]

        mass_unburned = f.density[0]
        mass_burned = f.density[-1]

    else:
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

    pres_unburned = cantera.one_atm  # pylint: disable=no-member
    pres_burned = cantera.one_atm  # pylint: disable=no-member

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyro_mech = get_pyrometheus_wrapper_class_from_cantera(cantera_soln)(actx.np)
    eos = PyrometheusMixture(pyro_mech, temperature_guess=1234.56789)

    species_names = pyro_mech.species_names

    # }}}

    if transport == "power-law":
        from mirgecom.transport import PowerLawTransport
        transport_model = PowerLawTransport(lewis=np.ones((nspecies,)),
                                            beta=4.093e-7)

    if transport == "mix-lewis":
        from mirgecom.transport import MixtureAveragedTransport
        transport_model = MixtureAveragedTransport(pyro_mech,
                                                   lewis=np.ones(nspecies,))
    if transport == "mix":
        from mirgecom.transport import MixtureAveragedTransport
        transport_model = MixtureAveragedTransport(pyro_mech)

    gas_model = GasModel(eos=eos, transport=transport_model)

    print(f"Pyrometheus mechanism species names {species_names}")
    print("Unburned:")
    print(f"T = {temp_unburned}")
    print(f"D = {mass_unburned}")
    print(f"Y = {y_unburned}")
    print(f"U = {vel_unburned}\n")
    print("Burned:")
    print(f"T = {temp_burned}")
    print(f"D = {mass_burned}")
    print(f"Y = {y_burned}")
    print(f"U = {vel_burned}\n")

# ############################################################################

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_fluid_cv(cv, temperature_seed, gas_model, dd=None):

        temperature = gas_model.eos.temperature(
            cv=cv, temperature_seed=temperature_seed)
        pressure = gas_model.eos.pressure(cv=cv, temperature=temperature)

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                                     mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)])

        # normalize to ensure sum_Yi = 1.0
        aux = actx.np.zeros_like(cv.mass)
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
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv)

    get_fluid_state = actx.compile(_get_fluid_state)

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = eos.internal_energy(cv) / cv.mass
        return make_obj_array(
            [pyro_mech.get_temperature_update_energy(e, temperature, y)]
        )

    compute_temperature_update = actx.compile(get_temperature_update)

# ############################################################################

    flame_start_loc = 0.0
    from mirgecom.initializers import PlanarDiscontinuity
    bulk_init = PlanarDiscontinuity(dim=dim, disc_location=flame_start_loc,
        sigma=0.0005, nspecies=nspecies,
        temperature_right=temp_burned, temperature_left=temp_unburned,
        pressure_right=pres_burned, pressure_left=pres_unburned,
        velocity_right=make_obj_array([vel_burned, 0.0]),
        velocity_left=make_obj_array([vel_unburned, 0.0]),
        species_mass_right=y_burned, species_mass_left=y_unburned)

    if rst_filename is None:
        current_t = 0.0
        current_step = 0
        first_step = 0

        tseed = 1234.56789 + zeros

        if rank == 0:
            logging.info("Initializing soln.")

        current_cv = bulk_init(x_vec=nodes, eos=eos, time=0.)

    else:
        current_t = restart_data["t"]
        current_step = restart_step
        first_step = restart_step + 0

        current_cv = restart_data["cv"]
        tseed = restart_data["temperature_seed"]

    tseed = force_evaluation(actx, tseed)
    current_cv = force_evaluation(actx, current_cv)
    current_state = get_fluid_state(current_cv, tseed)

# ############################################################################

    vis_timer = None
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

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

    # initialize the sponge field
    sponge_init = InitSponge(x_max=+0.100, x_min=-0.100, x_thickness=0.065,
                             amplitude=10000.0)

    sponge_sigma = sponge_init(x_vec=nodes)

    ref_cv = bulk_init(x_vec=nodes, eos=eos, time=0.)

# ############################################################################

    # from grudge.dof_desc import DD_VOLUME_ALL
    # dd_vol = DD_VOLUME_ALL

    # inflow_cv_cond = op.project(dcoll, dd_vol, dd_vol.trace("inlet"), ref_cv)

    # def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
    #     return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
    #                             temperature_seed=300.0)

    # from mirgecom.boundary import (
    #     PrescribedFluidBoundary, LinearizedOutflow2DBoundary)
    # inflow_bnd = PrescribedFluidBoundary(boundary_state_func=inlet_bnd_state_func)
    # outflow_bnd = LinearizedOutflow2DBoundary(
    #     free_stream_density=mass_burned, free_stream_pressure=101325.0,
    #     free_stream_velocity=make_obj_array([vel_burned, 0.0]),
    #     free_stream_species_mass_fractions=y_burned)

    from mirgecom.boundary import (
        LinearizedInflowBoundary, PressureOutflowBoundary)
    inflow_bnd = LinearizedInflowBoundary(
        free_stream_density=mass_unburned, free_stream_pressure=101325.0,
        free_stream_velocity=make_obj_array([vel_unburned, 0.0]),
        free_stream_species_mass_fractions=y_unburned)
    outflow_bnd = PressureOutflowBoundary(boundary_pressure=101325.0)

    boundaries = {BoundaryDomainTag("inlet"): inflow_bnd,
                  BoundaryDomainTag("outlet"): outflow_bnd}

# ############################################################################

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

# ############################################################################

    # def get_production_rates(cv, temperature):
    #     return make_obj_array([eos.get_production_rates(cv, temperature)])
    # compute_production_rates = actx.compile(get_production_rates)

    def my_write_viz(step, t, dt, state):

        y = state.cv.species_mass_fractions
        # gas_const = gas_model.eos.gas_const(species_mass_fractions=y)
        # gamma = eos.gamma(state.cv, state.temperature)

        # reaction_rates, = compute_production_rates(state.cv, state.temperature)
        viz_fields = [("CV_rho", state.cv.mass),
                      ("CV_rhoU", state.cv.momentum),
                      ("CV_rhoE", state.cv.energy),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      # ("reaction_rates", reaction_rates),
                      # ("sponge", sponge_sigma),
                      # ("R", gas_const),
                      # ("gamma", gamma),
                      # ("dt", dt),
                      # ("mu", state.tv.viscosity),
                      # ("kappa", state.tv.thermal_conductivity),
                      ]

        # species mass fractions
        viz_fields.extend(("Y_"+species_names[i], y[i]) for i in range(nspecies))

        # species diffusivity
        # viz_fields.extend(
        #     ("diff_"+species_names[i], state.tv.species_diffusivity[i])
        #     for i in range(nspecies))

        if rank == 0:
            logger.info("Writing solution file...")
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, cv, tseed):
        restart_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != rst_filename:
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

# ############################################################################

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

        # if check_range_local(dcoll, "vol", pressure, 101250., 101500.):
        #    health_error = True
        #    logger.info(f"{rank=}: Pressure range violation.")

        if check_range_local(dcoll, "vol", temperature, 290., 2450.):
            health_error = True
            logger.info(f"{rank=}: Temperature range violation.")

        # temperature_update is the next temperature update in the
        # `get_temperature` Newton solve. The relative size of this
        # update is used to gauge convergence of the current temperature
        # after a fixed number of Newton iters.
        # Note: The local max jig below works around a very long compile
        # in lazy mode.
        temp_update, = compute_temperature_update(cv, temperature)
        temp_resid = force_evaluation(actx, temp_update) / temperature
        temp_resid = (actx.to_numpy(op.nodal_max_loc(dcoll, "vol", temp_resid)))
        if temp_resid > 1e-8:
            health_error = True
            logger.info(f"{rank=}: Temperature is not converged {temp_resid=}.")

        return health_error

# ############################################################################

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed = state
        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)

        fluid_state = get_fluid_state(cv, tseed)

        if constant_cfl:
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                  t_final, constant_cfl)

        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_garbage = check_step(step=step, interval=ngarbage)

            t_elapsed = time.time() - t_start
            if t_shutdown - t_elapsed < 300.0:
                my_write_restart(step=step, t=t, cv=fluid_state.cv, tseed=tseed)

            if do_garbage:
                with gc_timer.start_sub_timer():
                    warn("Running gc.collect() to work around memory growth issue ")
                    gc.collect()

            if do_health:
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=fluid_state.cv, tseed=tseed)

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

        rhs = ns_rhs + chem_rhs
        if use_sponge:
            sponge_rhs = sponge_func(cv=fluid_state.cv, cv_ref=ref_cv,
                                     sigma=sponge_sigma)
            rhs = rhs + sponge_rhs

        return make_obj_array([rhs, zeros])

    def my_post_step(step, t, dt, state):
        if step == first_step + 1:
            with gc_timer.start_sub_timer():
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

# ############################################################################

    if constant_cfl:
        current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                      current_cfl, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt, t=current_t,
                      t_final=t_final, force_eval=force_eval_stepper,
                      state=make_obj_array([current_state.cv, tseed]))
    current_cv, tseed = stepper_state
    current_state = make_fluid_state(current_cv, gas_model, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_write_viz(step=current_step, t=current_t, dt=current_dt,
                 state=current_state)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     tseed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    sys.exit()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

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
    parser.add_argument("--leap", action="store_true",
                        help="use leap timestepper")
    parser.add_argument("--profiling", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
#    parser.add_argument("--log", action="store_true", default=True,
#                        help="enable logging profiling [ON]")
    parser.add_argument("--overintegration", action="store_true", default=False,
                        help="enable overintegration [OFF]")
    parser.add_argument("--esdg", action="store_true",
                        help="use entropy stable DG for inviscid computations.")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--numpy", action="store_true",
                        help="use numpy-based eager actx.")
    parser.add_argument("--tpe", action="store_true")

    args = parser.parse_args()

    # for writing output
    casename = "flame1D"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    rst_filename = None
    if args.restart_file:
        rst_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {rst_filename}")

    input_file = None
    if args.input_file:
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    main(actx_class, use_leap=args.leap, use_esdg=args.esdg, use_tpe=args.tpe,
         use_overintegration=args.overintegration or args.esdg,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
