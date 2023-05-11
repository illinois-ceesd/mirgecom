"""Illustrates the coupling between two domains governed by NS equations"""

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

import logging
import sys
import gc
import numpy as np
import pyopencl as cl

from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, DISCR_TAG_BASE, DISCR_TAG_QUAD,
    VolumeDomainTag
)

from mirgecom.utils import force_evaluation
from mirgecom.discretization import create_discretization_collection
from mirgecom.simutil import (
    check_step, get_sim_timestep, distribute_mesh, write_visfile,
    check_naninf_local, global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    AdiabaticSlipBoundary,
    PrescribedFluidBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import PowerLawTransport
import cantera
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)
from mirgecom.multiphysics.multiphysics_coupled_fluid_wall import (
    coupled_ns_operator
)

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array


class Initializer:

    def __init__(
            self, nspecies, velocity,
            species_mass_right, species_mass_left, *, dim=2, sigma=1.0):

        self._dim = dim
        self._nspecies = nspecies
        self._sigma = sigma
        self._vel = velocity
        self._yu = species_mass_right
        self._yb = species_mass_left
        self._disc = 1.0

    def __call__(self, x_vec, eos, *, time=0.0):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        x = x_vec[1]
        actx = x.array_context

        u_x = x*0.0 + self._vel[0]
        u_y = x*0.0 + self._vel[1]
        velocity = make_obj_array([u_x, u_y])

        aux = 0.5*(1.0 - actx.np.tanh(1.0/(self._sigma)*(x - self._disc)))
        y1 = self._yu*aux
        y2 = self._yb*(1.0-aux)

        y = y1+y2

        aux = 0.5*(1.0 - actx.np.tanh(1.0/(self._sigma)*(x - self._disc)))
        temperature = 300.0 + 300.0*aux

        pressure = 101325.0 + x*0.0

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        specmass = mass * y
        momentum = velocity*mass
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)
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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_profiling=False, casename="ns_coupling", lazy=False,
         restart_filename=None):
    """Driver the coupled NS example."""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

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
    nviz = 500
    nrestart = 10000
    nhealth = 10
    nstatus = 100
    ngarbage = 50

    # default timestepping control
    integrator = "ssprk43"
    current_dt = 1.0
    t_final = 10.0

    local_dt = False
    constant_cfl = False
    current_cfl = 0.4

    # discretization and model control
    order = 2
    use_overintegration = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dim = 2

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    if integrator == "compiled_lsrk45":
        from grudge.shortcuts import compiled_lsrk45_step
        timestepper = _compiled_stepper_wrapper
        force_eval = False

    if integrator == "ssprk43":
        from mirgecom.integrators.ssprk import ssprk43_step
        timestepper = ssprk43_step
        force_eval = True

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
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")


##########################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    mechanism_file = "uiuc"

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_cantera = 300.0

    x_left = np.zeros(nspecies)
    x_left[cantera_soln.species_index("O2")] = 1.0
    x_left[cantera_soln.species_index("N2")] = 0.0

    pres_cantera = cantera.one_atm  # pylint: disable=no-member

    cantera_soln.TPX = temp_cantera, pres_cantera, x_left
    y_left = cantera_soln.Y

    x_right = np.zeros(nspecies)
    x_right[cantera_soln.species_index("O2")] = 0.0
    x_right[cantera_soln.species_index("N2")] = 1.0

    cantera_soln.TPX = temp_cantera, pres_cantera, x_right
    y_right = cantera_soln.Y

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 300.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    species_names = pyrometheus_mechanism.species_names
    print(f"Pyrometheus mechanism species names {species_names}")

    # }}}

    fluid_transport_model = PowerLawTransport(lewis=np.ones((nspecies,)),
       beta=4.093e-7*0.2)

    solid_transport_model = PowerLawTransport(lewis=np.ones((nspecies,)),
       beta=4.093e-7)

    gas_model_fluid = GasModel(eos=eos, transport=fluid_transport_model)
    gas_model_solid = GasModel(eos=eos, transport=solid_transport_model)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_fluid_cv(cv, pressure, temperature, dd):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
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
        energy_lim = mass_lim*(gas_model_fluid.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_fluid,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

    get_fluid_state = actx.compile(_get_fluid_state)

    # ~~~~~~~~~~~~~~~~~~~~

    def _limit_solid_cv(cv, pressure, temperature, dd):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
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
        energy_lim = mass_lim*(gas_model_solid.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

    def _get_solid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_solid,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_solid)

    get_solid_state = actx.compile(_get_solid_state)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    velocity = np.zeros(shape=(dim,))

    flow_init = Initializer(dim=dim, sigma=0.2, nspecies=nspecies,
        velocity=velocity, species_mass_right=y_right, species_mass_left=y_left)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import os
    if rank == 0:
        os.system("rm -rf ns-coupled.msh ns-coupled-v2.msh")
        os.system("gmsh ns-coupled.geo -2 ns-coupled.msh")
        os.system("gmsh ns-coupled.msh -save -format msh2 -o ns-coupled-v2.msh")
    else:
        os.system("sleep 1s")

    restart_step = None
    if restart_filename:
        rst_filename = f"{restart_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        volume_to_local_mesh = restart_data["volume_to_local_mesh"]
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_ranks"] == num_ranks
    else:  # import the grid
        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                "ns-coupled-v2.msh", force_ambient_dim=2,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "Fluid": ["Upper"],
                "Solid": ["Lower"]}
            return mesh, tag_to_elements, volume_to_tags

        def partition_generator_func(mesh, tag_to_elements, num_ranks):
            from meshmode.distributed import get_partition_by_pymetis
            return get_partition_by_pymetis(mesh, num_ranks)

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data, partition_generator_func)
        volume_to_local_mesh = {
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()}

    local_fluid_mesh = volume_to_local_mesh["Fluid"]
    local_solid_mesh = volume_to_local_mesh["Solid"]

    local_nelements = local_fluid_mesh.nelements + local_solid_mesh.nelements

    dcoll = create_discretization_collection(actx, volume_to_local_mesh,
                                             order=order)

    dd_vol_fluid = DOFDesc(VolumeDomainTag("Fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("Solid"), DISCR_TAG_BASE)

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))

    fluid_zeros = force_evaluation(actx, fluid_nodes[0]*0.0)
    solid_zeros = force_evaluation(actx, solid_nodes[0]*0.0)

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if restart_file is None:
        current_step = 0
        current_t = 0.0
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = flow_init(fluid_nodes, eos)
        current_wv = flow_init(solid_nodes, eos)

        tseed = force_evaluation(actx, 300.0 + fluid_zeros)
        wv_tseed = force_evaluation(actx, 300.0 + solid_zeros)

    else:
        if rank == 0:
            logger.info("Restarting soln.")

        current_step = restart_step
        current_t = restart_data["t"]

        current_cv = restart_data["cv"]
        tseed = restart_data["temperature_seed"]
        current_wv = restart_data["wv"]
        wv_tseed = restart_data["wall_temperature_seed"]

    first_step = force_evaluation(actx, current_step)

    current_cv = force_evaluation(actx, current_cv)
    tseed = force_evaluation(actx, tseed)
    fluid_state = get_fluid_state(current_cv, tseed)

    current_wv = force_evaluation(actx, current_wv)
    wv_tseed = force_evaluation(actx, wv_tseed)
    solid_state = get_solid_state(current_wv, wv_tseed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{num_ranks}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    vis_timer = None
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fluid_boundary_solution(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        cv = flow_init(nodes, gas_model.eos)
        return make_fluid_state(cv=cv,
                                gas_model=gas_model,
                                temperature_seed=nodes[0]*0.0+300.0)

    def solid_boundary_solution(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        cv = flow_init(nodes, gas_model.eos)
        return make_fluid_state(cv=cv,
                                gas_model=gas_model,
                                temperature_seed=nodes[0]*0.0+600.0)

    fluid_boundaries = {
        dd_vol_fluid.trace("Upper Top").domain_tag:
        PrescribedFluidBoundary(boundary_state_func=fluid_boundary_solution),
        dd_vol_fluid.trace("Upper Sides").domain_tag:
        AdiabaticSlipBoundary(),
    }

    solid_boundaries = {
        dd_vol_solid.trace("Lower Bottom").domain_tag:
        PrescribedFluidBoundary(boundary_state_func=solid_boundary_solution),
        dd_vol_solid.trace("Lower Sides").domain_tag:
        AdiabaticSlipBoundary()
    }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

    initname = original_casename
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        dt=current_dt, t_final=t_final, nstatus=nstatus, nviz=nviz,
        t_initial=current_t, cfl=current_cfl, constant_cfl=constant_cfl,
        initname=initname, eosname=eosname, casename=casename)

    if rank == 0:
        logger.info(init_message)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_write_viz(step, t, dt, fluid_state, wall_state):

        fluid_viz_fields = [
            ("CV_rho", fluid_state.cv.mass),
            ("CV_rhoU", fluid_state.cv.momentum),
            ("CV_rhoE", fluid_state.cv.energy),
            ("DV_P", fluid_state.pressure),
            ("DV_T", fluid_state.temperature)]

        # species mass fractions
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        solid_viz_fields = [
            ("CV_rho", wall_state.cv.mass),
            ("CV_rhoU", wall_state.cv.momentum),
            ("CV_rhoE", wall_state.cv.energy),
            ("DV_P", wall_state.pressure),
            ("DV_T", wall_state.temperature)]

        # species mass fractions
        solid_viz_fields.extend(
            ("Y_"+species_names[i], wall_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        write_visfile(dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t,
            overwrite=True, comm=comm)
        write_visfile(dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        if rank == 0:
            print("Writing restart file...")

        cv, tseed, wv, wv_tseed = state
        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "nspecies": nspecies,
                "wv": wv,
                "wall_temperature_seed": wv_tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_ranks": num_ranks
            }

            write_restart_file(actx, restart_data, restart_fname, comm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(dcoll, "vol", pressure),
                         op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature),
                         op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv, tseed, wv, wv_tseed = state

        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)
        wv = force_evaluation(actx, wv)
        wv_tseed = force_evaluation(actx, wv_tseed)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed)
        cv = fluid_state.cv

        # construct species-limited fluid state
        solid_state = get_solid_state(wv, wv_tseed)
        wv = solid_state.cv

        try:

            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    gc.collect()

            state = make_obj_array([fluid_state.cv, fluid_state.temperature,
                                    solid_state.cv, solid_state.temperature])

            if check_step(step=step, interval=nhealth):
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if check_step(step=step, interval=nviz):
                print("Writing solution file")
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    wall_state=solid_state)

            if check_step(step=step, interval=nrestart):
                my_write_restart(step, t, state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                         wall_state=solid_state)
            raise

        return state, dt

    def my_rhs(t, state):
        cv, tseed, wv, wv_tseed = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model_fluid,
            temperature_seed=tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

        wall_state = make_fluid_state(cv=wv, gas_model=gas_model_solid,
            temperature_seed=wv_tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_solid)

        fluid_rhs, solid_rhs = coupled_ns_operator(
            dcoll, gas_model_fluid, gas_model_solid, dd_vol_fluid, dd_vol_solid,
            fluid_boundaries, solid_boundaries, fluid_state, wall_state,
            time=t, limiter_func=_limit_fluid_cv,
            inviscid_fluid_terms_on=False, inviscid_wall_terms_on=False,
            quadrature_tag=quadrature_tag)

        return make_obj_array([fluid_rhs, fluid_zeros, solid_rhs, solid_zeros])

    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer.start_sub_timer():
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections")
                gc.freeze()

        min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    stepper_state = make_obj_array([fluid_state.cv, fluid_state.temperature,
                                    solid_state.cv, solid_state.temperature])

    if constant_cfl:
        t = current_t
        dt = current_dt
        dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
            t_final, constant_cfl, local_dt, dd_vol_fluid)
    else:
        dt = 1.0*current_dt
        t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=dt, t=t, t_final=t_final,
                      force_eval=force_eval, state=stepper_state)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    final_cv, tseed, final_wv, wv_tseed = stepper_state

    final_fluid_state = get_fluid_state(final_cv, tseed)
    final_solid_state = get_solid_state(final_wv, wv_tseed)

    my_write_restart(step=final_step, t=final_t, state=stepper_state)

    my_write_viz(step=final_step, t=final_t, dt=current_dt,
                 fluid_state=final_fluid_state, wall_state=final_solid_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    sys.exit()


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
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    if (args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if (args.input_file):
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
