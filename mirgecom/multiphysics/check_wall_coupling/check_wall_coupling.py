""" Fri 10 Mar 2023 02:22:21 PM CST """

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
import numpy as np
import pyopencl as cl

from functools import partial

from dataclasses import dataclass, fields

from arraycontext import (
    dataclass_array_container, with_container_arithmetic,
    get_container_context_recursively
)

from meshmode.dof_array import DOFArray

from grudge import op
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, as_dofdesc, DISCR_TAG_BASE, BoundaryDomainTag, VolumeDomainTag
)
from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.utils import force_evaluation
from mirgecom.discretization import create_discretization_collection
from mirgecom.simutil import (
    check_step, get_sim_timestep, distribute_mesh, write_visfile,
    check_naninf_local, check_range_local, global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    IsothermalWallBoundary,
    PressureOutflowBoundary,
    AdiabaticSlipBoundary,
    PrescribedFluidBoundary,
    AdiabaticNoslipWallBoundary,
    LinearizedOutflowBoundary
)
from mirgecom.fluid import (
    velocity_gradient, species_mass_fraction_gradient, make_conserved
)
from mirgecom.transport import (
    PowerLawTransport,
    MixtureAveragedTransport
)
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
    # coupled_grad_t_operator,
    coupled_ns_operator
)
from mirgecom.diffusion import (
    diffusion_operator, DirichletDiffusionBoundary, NeumannDiffusionBoundary
)

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array

#########################################################################

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
        self._disc = 10.0

    def __call__(self, x_vec, eos, *, time=0.0):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        x = x_vec[1]
        actx = x.array_context
        
        u_x = x*0.0 + self._vel[0]
        u_y = x*0.0 + self._vel[1]
        velocity = make_obj_array([u_x, u_y])
        
        aux1 = - actx.np.tanh( 1.0/(self._sigma)*(x_vec[1] - self._disc ) )
        aux2 = + actx.np.tanh( 1.0/(self._sigma)*(x_vec[1] + self._disc ) )
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


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return DOFArray(actx, tuple(group_arrays))


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_filename=None):

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
    nviz = 25000
    nrestart = 25000
    nhealth = 1
    nstatus = 100

    plot_gradients = True

    # default timestepping control
#    integrator = "compiled_lsrk45"
    integrator = "ssprk43"
    current_dt = 2.5e-6 #order == 2
    t_final = 2.0

    local_dt = False
    constant_cfl = True
    current_cfl = 0.4
    
    # discretization and model control
    order = 4
    use_overintegration = False

    speedup_factor = 1

#    transport = "Mixture"
    transport = "PowerLaw"

#    # wall stuff
#    temp_wall = 300

#    wall_penalty_amount = 1.0
#    wall_time_scale = 1 #speedup_factor

#    use_radiation = False
#    emissivity = 0.0

##########################################################################
    
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
        if (constant_cfl == False):
            print(f"\tt_final = {t_final}")
        else:
            print(f"\tconstant_cfl = {constant_cfl}")
            print(f"\tcurrent_cfl = {current_cfl}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")


##########################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    mechanism_file = "inert"

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_cantera = 300.0

    x_left = np.zeros(nspecies)
    x_left[cantera_soln.species_index("Ar")] = 1.0
    x_left[cantera_soln.species_index("He")] = 0.0

    pres_cantera = cantera.one_atm

    cantera_soln.TPX = temp_cantera, pres_cantera, x_left
    y_left = cantera_soln.Y

    x_right = np.zeros(nspecies)
    x_right[cantera_soln.species_index("Ar")] = 0.0
    x_right[cantera_soln.species_index("He")] = 1.0

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

    # }}}    
    
    # {{{ Initialize transport model
    if transport == "Mixture":
        physical_transport = MixtureAveragedTransport(
            pyrometheus_mechanism, lewis=np.ones((nspecies,)), factor=speedup_factor)
    else:
        if transport == "PowerLaw":
            physical_transport = PowerLawTransport(lewis=np.ones((nspecies,)),
               beta=4.093e-7*speedup_factor)
        else:
            print('No transport class defined..')
            print('Use one of "Mixture" or "PowerLaw"')
            sys.exit()

    gas_model = GasModel(eos=eos, transport=physical_transport)

    print(f"Pyrometheus mechanism species names {species_names}")

#############################################################################

    from mirgecom.limiter import bound_preserving_limiter

    from grudge.discretization import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_MODAL
    from meshmode.transform_metadata import FirstAxisIsElementsTag

    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

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
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

    def _get_fluid_state(cv, temp_seed, limiter_dd):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=limiter_dd)

    get_fluid_state = actx.compile(_get_fluid_state)

##################################

    velocity = np.zeros(shape=(dim,))

    # use the burned conditions with a lower temperature
    flow_init = case1(dim=dim, sigma=0.5, nspecies=nspecies,
                      velocity=velocity,
                      species_mass_right=y_right,
                      species_mass_left=y_left)

##############################################################################

    restart_step = None
    if restart_file:
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        volume_to_local_mesh = restart_data["volume_to_local_mesh"]
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_ranks"] == num_ranks
    else:  # import the grid
        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                "multivolume-v2.msh", force_ambient_dim=2,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "Fluid": ["Upper"],
                "Solid": ["Lower"]}
            return mesh, tag_to_elements, volume_to_tags

        def partition_generator_func(mesh, tag_to_elements, num_ranks):
            from meshmode.distributed import get_partition_by_pymetis
            return get_partition_by_pymetis(mesh, num_ranks)

        from mirgecom.simutil import distribute_mesh
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

    #~~~~~~~~~~
    from grudge.dt_utils import characteristic_lengthscales
    char_length_fluid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)
    char_length_solid = characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid)

##############################################################################

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
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                restart_dcoll.discr_from_dd(dd_vol_fluid)
            )
            wall_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_solid),
                restart_dcoll.discr_from_dd(dd_vol_solid)
            )
            current_cv = fluid_connection(restart_data["cv"])
            tseed = fluid_connection(restart_data["temperature_seed"])
            current_wv = wall_connection(restart_data["wv"])
            wv_tseed = fluid_connection(restart_data["wall_temperature_seed"])
        else:
            current_cv = restart_data["cv"]
            tseed = restart_data["temperature_seed"]
            current_wv = restart_data["wv"]
            wv_tseed = restart_data["wall_temperature_seed"]

    current_cv = force_evaluation(actx, current_cv)
    tseed = force_evaluation(actx, tseed)
    fluid_state = get_fluid_state(current_cv, tseed, dd_vol_fluid)

    current_wv = force_evaluation(actx, current_wv)
    wv_tseed = force_evaluation(actx, wv_tseed)
    solid_state = get_fluid_state(current_wv, wv_tseed, dd_vol_solid)

##############################################################################

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
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

##############################################################################

    isothermal_wall_temp = 300.0
    fluid_boundaries = {
        dd_vol_fluid.trace("Upper Sides").domain_tag:  # pylint: disable=no-member
        IsothermalWallBoundary(wall_temperature=isothermal_wall_temp)}

    solid_boundaries = {
        dd_vol_solid.trace("Lower Sides").domain_tag:  # pylint: disable=no-member
        NeumannDiffusionBoundary(0.0)}

##############################################################################

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

##############################################################################

    def my_write_viz(step, t, dt, fluid_state, wall_state):

        fluid_viz_fields = [
            ("fluid_CV_rho", fluid_state.cv.mass),
            ("fluid_CV_rhoU", fluid_state.cv.momentum),
            ("fluid_CV_rhoE", fluid_state.cv.energy),
            ("fluid_DV_P", fluid_state.pressure),
            ("fluid_DV_T", fluid_state.temperature),
            ("fluid_DV_U", fluid_state.velocity[0]),
            ("fluid_DV_V", fluid_state.velocity[1]),
        ]

        # species mass fractions
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        solid_viz_fields = [
            ("solid_CV_rho", wall_state.cv.mass),
            ("solid_CV_rhoU", wall_state.cv.momentum),
            ("solid_CV_rhoE", wall_state.cv.energy),
            ("solid_DV_P", wall_state.pressure),
            ("solid_DV_T", wall_state.temperature),
            ("solid_DV_U", wall_state.velocity[0]),
            ("solid_DV_V", wall_state.velocity[1]),
        ]

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
            print('Writing restart file...')

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
                "num_parts": nparts
            }
            
            write_restart_file(actx, restart_data, restart_fname, comm)

#########################################################################

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

##############################################################################

    def _my_get_timestep_fluid(fluid_state, t, dt, dd=None):

        if not constant_cfl:
            return dt

        return get_sim_timestep(dcoll, fluid_state, t, dt,
            current_cfl, gas_model, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd)

    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)

##############################################################################

    import os
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        cv, tseed, wv, wv_tseed = state

        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)
        wv = force_evaluation(actx, wv)
        wv_tseed = force_evaluation(actx, wv_tseed)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed, dd_vol_fluid)
        cv = fluid_state.cv

        # construct species-limited fluid state
        solid_state = get_fluid_state(wv, wv_tseed, dd_vol_solid)
        wv = solid_state.cv

        if constant_cfl:
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                t_final, constant_cfl, local_dt, dd_vol_fluid)

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

            state = make_obj_array([fluid_state.cv, fluid_state.temperature,
                                    solid_state.cv, solid_state.temperature])

            file_exists = os.path.exists('write_solution')
            if file_exists:
              os.system('rm write_solution')
              do_viz = True
        
            file_exists = os.path.exists('write_restart')
            if file_exists:
              os.system('rm write_restart')
              do_restart = True

            if do_health:
                ## FIXME warning in lazy compilation
                from warnings import warn
                warn(f"Lazy does not like the health_check", stacklevel=2)
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_viz:

                print('Writing solution file')

                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    wall_state=solid_state)

            if do_restart:
                my_write_restart(step, t, state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                wv=wv, wdv=wdv, smoothness=smoothness)
            raise

        return state, dt


    def my_rhs(t, state):
        cv, tseed, wv, wv_tseed = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

        wall_state = make_fluid_state(cv=wv, gas_model=gas_model,
            temperature_seed=wv_tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_solid)

        fluid_rhs, solid_rhs = coupled_ns_operator(
                dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
                fluid_boundaries, solid_boundaries, fluid_state, wall_state,
                time=t, limiter_func=_limit_fluid_cv,
                quadrature_tag=quadrature_tag)

        return make_obj_array([fluid_rhs, fluid_zeros, solid_rhs, solid_zeros])

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

        min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
        if logmgr:
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([fluid_state.cv,
                                    fluid_state.temperature,
                                    solid_state.cv,
                                    solid_state.temperature])

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

    # 
    final_cv, tseed, final_wv, wv_tseed = stepper_state
    final_state = get_fluid_state(final_cv, tseed)

    final_wdv = 0

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_write_restart(step=final_step, t=final_t, state=stepper_state)

    my_write_viz(step=final_step, t=final_t, dt=current_dt,
                 cv=final_state.cv, dv=current_state.dv,
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
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

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

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile, casename=casename,
         lazy=args.lazy, restart_filename=restart_file)
