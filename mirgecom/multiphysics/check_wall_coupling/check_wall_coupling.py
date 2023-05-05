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
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    coupled_grad_t_operator,
    coupled_ns_heat_operator
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

        x = x_vec[0]
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
    constant_cfl = False
    current_cfl = 0.4
    
    # discretization and model control
    order = 4
    use_overintegration = False

#    transport = "Mixture"
    transport = "PowerLaw"

    # wall stuff
    temp_wall = 300

    wall_penalty_amount = 1.0
    wall_time_scale = 1 #speedup_factor

    use_radiation = False
    emissivity = 0.0

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
            print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")


##########################################################################

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    import os
    current_path = os.path.abspath(os.getcwd()) + "/"
    mechanism_file = current_path + "uiuc_sharp"

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = T_products

    air = "O2:0.21,N2:0.79"
    fuel = "C2H4:1"
    cantera_soln.set_equivalence_ratio(phi=equiv_ratio,
                                       fuel=fuel, oxidizer=air)
    x_unburned = cantera_soln.X
    pres_unburned = cantera.one_atm

    rho_int = cantera_soln.density

    r_int = 2.38*25.4/2000 #radius, actually
    r_ext = 2.89*25.4/2000 #radius, actually
    
#    mass_unb = flow_rate/np.sum(x_unburned[1:])
#    A_int = np.pi*r_int**2
#    lmin_to_m3s = 1.66667e-5
#    u_int = mass_unb*lmin_to_m3s/A_int
#    rhoU_int = rho_int*u_int
#    print("V_dot=",mass_unb,"(L/min)")

    mass_total = flow_rate*1.0
    A_int = np.pi*r_int**2
    lmin_to_m3s = 1.66667e-5
    u_int = mass_total*lmin_to_m3s/A_int
    rhoU_int = rho_int*u_int
    print("V_dot=",mass_total,"(L/min)")

    print(f"{rho_int= }","(kg/m^3)")
    print(f"{u_int= }","(m/s)")
    print(f"{A_int= }","(m^2)")
    print(f"{rhoU_int= }")

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x_unburned}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x_unburned

    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY
    mmw_unburned = cantera_soln.mean_molecular_weight

    cantera_soln.TPX = temp_ignition, pres_unburned, x_unburned
    cantera_soln.equilibrate("TP")
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P
    mmw_burned = cantera_soln.mean_molecular_weight

    # Pull temperature, density, mass fractions, and pressure from Cantera
    x = np.zeros(nspecies)
    x[cantera_soln.species_index("O2")] = 0.21
    x[cantera_soln.species_index("N2")] = 0.79
    cantera_soln.TPX = temp_unburned, pres_unburned, x

    y_atmosphere = np.zeros(nspecies)
    dummy, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    cantera_soln.equilibrate("TP")
    temp_atmosphere, rho_atmosphere, y_atmosphere = cantera_soln.TDY
    pres_atmosphere = cantera_soln.P
    mmw_atmosphere = cantera_soln.mean_molecular_weight
    
    # Pull temperature, density, mass fractions, and pressure from Cantera
    y_shroud = y_atmosphere*0.0
    y_shroud[cantera_soln.species_index("N2")] = 0.98
    y_shroud[cantera_soln.species_index("CO2")] = 0.01
    y_shroud[cantera_soln.species_index("CO")] = 0.01

    cantera_soln.TPY = 300.0, 101325.0, y_shroud
    temp_shroud, rho_shroud = cantera_soln.TD
    mmw_shroud = cantera_soln.mean_molecular_weight

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
                                cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 1000.0
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

        cv = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

        # make a new CV with the limited variables
        return cv

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

    get_fluid_state = actx.compile(_get_fluid_state)

##################################

    velocity = np.zeros(shape=(dim,))

    # use the burned conditions with a lower temperature
    flow_init = case1(dim=dim, sigma=0.1, nspecies=nspecies,
                      velocity=velocity,
                      species_mass_right=y_unburned,
                      species_mass_left=y_burned)

##############################################################################

    restart_step = None
    if restart_file is None:
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
                "multivolume.msh", force_ambient_dim=2,
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

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("solid"), DISCR_TAG_BASE)

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
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = flow_init(fluid_nodes, eos, flow_rate=flow_rate,
                               solve_the_flame=solve_the_flame)

        tseed = force_evaluation(actx, 1000.0 + fluid_zeros)

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
    current_state = get_fluid_state(current_cv, tseed)

    current_wv = force_evaluation(actx, current_wv)
    wv_tseed = force_evaluation(actx, wv_tseed)
    current_wdv = create_wall_dependent_vars_compiled(current_wv, wv_tseed)

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

    inflow_nodes = force_evaluation(actx,
                                    dcoll.nodes(dd_vol_fluid.trace('inlet')))
    inflow_temperature = force_evaluation(actx, inflow_nodes[0]*0.0 + 300.0)
    def bnd_temperature_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return inflow_temperature

    def inlet_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_cv_cond = ref_state(x_vec=inflow_nodes, eos=eos,
            flow_rate=flow_rate, solve_the_flame=solve_the_flame,
            state_minus=state_minus)
        return make_fluid_state(cv=inflow_cv_cond, gas_model=gas_model,
            temperature_seed=300.0)

    from mirgecom.inviscid import inviscid_flux
    from mirgecom.flux import num_flux_central
    from mirgecom.viscous import viscous_flux

    class MyPrescribedBoundary(PrescribedFluidBoundary):
        r"""My prescribed boundary function. """

        def __init__(self, bnd_state_func, temperature_func):
            """Initialize the boundary condition object."""
            self.bnd_state_func = bnd_state_func
            PrescribedFluidBoundary.__init__(self,
            boundary_state_func=bnd_state_func,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_temperature_func=temperature_func,
            boundary_gradient_cv_func=self.grad_cv_bc)

        def prescribed_state_for_advection(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
            state_plus = self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                             state_minus, **kwargs)

            mom_x = - state_minus.cv.momentum[0]
            mom_y = - state_minus.cv.momentum[1] + 2.0*state_plus.cv.momentum[1]
            mom_plus = make_obj_array([mom_x, mom_y])

            kin_energy_ref = 0.5/state_plus.cv.mass*np.dot(state_plus.cv.momentum, state_plus.cv.momentum)
            kin_energy_mod = 0.5/state_plus.cv.mass*np.dot(mom_plus, mom_plus)
            energy_plus = state_plus.cv.energy - kin_energy_ref + kin_energy_mod

            cv = make_conserved(dim=2, mass=state_plus.cv.mass,
                energy=energy_plus, momentum=mom_plus,
                species_mass=state_plus.cv.species_mass)

            return make_fluid_state(cv=cv, gas_model=gas_model,
                                    temperature_seed=300.0)

        def prescribed_state_for_diffusion(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
            return self.bnd_state_func(dcoll, dd_bdry, gas_model,
                                       state_minus, **kwargs)

        def inviscid_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                numerical_flux_func, **kwargs):

            state_plus = self.prescribed_state_for_advection(dcoll=dcoll,
                dd_bdry=dd_bdry, gas_model=gas_model, 
                state_minus=state_minus,**kwargs)

            state_pair = TracePair(dd_bdry, interior=state_minus,
                                   exterior=state_plus)

            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            actx = state_pair.int.array_context
            lam = actx.np.maximum(state_pair.int.wavespeed,
                                  state_pair.ext.wavespeed)
            from mirgecom.flux import num_flux_lfr
            return num_flux_lfr(
                f_minus_normal=inviscid_flux(state_pair.int)@normal,
                f_plus_normal=inviscid_flux(state_pair.ext)@normal,
                q_minus=state_pair.int.cv,
                q_plus=state_pair.ext.cv, lam=lam)

        def grad_cv_bc(self, state_plus, state_minus, grad_cv_minus, normal,
                       **kwargs):
            """Return grad(CV) for boundary calculation of viscous flux."""
            return grad_cv_minus

        def viscous_wall_flux(self, dcoll, dd_bdry, gas_model, state_minus,
            grad_cv_minus, grad_t_minus, numerical_flux_func, **kwargs):
            """Return the boundary flux for viscous flux."""
            actx = state_minus.array_context
            normal = actx.thaw(dcoll.normal(dd_bdry))

            state_plus = self.prescribed_state_for_diffusion(dcoll=dcoll,
                dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, **kwargs)

            grad_cv_plus = self.grad_cv_bc(state_plus=state_plus,
                state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                normal=normal, **kwargs)

            grad_t_plus = self._bnd_grad_temperature_func(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                grad_t_minus=grad_t_minus)

            # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
            # *not* the numerical viscous flux as advised by [Bassi_1997]_.
            f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                                 grad_t=grad_t_plus)
            return f_ext@normal


    linear_bnd = LinearizedOutflowBoundary(
        free_stream_density=rho_atmosphere, free_stream_pressure=101325.0,
        free_stream_velocity=np.zeros(shape=(dim,)),
        free_stream_species_mass_fractions=y_atmosphere)

    fluid_boundaries = {
        dd_vol_fluid.trace("inlet").domain_tag:
            MyPrescribedBoundary(bnd_state_func=inlet_bnd_state_func, 
                                 temperature_func=bnd_temperature_func),
        dd_vol_fluid.trace("symmetry").domain_tag: AdiabaticSlipBoundary(),
        dd_vol_fluid.trace("wall").domain_tag: AdiabaticNoslipWallBoundary(),
        dd_vol_fluid.trace("linear").domain_tag: linear_bnd,
        dd_vol_fluid.trace("outlet").domain_tag:
            PressureOutflowBoundary(boundary_pressure=101325.0),
    }

    wall_symmetry = NeumannDiffusionBoundary(0.0)
    solid_boundaries = {
        dd_vol_solid.trace("wall_sym").domain_tag: wall_symmetry
    }

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

    def my_write_viz(step, t, dt, fluid_state, wv, wdv,
        smoothness=None, fluid_rhs=None, solid_rhs=None,
        grad_cv_fluid=None, grad_t_fluid=None, grad_t_solid=None,
        fluid_sources=None, solid_sources=None,):

#        heat_rls = pyrometheus_mechanism.heat_release(fluid_state)

        fluid_viz_fields = [
            ("CV_rho", fluid_state.cv.mass),
            ("CV_rhoU", fluid_state.cv.momentum),
            ("CV_rhoE", fluid_state.cv.energy),
            ("DV_P", fluid_state.pressure),
            ("DV_T", fluid_state.temperature),
            ("DV_U", fluid_state.velocity[0]),
            ("DV_V", fluid_state.velocity[1]),
            ("dt", dt[0] if local_dt else None),
            ("sponge", sponge_sigma),
            ("smoothness", 1.0 - theta_factor*smoothness),
            ("RR", chem_rate*reaction_rates_damping),
#            ("heat_rls", heat_rls),
        ]

        if grad_cv_fluid is not None:
            fluid_viz_fields.extend((
                ("fluid_grad_cv_rho", grad_cv_fluid.mass),
                ("fluid_grad_cv_rhoU", grad_cv_fluid.momentum[0]),
                ("fluid_grad_cv_rhoV", grad_cv_fluid.momentum[1]),
                ("fluid_grad_cv_rhoE", grad_cv_fluid.energy),
            ))

        if grad_t_fluid is not None:
            fluid_viz_fields.append(("fluid_grad_t", grad_t_fluid))

        if fluid_rhs is not None:
            fluid_viz_fields.append(("fluid_rhs", fluid_rhs))

        if fluid_sources is not None:
            fluid_viz_fields.append(("fluid_sources", fluid_sources))

        # species mass fractions
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        solid_viz_fields = [
            ("wv", wv),
            ("wall_kappa", wdv.thermal_conductivity),
            ("wall_alpha", wall_model.thermal_diffusivity(
                              wv.mass, wdv.temperature,
                              thermal_conductivity=wdv.thermal_conductivity)),
            ("wall_ox_diff", wdv.oxygen_diffusivity),
            ("wall_temperature", wdv.temperature),
            ("wall_grad_t", grad_t_solid),
            ("dt", dt[2] if local_dt else None),
        ]

        if grad_t_solid is not None:
            solid_viz_fields.append(("solid_grad_t", grad_t_solid))

        if solid_rhs is not None:
            solid_viz_fields.append(("solid_rhs", solid_rhs))

        if solid_sources is not None:
            solid_viz_fields.append(("solid_sources", solid_sources))

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

    from mirgecom.boundary import DummyBoundary
    from mirgecom.diffusion import DiffusionBoundary
    class DummyDiffusionBoundary(DiffusionBoundary):
        def get_grad_flux(self, dcoll, dd_bdry, kappa_minus, u_minus):
            return None
        def get_diffusion_flux(self, dcoll, dd_bdry, kappa_minus, u_minus,
            grad_u_minus, lengthscales_minus, *, penalty_amount=None):
            return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from arraycontext import outer
    from grudge.trace_pair import (
        interior_trace_pairs, tracepair_with_discr_tag)
    from meshmode.discretization.connection import FACE_RESTR_ALL

    class _MyGradTag:
        pass

    fluid_field = fluid_nodes[0]*0.0
    wall_field = solid_nodes[0]*0.0
    pairwise_field = {
        (dd_vol_fluid, dd_vol_solid): (fluid_field, wall_field)}
    pairwise_field_tpairs = inter_volume_trace_pairs(
        dcoll, pairwise_field, comm_tag=_MyGradTag)
    field_tpairs_F = pairwise_field_tpairs[dd_vol_solid, dd_vol_fluid]
    field_tpairs_W = pairwise_field_tpairs[dd_vol_fluid, dd_vol_solid]

    axisym_fluid_boundaries = {}
    axisym_fluid_boundaries.update(fluid_boundaries)
    axisym_fluid_boundaries.update({
            tpair.dd.domain_tag: DummyBoundary()
            for tpair in field_tpairs_F})

    axisym_wall_boundaries = {}
    axisym_wall_boundaries.update(solid_boundaries)
    axisym_wall_boundaries.update({
            tpair.dd.domain_tag: DummyDiffusionBoundary()
            for tpair in field_tpairs_W})

    def my_derivative_function(actx, dcoll, field, field_bounds, dd_vol,
                               bnd_cond):    

        dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

        interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll,
                                      quadrature_tag)

        def interior_flux(field_tpair):
            dd_trace_quad = field_tpair.dd.with_discr_tag(quadrature_tag)
            normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
            bnd_tpair_quad = interp_to_surf_quad(field_tpair)
            flux_int = outer(num_flux_central(bnd_tpair_quad.int,
                                              bnd_tpair_quad.ext),
                             normal_quad)

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad,
                              flux_int)

        def boundary_flux(bdtag, bdry):
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = actx.thaw(dcoll.normal(dd_bdry_quad)) 
            int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field)

            if bnd_cond == 'symmetry' and bdtag  == '-0':
                ext_soln_quad = 0.0*int_soln_quad
            else:
                ext_soln_quad = 1.0*int_soln_quad

            bnd_tpair = TracePair(dd_bdry_quad,
                interior=int_soln_quad, exterior=ext_soln_quad)
            flux_bnd = outer(num_flux_central(bnd_tpair.int, bnd_tpair.ext),
                             normal_quad)
        
            return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad, flux_bnd)

        return -op.inverse_mass(
            dcoll, dd_vol,
            op.weak_local_grad(dcoll, dd_vol, field)
            - op.face_mass(dcoll, dd_allfaces_quad,
                (sum(interior_flux(u_tpair) for u_tpair in
                    interior_trace_pairs(dcoll, field, volume_dd=dd_vol,
                                         comm_tag=_MyGradTag))
                + sum(boundary_flux(bdtag, bdry) for bdtag, bdry in
                    field_bounds.items())
                )
            )
        )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    off_axis_x = 1e-7
    fluid_nodes_are_off_axis = actx.np.greater(fluid_nodes[0], off_axis_x)
    solid_nodes_are_off_axis = actx.np.greater(solid_nodes[0], off_axis_x)
   
    def axisym_source_fluid(actx, dcoll, state, grad_cv, grad_t):      
        cv = state.cv
        dv = state.dv
        
        mu = state.tv.viscosity
        beta = state.tv.volume_viscosity
        kappa = state.tv.thermal_conductivity
        d_ij = state.tv.species_diffusivity
        
        grad_v = velocity_gradient(cv,grad_cv)
        grad_y = species_mass_fraction_gradient(cv, grad_cv)

        u = state.velocity[0]
        v = state.velocity[1]

        dudr = grad_v[0][0]
        dudy = grad_v[0][1]
        dvdr = grad_v[1][0]
        dvdy = grad_v[1][1]
        
        drhoudr = (grad_cv.momentum[0])[0]

        d2udr2   = my_derivative_function(actx, dcoll, dudr, fluid_boundaries,
                                          dd_vol_fluid, 'replicate')[0] #XXX
        d2vdr2   = my_derivative_function(actx, dcoll, dvdr, fluid_boundaries,
                                          dd_vol_fluid, 'replicate')[0] #XXX
        d2udrdy  = my_derivative_function(actx, dcoll, dudy, fluid_boundaries,
                                          dd_vol_fluid, 'replicate')[0] #XXX
                
        dmudr    = my_derivative_function(actx, dcoll,   mu, fluid_boundaries,
                                          dd_vol_fluid, 'replicate')[0]
        dbetadr  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, 'replicate')[0]
        dbetady  = my_derivative_function(actx, dcoll, beta, fluid_boundaries,
                                          dd_vol_fluid, 'replicate')[1]
        
        qr = - kappa*grad_t[0]
        dqrdr = 0.0 #- (dkappadr*grad_t[0] + kappa*d2Tdr2) #XXX
        
        dyidr = grad_y[:,0]
        #dyi2dr2 = my_derivative_function(actx, dcoll, dyidr, 'replicate')[:,0]   #XXX
        
        tau_ry = 1.0*mu*(dudy + dvdr)
        tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
        tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
        tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
                              fluid_nodes_are_off_axis, u/fluid_nodes[0], dudr )

        dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

        #~~~~~~
        source_mass_dom = - cv.momentum[0]

        source_rhoU_dom = - cv.momentum[0]*u \
                          + tau_rr - tau_tt \
                          + u*dbetadr + beta*dudr \
                          + beta*actx.np.where(
                              fluid_nodes_are_off_axis, -u/fluid_nodes[0], -dudr )
                              
        source_rhoV_dom = - cv.momentum[0]*v \
                          + tau_ry \
                          + u*dbetady + beta*dudy
        
        source_rhoE_dom = -( (cv.energy+dv.pressure)*u + qr ) \
                          + u*tau_rr + v*tau_ry \
                          + u**2*dbetadr + beta*2.0*u*dudr \
                          + u*v*dbetady + u*beta*dvdy + v*beta*dudy

        source_spec_dom = - cv.species_mass*u + d_ij*dyidr

        #~~~~~~
        source_mass_sng = - drhoudr
        source_rhoU_sng = 0.0  # mu*d2udr2 + 0.5*beta*d2udr2  #XXX
        source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady
        source_rhoE_sng = -( (cv.energy+dv.pressure)*dudr + dqrdr ) \
                                + tau_rr*dudr + v*dtaurydr \
                                + 2.0*beta*dudr**2 \
                                + beta*dudr*dvdy \
                                + v*dudr*dbetady \
                                + v*beta*d2udrdy
        source_spec_sng = - cv.species_mass*dudr #+ d_ij*dyi2dr2
        
        #~~~~~~
        source_mass = actx.np.where( fluid_nodes_are_off_axis,
                          source_mass_dom/fluid_nodes[0], source_mass_sng )
        source_rhoU = actx.np.where( fluid_nodes_are_off_axis,
                          source_rhoU_dom/fluid_nodes[0], source_rhoU_sng )
        source_rhoV = actx.np.where( fluid_nodes_are_off_axis,
                          source_rhoV_dom/fluid_nodes[0], source_rhoV_sng )
        source_rhoE = actx.np.where( fluid_nodes_are_off_axis,
                          source_rhoE_dom/fluid_nodes[0], source_rhoE_sng )
        source_spec = make_obj_array([
                      actx.np.where( fluid_nodes_are_off_axis,
                          source_spec_dom[i]/fluid_nodes[0], source_spec_sng[i] )
                      for i in range(nspecies)])
        
        return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                       momentum=make_obj_array([source_rhoU, source_rhoV]),
                       species_mass=source_spec)

    compiled_axisym_source_fluid = actx.compile(axisym_source_fluid)

    # ~~~~~~~
    def axisym_source_solid(actx, dcoll, temperature, kappa, grad_t,
                            ox_concentration, ox_diff, grad_ox):              
        dkappadr = 0.0*solid_nodes[0]
        
        qr = - kappa*grad_t[0]
#        d2Tdr2  = my_derivative_function(actx, dcoll, grad_t[0], axisym_wall_boundaries, dd_vol_solid,  'symmetry')[0]
#        dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)
                
        source_mass = 0.0*solid_nodes[0]

        source_ox_dom = + ox_diff*grad_ox[0]
        source_ox_sng = 0.0
        source_ox = actx.np.where( solid_nodes_are_off_axis,
                          source_ox_dom/solid_nodes[0], source_ox_sng )

        source_rhoE_dom = - qr
        source_rhoE_sng = 0.0 #- dqrdr
        source_rhoE = actx.np.where( solid_nodes_are_off_axis,
                          source_rhoE_dom/solid_nodes[0], source_rhoE_sng )

        return WallVars(mass=source_mass, ox_mass=source_ox,
                          energy=source_rhoE)

    compiled_axisym_source_solid = actx.compile(axisym_source_solid)

    # ~~~~~~~
    def gravity_source_terms(cv):
        """Gravity."""
        delta_rho = cv.mass - rho_atmosphere
        return make_conserved(dim=2, mass=cv.mass*0.0,
            energy=delta_rho*cv.velocity[1]*-9.80665,
            momentum=make_obj_array([cv.mass*0.0, delta_rho*-9.80665]),
            species_mass=cv.species_mass*0.0)

    # ~~~~~~~
    def chemical_source_term(cv, temperature):
        if solve_the_flame:
            return chem_rate*reaction_rates_damping*(
                eos.get_species_source_terms(cv, temperature))
        else:
            return zeros

##############################################################################

    from grudge.dof_desc import DD_VOLUME_ALL
    def my_get_wall_timestep(wv, wdv):
        wall_diffusivity = wall_model.thermal_diffusivity(wv.mass,
            wdv.temperature, wdv.thermal_conductivity)
        return char_length_solid**2/(wall_time_scale * actx.np.maximum(
            wall_diffusivity, wdv.oxygen_diffusivity))

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


    def _my_get_timestep_fluid(fluid_state, t, dt):

        if not constant_cfl:
            return dt

        return get_sim_timestep(dcoll, fluid_state, t, dt,
            current_cfl, gas_model, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd_vol_fluid)

    my_get_timestep_wall = actx.compile(_my_get_timestep_wall)
    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)

##############################################################################

    def _get_rhs(t, state):
        cv, tseed, wv, wv_tseed = state

        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)
        wv = force_evaluation(actx, wv)
        wv_tseed = force_evaluation(actx, wv_tseed)

        # include both outflow and sponge in the damping region
        smoothness = force_evaluation(actx,
            smooth_region + sponge_sigma/sponge_amp)

        # apply outflow damping
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed)

        # make sure we get the limited state
        cv = fluid_state.cv

        # wall variables
        wdv = create_wall_dependent_vars_compiled(wv, wv_tseed)

        #~~~~~~~~~~~~~
        fluid_rhs, wall_energy_rhs, grad_cv_fluid, grad_t_fluid, grad_t_solid = \
            coupled_ns_heat_operator(
                dcoll, gas_model, dd_vol_fluid, dd_vol_solid, fluid_boundaries, solid_boundaries,
                fluid_state, wdv.thermal_conductivity, wdv.temperature, time=t, quadrature_tag=quadrature_tag,
                interface_radiation=use_radiation, wall_epsilon=emissivity, return_gradients=True)

        chem_rhs = chemical_source_term(fluid_state.cv, fluid_state.temperature)

        fluid_sources = (
            sponge_func(cv=cv, cv_ref=ref_cv, sigma=sponge_sigma)
            + gravity_source_terms(cv)
            + axisym_source_fluid(actx, dcoll, fluid_state, grad_cv_fluid, grad_t_fluid))

        #~~~~~~~~~~~~~
        wall_mass_rhs = -oxidation.get_source_terms(temperature=wdv.temperature,
            tau=wdv.tau, ox_mass=wv.ox_mass)

        idx_O2 = 1
        fluid_ox_mass = cv.species_mass[idx_O2]

        pairwise_ox = {(dd_vol_fluid, dd_vol_solid): (fluid_ox_mass, wv.ox_mass)}
        pairwise_ox_tpairs = inter_volume_trace_pairs(
            dcoll, pairwise_ox, comm_tag=_OxCommTag)
        ox_tpairs = pairwise_ox_tpairs[dd_vol_fluid, dd_vol_solid]

        wall_ox_boundaries = {
            dd_vol_solid.trace("wall_sym").domain_tag:
                NeumannDiffusionBoundary(0.0)}
        wall_ox_boundaries.update({
            tpair.dd.domain_tag: DirichletDiffusionBoundary(tpair.ext)
            for tpair in ox_tpairs})

        wall_ox_mass_rhs, grad_ox_solid = diffusion_operator(dcoll,
            wdv.oxygen_diffusivity, wall_ox_boundaries, wv.ox_mass,
            return_grad_u=True, penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag, dd=dd_vol_solid,
            comm_tag=_WallOxDiffCommTag)

        solid_rhs = wall_time_scale * WallVars(mass=wall_mass_rhs,
            energy=wall_energy_rhs, ox_mass=wall_ox_mass_rhs)

        solid_sources = axisym_source_solid(actx, dcoll,
            wdv.temperature, wdv.thermal_conductivity, grad_t_solid,
            wv.ox_mass, wdv.oxygen_diffusivity, grad_ox_solid)

        #~~~~~~~~~~~~~
        fluid_rhs = fluid_rhs + chem_rhs + fluid_sources
        solid_rhs = solid_rhs + solid_sources

        return make_obj_array([fluid_rhs, solid_rhs,
            fluid_sources, solid_sources,
            grad_cv_fluid, grad_t_fluid, grad_t_solid, grad_ox_solid])

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

        # include both outflow and sponge in the damping region
        smoothness = force_evaluation(actx,
            smooth_region + sponge_sigma/sponge_amp)

        # apply outflow damping
        cv = drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(cv, tseed)

        # make sure we get the limited state
        cv = fluid_state.cv

        # wall variables
        wdv = create_wall_dependent_vars_compiled(wv, wv_tseed)

        if local_dt:
            t = force_evaluation(actx, t)

            dt_fluid = force_evaluation(actx, actx.np.minimum(
                current_dt, my_get_timestep_fluid(fluid_state, t[0], dt[0])))

            dt_wall = force_evaluation(actx, actx.np.minimum(
                1.0e-8, my_get_timestep_wall(wv, wdv, t[2], dt[2])))

            dt = make_obj_array([dt_fluid, dt_fluid*0., dt_wall, dt_wall*0.])
        else:
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
                                    wv, wdv.temperature])

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

                fluid_rhs = None
                solid_rhs = None
                grad_cv_fluid = None
                grad_t_fluid = None
                grad_t_solid = None
                fluid_sources = None
                solid_sources = None

                import datetime
                print(datetime.datetime.now())

                fluid_rhs, solid_rhs, fluid_sources, solid_sources, \
                    grad_cv_fluid, grad_t_fluid, grad_t_solid, \
                    grad_ox_solid = _get_rhs(t, state)

                fluid_rhs = force_evaluation(actx, fluid_rhs)
                solid_rhs = force_evaluation(actx, solid_rhs)
                fluid_sources = force_evaluation(actx, fluid_sources)
                solid_sources = force_evaluation(actx, solid_sources)
                grad_cv_fluid = force_evaluation(actx, grad_cv_fluid)
                grad_t_fluid = force_evaluation(actx, grad_t_fluid)
                grad_t_solid = force_evaluation(actx, grad_t_solid)
                grad_ox_solid = force_evaluation(actx, grad_ox_solid)

                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    wv=wv, wdv=wdv, smoothness=smoothness,
                    fluid_rhs=fluid_rhs, solid_rhs=solid_rhs,
                    grad_cv_fluid=grad_cv_fluid,
                    grad_t_fluid=grad_t_fluid, grad_t_solid=grad_t_solid,
                    fluid_sources=fluid_sources, solid_sources=solid_sources)

                print(datetime.datetime.now())

            if do_restart:
                my_write_restart(step, t, state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                wv=wv, wdv=wdv, smoothness=smoothness)
            raise

        return make_obj_array([fluid_state.cv, fluid_state.temperature,
                               wv, wdv.temperature]), \
               dt


    class _WallOxDiffCommTag:
        pass


    class _OxCommTag:
        pass


    def my_rhs(t, state):
        cv, tseed, wv, wv_tseed = state

        # include both outflow and sponge in the damping region
        smoothness = smooth_region + sponge_sigma/sponge_amp

        # apply outflow damping
        cv = _drop_order_cv(cv, smoothness, theta_factor)

        # construct species-limited fluid state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

        # make sure we get the limited state
        cv = fluid_state.cv

        # wall variables
        wdv = wall_model.dependent_vars(wv, wv_tseed)

        #~~~~~~~~~~~~~
        fluid_rhs, wall_energy_rhs, grad_cv_fluid, grad_t_fluid, grad_t_solid = \
            coupled_ns_heat_operator(
                dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
                fluid_boundaries, solid_boundaries, fluid_state,
                wdv.thermal_conductivity, wdv.temperature, time=t,
                quadrature_tag=quadrature_tag, 
                interface_radiation=use_radiation, wall_epsilon=emissivity,
                return_gradients=True)

        chem_rhs = chemical_source_term(cv, fluid_state.temperature)

        fluid_sources = (
            sponge_func(cv=fluid_state.cv, cv_ref=ref_cv, sigma=sponge_sigma)
            + gravity_source_terms(fluid_state.cv)
            + axisym_source_fluid(actx, dcoll, fluid_state,
                                  grad_cv_fluid, grad_t_fluid))

        #~~~~~~~~~~~~~
        wall_mass_rhs = -oxidation.get_source_terms(temperature=wdv.temperature,
            tau=wdv.tau, ox_mass=wv.ox_mass)

        idx_O2 = 1
        fluid_ox_mass = cv.species_mass[idx_O2]

        pairwise_ox = {(dd_vol_fluid, dd_vol_solid): (fluid_ox_mass, wv.ox_mass)}
        pairwise_ox_tpairs = inter_volume_trace_pairs(
            dcoll, pairwise_ox, comm_tag=_OxCommTag)
        ox_tpairs = pairwise_ox_tpairs[dd_vol_fluid, dd_vol_solid]

        wall_ox_boundaries = {
            dd_vol_solid.trace("wall_sym").domain_tag:
                NeumannDiffusionBoundary(0.0)}
        wall_ox_boundaries.update({
            tpair.dd.domain_tag: DirichletDiffusionBoundary(tpair.ext)
            for tpair in ox_tpairs})

        wall_ox_mass_rhs, grad_ox_solid = diffusion_operator(dcoll,
            wdv.oxygen_diffusivity, wall_ox_boundaries, wv.ox_mass,
            return_grad_u=True, penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag, dd=dd_vol_solid,
            comm_tag=_WallOxDiffCommTag)

        solid_rhs = wall_time_scale * WallVars(mass=wall_mass_rhs,
            energy=wall_energy_rhs, ox_mass=wall_ox_mass_rhs)

        solid_sources = axisym_source_solid(actx, dcoll,
            wdv.temperature, wdv.thermal_conductivity, grad_t_solid,
            wv.ox_mass, wdv.oxygen_diffusivity, grad_ox_solid)

        #~~~~~~~~~~~~~
        fluid_rhs = fluid_rhs + chem_rhs + fluid_sources
        solid_rhs = solid_rhs + solid_sources

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

    stepper_state = make_obj_array([current_state.cv,
                                    current_state.temperature,
                                    current_wv,
                                    current_wdv.temperature])

    if local_dt == True:
        dt_fluid = force_evaluation(actx, actx.np.minimum(
            current_dt, my_get_timestep_fluid(current_state,
                            force_evaluation(actx, current_t + fluid_zeros),
                            force_evaluation(actx, current_dt + fluid_zeros))
            )
        )

        dt_wall = force_evaluation(actx, actx.np.minimum(
            1.0e-8, my_get_timestep_wall(current_wv, current_wdv,
                            force_evaluation(actx, current_t + solid_zeros),
                            force_evaluation(actx, 1.0e-8 + solid_zeros))
            )
        )

        dt = make_obj_array([dt_fluid, dt_fluid*0.0, dt_wall, dt_wall*0.0])

        t_fluid = force_evaluation(actx, current_t + fluid_zeros)
        t_wall = force_evaluation(actx, current_t + solid_zeros)
        t = make_obj_array([t_fluid, t_fluid, t_wall, t_wall])
    else:
        if constant_cfl:
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
                      max_steps=niter, local_dt=local_dt,
                      force_eval=force_eval, state=stepper_state)

    # 
    final_cv, tseed, final_wv, wv_tseed = stepper_state
    final_state = get_fluid_state(final_cv, tseed)

    final_wdv = wall_model.dependent_vars(final_wv, wv_tseed)

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
