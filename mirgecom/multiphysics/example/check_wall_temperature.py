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
    write_visfile,
    generate_and_distribute_mesh,
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
#from mirgecom.multiphysics.multiphysics_coupled_fluid_wall import (
#    # coupled_grad_t_operator,
#    coupled_ns_operator
#)
from mirgecom.diffusion import (
    diffusion_operator, DirichletDiffusionBoundary, NeumannDiffusionBoundary
)

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array

#########################################################################

class InitSolidDomain:

    def __init__(
            self, *, dim=2, sigma=1.0, nspecies, velocity,
                     species_mass_right, species_mass_left):
                   
        self._dim = dim
        self._nspecies = nspecies 
        self._sigma = sigma
        self._vel = velocity
        self._yu = species_mass_right
        self._yb = species_mass_left
        self._disc = 1.0

    def __call__(self, x_vec, gas_model, wall_vars, *, time=0.0):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        eos = gas_model.eos
        wall = gas_model.wall

        x = x_vec[0]
        actx = x.array_context
        
        velocity = make_obj_array([0.0])
        
        aux = 0.5*(1.0 - actx.np.tanh( 1.0/(self._sigma)*(x - self._disc ) ))
        y1 = self._yu*aux
        y2 = self._yb*(1.0-aux)
        
        y = y1+y2

        aux = 0.5*(1.0 - actx.np.tanh( 1.0/(self._sigma)*(x - (self._disc) ) ))
        temperature = 300.0 + 300.0*aux
        
        pressure = 101325.0 + x*0.0

        solid_energy = wall_vars.mass*wall.enthalpy(temperature, 1.0)

        epsilon = 0.9
        mass = epsilon*eos.get_density(pressure, temperature, species_mass_fractions=y)
        specmass = mass * y
        momentum = velocity*mass        
        internal_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass*(internal_energy + kinetic_energy) + solid_energy

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
    nviz = 500
    nrestart = 25000
    nhealth = 1
    nstatus = 100

    # default timestepping control
#    integrator = "compiled_lsrk45"
    integrator = "ssprk43"
    current_dt = 0.2 #order == 2
    t_final = 50000.0

    local_dt = False
    constant_cfl = False
    current_cfl = 0.4
    
    # discretization and model control
    order = 3
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
    
    dim = 1

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

    rst_path = "restart_data/"
    rst_pattern = rst_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    if restart_file:  # read the grid from restart data
        rst_filename = f"{restart_file}"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts

    else:  # generate the grid from scratch
        from functools import partial
        nel_1d = 201

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(-2.0,)*dim, b=(+2.0,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={"prescribed": ["+x"], "neumann": ["-x"]})

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    quadrature_tag = DISCR_TAG_BASE

    nodes = actx.thaw(dcoll.nodes())

    from grudge.dof_desc import DD_VOLUME_ALL
    dd_vol = DD_VOLUME_ALL

    if rank == 0:
        logger.info("Done making discretization")

    #~~~~~~~~~~
    wall_sample_mask = nodes[0]*0.0 + 1.0

    zeros = nodes[0]*0.0

##############################################################################

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
    x_left[cantera_soln.species_index("O2")] = 0.2
    x_left[cantera_soln.species_index("N2")] = 0.8

    pres_cantera = cantera.one_atm

    cantera_soln.TPX = temp_cantera, pres_cantera, x_left
    y_left = cantera_soln.Y

    x_right = np.zeros(nspecies)
    x_right[cantera_soln.species_index("O2")] = 0.2
    x_right[cantera_soln.species_index("N2")] = 0.8

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

    # }}}

    # {{{ Initialize wall model

    import mirgecom.multiphysics.oxidation as wall
    import mirgecom.materials.carbon_fiber as my_material

    my_solid = my_material.SolidProperties()
    oxidation = my_material.Oxidation()

    def _get_wall_enthalpy(temperature, tau):
        wall_sample_h = my_solid.solid_enthalpy(temperature)
        return wall_sample_h * wall_sample_mask

    def _get_wall_heat_capacity(temperature, tau):
        wall_sample_cp = my_solid.solid_heat_capacity(temperature)
        return wall_sample_cp * wall_sample_mask

    def _get_wall_thermal_conductivity(temperature, tau):
        scaled_sample_kappa = \
            my_solid.solid_thermal_conductivity(temperature, tau)
        return scaled_sample_kappa * wall_sample_mask

    wall_degradation_model = wall.OxidationWallModel(solid_data=my_solid)

    from mirgecom.multiphysics.wall_model import WallEOS
    wall_model = WallEOS(
        wall_degradation_model=wall_degradation_model,
        wall_sample_mask=wall_sample_mask,
        enthalpy_func=_get_wall_enthalpy,
        heat_capacity_func=_get_wall_heat_capacity,
        thermal_conductivity_func=_get_wall_thermal_conductivity)

    # }}}

    gas_model = GasModel(eos=eos, transport=physical_transport,
                         wall=wall_model)

    print(f"Pyrometheus mechanism species names {species_names}")

##############################################################################

    from mirgecom.multiphysics.wall_model import WallConservedVars
    wall_vars = WallConservedVars(mass=zeros + 160.0)

    velocity = np.zeros(shape=(dim,))

    wall_init = InitSolidDomain(dim=dim, sigma=0.1, nspecies=nspecies,
                                velocity=velocity,
                                species_mass_right=y_right,
                                species_mass_left=y_left)

##############################################################################

    current_step = 0
    current_t = 0.0
    if rank == 0:
        logging.info("Initializing soln.")
    current_wv = wall_init(nodes, gas_model, wall_vars)

    wv_tseed = force_evaluation(actx, 300.0 + zeros)

    first_step = force_evaluation(actx, current_step)

    current_wv = force_evaluation(actx, current_wv)
    wv_tseed = force_evaluation(actx, wv_tseed)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    solid_state = make_fluid_state(cv=current_wv, gas_model=gas_model,
            wall_vars=wall_vars, temperature_seed=wv_tseed)

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

    solid_boundaries = {}

##############################################################################

    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol)

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

    def my_write_viz(step, t, dt, wall_state):

        solid_viz_fields = [
            ("x", nodes[0]),
            ("CV_rho", wall_state.cv.mass),
            ("CV_rhoU", wall_state.cv.momentum),
            ("CV_rhoE", wall_state.cv.energy),
            ("DV_P", wall_state.pressure),
            ("DV_T", wall_state.temperature),
        ]

        # species mass fractions
        solid_viz_fields.extend(
            ("Y_"+species_names[i], wall_state.cv.species_mass_fractions[i])
                for i in range(nspecies))

        write_visfile(dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True, comm=comm)

##############################################################################

    my_write_viz(step=0, t=0.0, dt=0.0, wall_state=solid_state)

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
