"""Demonstrates coupling of multiple domains."""

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
from dataclasses import dataclass
from arraycontext import (
    dataclass_array_container, with_container_arithmetic,
    get_container_context_recursively
)
from meshmode.dof_array import DOFArray
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, VolumeDomainTag
)
from mirgecom.utils import force_evaluation
from mirgecom.simutil import (
    check_step, distribute_mesh, write_visfile,
    check_naninf_local, global_reduce
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.steppers import advance_state
from mirgecom.boundary import IsothermalWallBoundary, AdiabaticSlipBoundary
from mirgecom.fluid import make_conserved
from mirgecom.transport import SimpleTransport
import cantera
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    make_operator_fluid_states
)
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
)
from mirgecom.navierstokes import (
    grad_t_operator,
    grad_cv_operator,
    ns_operator
)
from mirgecom.multiphysics.multiphysics_coupled_fluid_wall import (
    add_interface_boundaries as add_multiphysics_interface_boundaries,
    add_interface_boundaries_no_grad as add_multiphysics_interface_boundaries_no_grad
)
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    add_interface_boundaries as add_thermal_interface_boundaries,
    add_interface_boundaries_no_grad as add_thermal_interface_boundaries_no_grad
)
from mirgecom.diffusion import (
    diffusion_operator,
    grad_operator as wall_grad_t_operator,
    NeumannDiffusionBoundary
)

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array

#########################################################################


class _FluidGradCVTag:
    pass


class _FluidGradTempTag:
    pass


class _SampleGradCVTag:
    pass


class _SampleGradTempTag:
    pass


class _HolderGradTempTag:
    pass


class _FluidOperatorTag:
    pass


class _SampleOperatorTag:
    pass


class _HolderOperatorTag:
    pass


class _FluidOpStatesTag:
    pass


class _WallOpStatesTag:
    pass


class HolderInitializer:

    def __init__(self, temperature):
        self._temp = temperature

    def __call__(self, x_vec, wall_model):
        mass = wall_model.density() + x_vec[0]*0.0
        energy = mass * wall_model.enthalpy(self._temp)
        return HolderWallVars(mass=mass, energy=energy)


class SampleInitializer:

    def __init__(self, pressure, temperature, species_atm):

        self._pres = pressure
        self._ya = species_atm
        self._temp = temperature

    def __call__(self, x_vec, gas_model, wall_density):

        eos = gas_model.eos
        zeros = x_vec[0]*0.0

        tau = zeros + 1.0

        velocity = make_obj_array([zeros, zeros])

        pressure = self._pres + zeros
        temperature = self._temp + zeros
        y = self._ya + zeros

        int_energy = eos.get_internal_energy(temperature, species_mass_fractions=y)
        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)

        epsilon = gas_model.wall.void_fraction(tau=tau)
        eps_rho_g = epsilon * mass
        eps_rhoU_g = eps_rho_g * velocity  # noqa N806
        eps_rhoY_g = eps_rho_g * y  # noqa N806

        eps_rho_s = wall_density + zeros
        enthalpy_s = gas_model.wall.enthalpy(temperature=temperature, tau=tau)

        energy = eps_rho_g * int_energy + eps_rho_s * enthalpy_s

        return make_conserved(dim=2, mass=eps_rho_g,
            momentum=eps_rhoU_g, energy=energy, species_mass=eps_rhoY_g)


class FluidInitializer:

    def __init__(self, species_left, species_right):
        self._yl = species_left
        self._yr = species_right

    def __call__(self, x_vec, gas_model):

        actx = x_vec[0].array_context
        eos = gas_model.eos

        hot_temp = 2000.0
        cold_temp = 300.0

        aux = 0.5*(1.0 - actx.np.tanh(1.0/(.01)*(x_vec[0] + 0.25)))
        y1 = self._yl*aux
        y2 = self._yr*(1.0-aux)
        y = y1+y2

        pressure = 101325.0 + x_vec[0]*0.0
        temperature = cold_temp + \
            (hot_temp - cold_temp)*.5*(1. - actx.np.tanh(1.0/.01*(x_vec[0]+.25)))

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        momentum = make_obj_array([0.0*x_vec[0], 0.0*x_vec[0]])
        specmass = mass * y
        energy = mass * eos.get_internal_energy(temperature,
                                                species_mass_fractions=y)

        return make_conserved(dim=2, mass=mass, energy=energy,
                momentum=momentum, species_mass=specmass)


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class HolderWallVars:

    mass: DOFArray
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)


@dataclass_array_container
@dataclass(frozen=True)
class HolderDependentVars:
    thermal_conductivity: DOFArray
    temperature: DOFArray


@dataclass_array_container
@dataclass(frozen=True)
class HolderState():
    cv: HolderWallVars
    dv: HolderDependentVars


class HolderWallModel:
    """Model for calculating wall quantities."""
    def __init__(self, density_func, enthalpy_func, heat_capacity_func,
                 thermal_conductivity_func):
        self._density_func = density_func
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def density(self):
        return self._density_func()

    def heat_capacity(self):
        return self._heat_capacity_func()

    def enthalpy(self, temperature):
        return self._enthalpy_func(temperature)

    def thermal_diffusivity(self, mass, temperature,
                            thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity()
        return thermal_conductivity/(mass * self.heat_capacity())

    def thermal_conductivity(self):
        return self._thermal_conductivity_func()

    def eval_temperature(self, wv):
        return wv.energy/(self.density()*self.heat_capacity())

    def dependent_vars(self, wv):
        temperature = self.eval_temperature(wv)
        kappa = self.thermal_conductivity() + wv.mass*0.0
        return HolderDependentVars(thermal_conductivity=kappa,
                                   temperature=temperature)


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    rst_path = "./"
    viz_path = "./"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 100
    nrestart = 25000
    nhealth = 1
    nstatus = 100
    ngarbage = 50

    # default timestepping control
    integrator = "ssprk43"
    current_dt = 2.5e-2
    t_final = 2.5e-1

    local_dt = False
    constant_cfl = False
    current_cfl = 0.4

    # discretization and model control
    order = 1
    use_overintegration = False

    # wall stuff
    temp_wall = 300.0
    wall_penalty_amount = 1.0

    use_radiation = True  # or False
    emissivity = 1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dim = 2

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
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    restart_step = None
    if restart_file is None:

        mesh_filename = "mult_coupled_vols-v2.msh"

        import os
        if rank == 0:
            os.system("rm -rf mult_coupled_vols.msh mult_coupled_vols-v2.msh")
            os.system("gmsh mult_coupled_vols.geo -2 mult_coupled_vols.msh")
            os.system("gmsh mult_coupled_vols.msh -save -format msh2 -o mult_coupled_vols-v2.msh")  # noqa E501
        else:
            os.system("sleep 2s")

        current_step = 0
        first_step = current_step + 0
        current_t = 0.0

        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "fluid": ["Fluid"],
                "sample": ["Sample"],
                "holder": ["Holder"]
                }
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        first_step = restart_step+0

        assert comm.Get_size() == restart_data["num_parts"]

    local_nelements = (
        + volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["sample"][0].nelements
        + volume_to_local_mesh_data["holder"][0].nelements)

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_sample = DOFDesc(VolumeDomainTag("sample"), DISCR_TAG_BASE)
    dd_vol_holder = DOFDesc(VolumeDomainTag("holder"), DISCR_TAG_BASE)

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    sample_nodes = actx.thaw(dcoll.nodes(dd_vol_sample))
    holder_nodes = actx.thaw(dcoll.nodes(dd_vol_holder))

    fluid_zeros = force_evaluation(actx, fluid_nodes[0]*0.0)
    sample_zeros = force_evaluation(actx, sample_nodes[0]*0.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    mechanism_file = "inert"

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    temp_cantera = 300.0
    pres_cantera = cantera.one_atm  # pylint: disable=no-member

    # Set Cantera internal gas temperature, pressure, and mole fractios
    x_left = np.zeros(nspecies)
    x_left[cantera_soln.species_index("Ar")] = 0.0
    x_left[cantera_soln.species_index("N2")] = 1.0

    cantera_soln.TPX = temp_cantera, pres_cantera, x_left
    y_left = cantera_soln.Y

    x_right = np.zeros(nspecies)
    x_right[cantera_soln.species_index("Ar")] = 1.0
    x_right[cantera_soln.species_index("N2")] = 0.0

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

    # {{{ Initialize transport model

    fluid_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.1,
        species_diffusivity=np.zeros(nspecies,) + 0.001)

    sample_transport = SimpleTransport(viscosity=0.0, thermal_conductivity=0.2,
        species_diffusivity=np.zeros(nspecies,) + 0.001)

    # }}}

    # ~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    from mirgecom.wall_model import WallEOS
    import mirgecom.materials.carbon_fiber as my_material

    fiber = my_material.SolidProperties()
    sample_degradation_model = WallEOS(wall_material=fiber)

    # }}}

    # ~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    # Averaging from https://www.azom.com/properties.aspx?ArticleID=52 for alumina
    wall_holder_rho = 10.0
    wall_holder_cp = 1000.0
    wall_holder_kappa = 2.00

    def _get_holder_density():
        return wall_holder_rho

    def _get_holder_enthalpy(temperature):
        return wall_holder_cp * temperature

    def _get_holder_heat_capacity():
        return wall_holder_cp

    def _get_holder_thermal_conductivity():
        return wall_holder_kappa

    holder_wall_model = HolderWallModel(
        density_func=_get_holder_density,
        enthalpy_func=_get_holder_enthalpy,
        heat_capacity_func=_get_holder_heat_capacity,
        thermal_conductivity_func=_get_holder_thermal_conductivity)

    # }}}

    gas_model_fluid = GasModel(eos=eos, transport=fluid_transport)

    gas_model_sample = GasModel(eos=eos, wall=sample_degradation_model,
                                transport=sample_transport)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter

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
        mass_lim = gas_model_fluid.eos.get_density(pressure=pressure,
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

    # ~~~~~~~~~~

    def _limit_sample_cv(cv, wv, pressure, temperature, epsilon, dd=None):

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

        # recompute gas density
        mass_lim = epsilon*gas_model_sample.eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute gas energy
        energy_gas = mass_lim*(gas_model_sample.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # compute solid energy
        tau = gas_model_sample.wall.decomposition_progress(wv)
        energy_solid = wv*gas_model_sample.wall.enthalpy(temperature, tau)

        # the total energy is a composition of both solid and gas
        energy = energy_gas + energy_solid

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

    def _get_sample_state(cv, wv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model_sample,
            wall_density=wv,
            temperature_seed=temp_seed,
            limiter_func=_limit_sample_cv, limiter_dd=dd_vol_sample)

    get_sample_state = actx.compile(_get_sample_state)

    # ~~~~~~~~~~

    def _get_holder_state(wv):
        dep_vars = holder_wall_model.dependent_vars(wv)
        return HolderState(cv=wv, dv=dep_vars)

    get_holder_state = actx.compile(_get_holder_state)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_init = FluidInitializer(species_left=y_left, species_right=y_right)

    sample_init = SampleInitializer(pressure=101325.0, temperature=300.0,
                                    species_atm=y_right)

    holder_init = HolderInitializer(temperature=300.0)

##############################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")

        fluid_tseed = temperature_seed + fluid_zeros
        sample_tseed = temp_wall + sample_zeros
        fluid_cv = fluid_init(fluid_nodes, gas_model_fluid)

        sample_density = 0.1*1600.0 + sample_zeros
        sample_cv = sample_init(sample_nodes, gas_model_sample, sample_density)

        holder_cv = holder_init(holder_nodes, holder_wall_model)

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
            sample_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_sample),
                restart_dcoll.discr_from_dd(dd_vol_sample)
            )
            holder_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_holder),
                restart_dcoll.discr_from_dd(dd_vol_holder)
            )
            fluid_cv = fluid_connection(restart_data["fluid_cv"])
            fluid_tseed = fluid_connection(restart_data["fluid_temperature_seed"])
            sample_cv = sample_connection(restart_data["sample_cv"])
            sample_tseed = sample_connection(restart_data["wall_temperature_seed"])
            sample_density = sample_connection(restart_data["sample_density"])
            holder_cv = holder_connection(restart_data["holder_cv"])
        else:
            fluid_cv = restart_data["fluid_cv"]
            fluid_tseed = restart_data["fluid_temperature_seed"]
            sample_cv = restart_data["sample_cv"]
            sample_tseed = restart_data["sample_temperature_seed"]
            sample_density = restart_data["sample_density"]
            holder_cv = restart_data["holder_cv"]

    fluid_cv = force_evaluation(actx, fluid_cv)
    fluid_tseed = force_evaluation(actx, fluid_tseed)
    fluid_state = get_fluid_state(fluid_cv, fluid_tseed)

    sample_cv = force_evaluation(actx, sample_cv)
    sample_tseed = force_evaluation(actx, sample_tseed)
    sample_density = force_evaluation(actx, sample_density)
    sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)

    holder_cv = force_evaluation(actx, holder_cv)
    holder_state = get_holder_state(holder_cv)

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

    fluid_boundaries = {
        dd_vol_fluid.trace("Fluid Hot").domain_tag:
            IsothermalWallBoundary(wall_temperature=2000.0),
        dd_vol_fluid.trace("Fluid Cold").domain_tag:
            IsothermalWallBoundary(wall_temperature=300.0),
        dd_vol_fluid.trace("Fluid Sides").domain_tag: AdiabaticSlipBoundary(),
    }

    # ~~~~~~~~~~
    sample_boundaries = {
        dd_vol_sample.trace("Sample Sides").domain_tag: AdiabaticSlipBoundary()
    }

    # ~~~~~~~~~~
    holder_boundaries = {
        dd_vol_holder.trace("Holder Sides").domain_tag: NeumannDiffusionBoundary(0.0)
    }

##############################################################################

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    sample_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_sample)
    holder_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_holder)

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

    def my_write_viz(step, t, dt, fluid_state, sample_state, holder_state):

        fluid_viz_fields = [
            ("rho_g", fluid_state.cv.mass),
            ("rhoU_g", fluid_state.cv.momentum),
            ("rhoE_g", fluid_state.cv.energy),
            ("pressure", fluid_state.pressure),
            ("temperature", fluid_state.temperature),
            ("Vx", fluid_state.velocity[0]),
            ("Vy", fluid_state.velocity[1]),
            ("dt", dt[0] if local_dt else None)]
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        sample_viz_fields = [
            ("rho_g", sample_state.cv.mass),
            ("rhoU_g", sample_state.cv.momentum),
            ("rhoE_b", sample_state.cv.energy),
            ("pressure", sample_state.pressure),
            ("temperature", sample_state.temperature),
            ("solid_mass", sample_state.dv.wall_density),
            ("Vx", sample_state.velocity[0]),
            ("Vy", sample_state.velocity[1]),
            ("kappa", sample_state.thermal_conductivity)]
        sample_viz_fields.extend(
            ("Y_"+species_names[i], sample_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        holder_viz_fields = [
            ("solid_mass", holder_state.cv.mass),
            ("rhoE_s", holder_state.cv.energy),
            ("temperature", holder_state.dv.temperature),
            ("kappa", holder_state.dv.thermal_conductivity)]

        write_visfile(dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t, overwrite=True, comm=comm)
        write_visfile(dcoll, sample_viz_fields, sample_visualizer,
            vizname=vizname+"-sample", step=step, t=t, overwrite=True, comm=comm)
        write_visfile(dcoll, holder_viz_fields, holder_visualizer,
            vizname=vizname+"-holder", step=step, t=t, overwrite=True, comm=comm)

    def my_write_restart(step, t, fluid_state, sample_state, holder_state):
        if rank == 0:
            print("Writing restart file...")

        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "fluid_cv": fluid_state.cv,
                "fluid_temperature_seed": fluid_state.temperature,
                "sample_cv": sample_state.cv,
                "sample_density": sample_state.dv.wall_density,
                "sample_temperature_seed": sample_state.temperature,
                "holder_cv": holder_state.cv,
                "holder_temperature_seed": holder_state.dv.temperature,
                "nspecies": nspecies,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }

            write_restart_file(actx, restart_data, restart_fname, comm)

##########################################################################

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

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        fluid_cv, fluid_tseed, \
            sample_cv, sample_tseed, sample_density, \
            holder_cv = state

        fluid_cv = force_evaluation(actx, fluid_cv)
        fluid_tseed = force_evaluation(actx, fluid_tseed)
        sample_cv = force_evaluation(actx, sample_cv)
        sample_tseed = force_evaluation(actx, sample_tseed)
        sample_density = force_evaluation(actx, sample_density)
        holder_cv = force_evaluation(actx, holder_cv)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(fluid_cv, fluid_tseed)
        fluid_cv = fluid_state.cv

        # construct species-limited solid state
        sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)
        sample_cv = sample_state.cv

        # construct species-limited solid state
        holder_state = get_holder_state(holder_cv)

        try:
            state = make_obj_array([
                fluid_cv, fluid_state.temperature,
                sample_cv, sample_state.temperature, sample_density,
                holder_cv])

            do_garbage = check_step(step=step, interval=ngarbage)
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

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

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    sample_state=sample_state, holder_state=holder_state)

            if do_restart:
                my_write_restart(step, t, fluid_state, sample_state, holder_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                sample_state=sample_state, holder_state=holder_state)
            raise

        return state, dt

    def my_rhs(time, state):

        fluid_cv, fluid_tseed, \
            sample_cv, sample_tseed, sample_density, \
            holder_cv = state

        # construct species-limited fluid state
        fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model_fluid,
            temperature_seed=fluid_tseed,
            limiter_func=_limit_fluid_cv, limiter_dd=dd_vol_fluid)
        fluid_cv = fluid_state.cv

        # construct species-limited solid state
        sample_state = make_fluid_state(cv=sample_cv, gas_model=gas_model_sample,
            wall_density=sample_density,
            temperature_seed=sample_tseed,
            limiter_func=_limit_sample_cv, limiter_dd=dd_vol_sample)
        sample_cv = sample_state.cv

        # construct species-limited solid state
        holder_state = _get_holder_state(holder_cv)
        holder_cv = holder_state.cv

        # ~~~~~~~~~~~~~

        fluid_all_boundaries_no_grad, sample_all_boundaries_no_grad = \
            add_multiphysics_interface_boundaries_no_grad(
                dcoll, dd_vol_fluid, dd_vol_sample,
                fluid_state, sample_state,
                fluid_boundaries, sample_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                use_kappa_weighted_grad_flux_in_fluid=False,
                wall_penalty_amount=wall_penalty_amount)

        fluid_all_boundaries_no_grad, holder_all_boundaries_no_grad = \
            add_thermal_interface_boundaries_no_grad(
                dcoll,
                dd_vol_fluid, dd_vol_holder,
                fluid_state, holder_state.dv.thermal_conductivity,
                holder_state.dv.temperature,
                fluid_all_boundaries_no_grad, holder_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                use_kappa_weighted_grad_flux_in_fluid=False)

        sample_all_boundaries_no_grad, holder_all_boundaries_no_grad = \
            add_thermal_interface_boundaries_no_grad(
                dcoll,
                dd_vol_sample, dd_vol_holder,
                sample_state, holder_state.dv.thermal_conductivity,
                holder_state.dv.temperature,
                sample_all_boundaries_no_grad, holder_all_boundaries_no_grad,
                interface_noslip=True, interface_radiation=False,
                use_kappa_weighted_grad_flux_in_fluid=False)

        # ~~~~~~~~~~~~~~

        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model_fluid, fluid_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_fluid, comm_tag=_FluidOpStatesTag,
            limiter_func=_limit_fluid_cv)

        sample_operator_states_quad = make_operator_fluid_states(
            dcoll, sample_state, gas_model_sample, sample_all_boundaries_no_grad,
            quadrature_tag, dd=dd_vol_sample, comm_tag=_WallOpStatesTag,
            limiter_func=_limit_sample_cv)

        # ~~~~~~~~~~~~~~

        # fluid grad CV
        fluid_grad_cv = grad_cv_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradCVTag
        )

        # fluid grad T
        fluid_grad_temperature = grad_t_operator(
            dcoll, gas_model_fluid, fluid_all_boundaries_no_grad, fluid_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            comm_tag=_FluidGradTempTag
        )

        # sample grad CV
        sample_grad_cv = grad_cv_operator(
            dcoll, gas_model_sample, sample_all_boundaries_no_grad, sample_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            comm_tag=_SampleGradCVTag
        )

        # sample grad T
        sample_grad_temperature = grad_t_operator(
            dcoll, gas_model_sample, sample_all_boundaries_no_grad, sample_state,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            comm_tag=_SampleGradTempTag
        )

        # holder grad T
        holder_grad_temperature = wall_grad_t_operator(
            dcoll, holder_state.dv.thermal_conductivity,
            holder_all_boundaries_no_grad, holder_state.dv.temperature,
            quadrature_tag=quadrature_tag, dd=dd_vol_holder,
            comm_tag=_HolderGradTempTag
        )

        # ~~~~~~~~~~~~~~~~~

        fluid_all_boundaries, sample_all_boundaries = \
            add_multiphysics_interface_boundaries(
                dcoll, dd_vol_fluid, dd_vol_sample,
                fluid_state, sample_state,
                fluid_grad_cv, sample_grad_cv,
                fluid_grad_temperature, sample_grad_temperature,
                fluid_boundaries, sample_boundaries,
                interface_noslip=True, interface_radiation=use_radiation,
                wall_emissivity=emissivity, sigma=5.67e-8,
                ambient_temperature=300.0,
                use_kappa_weighted_grad_flux_in_fluid=False,
                wall_penalty_amount=wall_penalty_amount)

        fluid_all_boundaries, holder_all_boundaries = \
            add_thermal_interface_boundaries(
                dcoll, dd_vol_fluid, dd_vol_holder,
                fluid_all_boundaries, holder_boundaries,
                fluid_state, holder_state.dv.thermal_conductivity,
                holder_state.dv.temperature,
                fluid_grad_temperature, holder_grad_temperature,
                interface_noslip=True, interface_radiation=use_radiation,
                use_kappa_weighted_grad_flux_in_fluid=False,
                wall_emissivity=emissivity, sigma=5.67e-8,
                ambient_temperature=300.0,
                wall_penalty_amount=wall_penalty_amount)

        sample_all_boundaries, holder_all_boundaries = \
            add_thermal_interface_boundaries(
                dcoll, dd_vol_sample, dd_vol_holder,
                sample_all_boundaries, holder_all_boundaries,
                sample_state, holder_state.dv.thermal_conductivity,
                holder_state.dv.temperature,
                sample_grad_temperature, holder_grad_temperature,
                interface_noslip=True, interface_radiation=False,
                use_kappa_weighted_grad_flux_in_fluid=False,
                wall_penalty_amount=wall_penalty_amount)

        # ~~~~~~~~~~~~~

        fluid_rhs = ns_operator(
            dcoll, gas_model_fluid, fluid_state, fluid_all_boundaries,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            operator_states_quad=fluid_operator_states_quad,
            grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature,
            comm_tag=_FluidOperatorTag, inviscid_terms_on=False)

        sample_rhs = ns_operator(
            dcoll, gas_model_sample, sample_state, sample_all_boundaries,
            time=time, quadrature_tag=quadrature_tag, dd=dd_vol_sample,
            operator_states_quad=sample_operator_states_quad,
            grad_cv=sample_grad_cv, grad_t=sample_grad_temperature,
            comm_tag=_SampleOperatorTag, inviscid_terms_on=False)

        holder_rhs = diffusion_operator(
            dcoll, holder_state.dv.thermal_conductivity, holder_all_boundaries,
            holder_state.dv.temperature,
            penalty_amount=wall_penalty_amount, quadrature_tag=quadrature_tag,
            dd=dd_vol_holder, grad_u=holder_grad_temperature,
            comm_tag=_HolderOperatorTag)

        sample_mass_rhs = sample_zeros

        # ~~~~~~~~~~~~~
        return make_obj_array([
            fluid_rhs, fluid_zeros,
            sample_rhs, sample_zeros, sample_mass_rhs,
            holder_rhs])

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

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

        return state, dt

##############################################################################

    stepper_state = make_obj_array([
        fluid_state.cv, fluid_state.temperature,
        sample_state.cv, sample_state.temperature, sample_state.dv.wall_density,
        holder_state.cv])

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

    fluid_cv, fluid_tseed, \
        sample_cv, sample_tseed, sample_density, \
        holder_cv = stepper_state

    fluid_state = get_fluid_state(fluid_cv, fluid_tseed)
    sample_state = get_sample_state(sample_cv, sample_density, sample_tseed)
    holder_state = get_holder_state(holder_cv)

    my_write_viz(step=final_step, t=final_t, dt=dt, fluid_state=fluid_state,
                 sample_state=sample_state, holder_state=holder_state)

    my_write_restart(final_step, final_t, fluid_state, sample_state,
                     holder_state)

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
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "coupled_volumes"
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

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=True)

    main(actx_class, use_logmgr=args.log,
         use_profiling=args.profile, casename=casename,
         lazy=args.lazy, restart_filename=restart_file)
