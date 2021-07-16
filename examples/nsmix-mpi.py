"""Demonstrate combustive mixture with Pyrometheus."""

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
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.inviscid import (
    get_inviscid_timestep
)
from mirgecom.transport import SimpleTransport
from mirgecom.viscous import get_viscous_timestep
from mirgecom.navierstokes import ns_operator
# from mirgecom.heat import heat_operator

from mirgecom.simutil import (
    sim_checkpoint,
    check_step,
    generate_and_distribute_mesh,
    ExactSolutionMismatch
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (  # noqa
    AdiabaticSlipBoundary,
    IsothermalNoSlipBoundary,
)
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import PyrometheusMixture
from mirgecom.fluid import split_conserved
import cantera
import pyrometheus as pyro

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context):
    """Drive example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nel_1d = 8
    order = 1

    # This example runs only 3 steps by default (to keep CI ~short)
    # With the mixture defined below, equilibrium is achieved at ~40ms
    # To run to equlibrium, set t_final >= 40ms.
    t_final = 3e-9
    current_cfl = 1.0
    velocity = np.zeros(shape=(dim,))
    current_dt = 1e-9
    current_t = 0
    constant_cfl = False
    nstatus = 1
    nviz = 5
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    box_ll = -0.005
    box_ur = 0.005
    error_state = False
    debug = False

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, n=(nel_1d,) * dim)
    local_mesh, global_nelements = generate_and_distribute_mesh(comm, generate_mesh)
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.
    from mirgecom.mechanisms import get_mechanism_cti
    mech_cti = get_mechanism_cti("uiuc")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    init_temperature = 1500.0  # Initial temperature hot enough to burn
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 3.0
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index("C2H4")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    x = np.zeros(nspecies)
    # Set the species mole fractions according to our desired fuel/air mixture
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm  # pylint: disable=no-member
    # one_atm = 101325.0

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({init_temperature}, {one_atm}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = init_temperature, one_atm, x
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    can_t, can_rho, can_y = cantera_soln.TDY
    can_p = cantera_soln.P
    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # {{{ Initialize simple transport model
    kappa = 1e-5
    spec_diffusivity = 1e-5 * np.ones(nspecies)
    sigma = 1e-5
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)
    # }}}

    # Create a Pyrometheus EOS with the Cantera soln. Pyrometheus uses Cantera and
    # generates a set of methods to calculate chemothermomechanical properties and
    # states for this particular mechanism.
    casename = "nsmix"
    pyrometheus_mechanism = pyro.get_thermochem_class(cantera_soln)(actx.np)
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=init_temperature,
                             transport_model=transport_model)

    # }}}

    # {{{ MIRGE-Com state initialization

    # Initialize the fluid/gas state with Cantera-consistent data:
    # (density, pressure, temperature, mass_fractions)
    print(f"Cantera state (rho,T,P,Y) = ({can_rho}, {can_t}, {can_p}, {can_y}")
    initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                     pressure=can_p, temperature=can_t,
                                     massfractions=can_y, velocity=velocity)

    #    my_boundary = AdiabaticSlipBoundary()
    my_boundary = IsothermalNoSlipBoundary(wall_temperature=can_t)
    visc_bnds = {BTAG_ALL: my_boundary}
    current_state = initializer(eos=eos, x_vec=nodes, t=0)

    # Inspection at physics debugging time
    if debug:
        cv = split_conserved(dim, current_state)
        print("Initial MIRGE-Com state:")
        print(f"{cv.mass=}")
        print(f"{cv.energy=}")
        print(f"{cv.momentum=}")
        print(f"{cv.species_mass=}")
        print(f"Initial Y: {cv.species_mass / cv.mass}")
        print(f"Initial DV pressure: {eos.pressure(cv)}")
        print(f"Initial DV temperature: {eos.temperature(cv)}")

    # }}}

    visualizer = make_visualizer(discr, order + 3
                                 if discr.dim == 2 else order)
    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)

    # Cantera equilibrate calculates the expected end state @ chemical equilibrium
    # i.e. the expected state after all reactions
    cantera_soln.equilibrate("UV")
    eq_temperature, eq_density, eq_mass_fractions = cantera_soln.TDY
    eq_pressure = cantera_soln.P

    # Report the expected final state to the user
    if rank == 0:
        logger.info(init_message)
        logger.info(f"Expected equilibrium state:"
                    f" {eq_pressure=}, {eq_temperature=},"
                    f" {eq_density=}, {eq_mass_fractions=}")

    def get_timestep(state):
        next_dt = current_dt
        t_end = t_final
        if constant_cfl is True:
            inviscid_dt = get_inviscid_timestep(discr=discr, eos=eos,
                                                cfl=current_cfl, q=state)
            viscous_dt = get_viscous_timestep(discr=discr, eos=eos,
                                              transport_model=transport_model,
                                              cfl=current_cfl, q=state)
            next_dt = min([next_dt, inviscid_dt, viscous_dt])
        # else:
        #     inviscid_cfl = get_inviscid_cfl(discr=discr, eos=eos,
        #                                     dt=next_dt, state=state)
        #     viscous_cfl = get_viscous_cfl(discr, eos=eos,
        #                                   transport_model=transport_model,
        #                                   dt=next_dt, state=state)
        if(current_t + next_dt) > t_end:
            next_dt = t_end - current_t

        return next_dt

    def my_rhs(t, state):
        cv = split_conserved(dim=dim, q=state)
        ns_rhs = ns_operator(discr, q=state, t=t,
                             boundaries=visc_bnds, eos=eos)
        reaction_source = eos.get_species_source_terms(cv)
        return ns_rhs + reaction_source

    def my_checkpoint(step, t, dt, state):
        cv = split_conserved(dim, state)
        reaction_rates = eos.get_production_rates(cv)
        viz_fields = [("reaction_rates", reaction_rates)]
        return sim_checkpoint(discr, visualizer, eos, q=state,
                              vizname=casename, step=step,
                              t=t, dt=dt, nstatus=nstatus, nviz=nviz,
                              constant_cfl=constant_cfl, comm=comm,
                              viz_fields=viz_fields)

    try:
        (current_step, current_t, current_state) = \
            advance_state(rhs=my_rhs, timestepper=timestepper,
                          checkpoint=my_checkpoint,
                          get_timestep=get_timestep, state=current_state,
                          t=current_t, t_final=t_final)
    except ExactSolutionMismatch as ex:
        error_state = True
        current_step = ex.step
        current_t = ex.t
        current_state = ex.state

    if not check_step(current_step, nviz):  # If final step not an output step
        if rank == 0:
            logger.info("Checkpointing final state ...")
        my_checkpoint(current_step, t=current_t,
                      dt=(current_t - checkpoint_t),
                      state=current_state)

    if current_t - t_final < 0:
        error_state = True

    if error_state:
        raise ValueError("Simulation did not complete successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

# vim: foldmethod=marker
