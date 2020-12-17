"""Demonstrate combustive mixture with Prometheus."""

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


from mirgecom.euler import inviscid_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    create_parallel_grid,
    ExactSolutionMismatch
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import PrometheusMixture
from mirgecom.euler import split_conserved, join_conserved
# from mirgecom.prometheus import UIUCMechanism
import cantera
import prometheus as pyro

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context):
    """Drive example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nel_1d = 2
    order = 1

    t_final = 1e-7
    current_cfl = 1.0
    velocity = np.zeros(shape=(dim,))
    # velocity[:dim] = 1.0
    current_dt = 1e-8
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
    error_state = 0

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_grid = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, n=(nel_1d,) * dim)
    local_mesh, global_nelements = create_parallel_grid(comm, generate_grid)
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    casename = "autoignition"
    cantera_soln = cantera.Solution("uiuc.cti", "gas")
    prometheus_mechanism = pyro.get_thermochem_class(cantera_soln)(actx.np)
    # prometheus_mechanism = UIUCMechanism(actx.np)

    init_temperature = 1500.0
    eos = PrometheusMixture(prometheus_mechanism, tguess=init_temperature)

    # Use Cantera for initialization (and soon for pyro code gen)
    #    cantera_soln = cantera.Solution("uiuc.cti", "gas")
    nspecies = prometheus_mechanism.num_species
    # These are used for Cantera - doesn't work on Lassen
    #    equiv_ratio = 1.0
    #    ox_di_ratio = 0.21
    #    stoich_ratio = 3.0
    #    i_fu = cantera_soln.species_index("C2H4")
    #    i_ox = cantera_soln.species_index("O2")
    #    i_di = cantera_soln.species_index("N2")
    #    x = np.zeros(nspecies)
    #    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    #    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    #    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    #    one_atm = cantera.one_atm

    #    print(f"Input state (T,P,X) = ({init_temperature}, {one_atm}, {x}")
    #    cantera_soln.TPX = init_temperature, one_atm, x
    #    can_t, can_rho, can_y = cantera_soln.TDY
    #    can_p = cantera_soln.P
    # Cantera state (rho,T,P,Y) = (0.23397065362031969, 1500.0, 101325.0,
    # [0.06372925 0.21806609 0.         0.         0.         0. 0.71820466]

    can_t = init_temperature
    can_p = 101325.0
    can_rho = 0.23397065362031969
    can_y = np.zeros(nspecies)
    can_y[0] = 0.06372925
    can_y[1] = .21806609
    can_y[6] = .71820466
    print(f"Input state (rho,T,P,Y) = ({can_rho}, {can_t}, {can_p}, {can_y}")
    initializer = MixtureInitializer(numdim=dim, nspecies=nspecies,
                                     pressure=can_p, temperature=can_t,
                                     massfractions=can_y, velocity=velocity)

    my_boundary = AdiabaticSlipBoundary()
    boundaries = {BTAG_ALL: my_boundary}
    current_state = initializer(eos=eos, x_vec=nodes, t=0)
    cv = split_conserved(dim, current_state)

    print(f"Initial CV rho: {cv.mass}")
    print(f"Initial CV rhoE: {cv.energy}")
    print(f"Initial CV rhoV: {cv.momentum}")
    print(f"Initial CV rhoY: {cv.massfractions}")
    print(f"Initial Y: {cv.massfractions / cv.mass}")

    print(f"Initial DV pressure: {eos.pressure(cv)}")
    print(f"Initial DV temperature: {eos.temperature(cv)}")

    def my_chem_sources(discr, q, eos):
        cv = split_conserved(dim, q)
        omega = eos.get_production_rates(cv)
        w = eos.get_species_molecular_weights()
        species_sources = w * omega
        rho_source = 0 * cv.mass
        mom_source = 0 * cv.momentum
        energy_source = 0 * cv.energy
        return join_conserved(dim, rho_source, energy_source, mom_source,
                              species_sources)
    sources = {my_chem_sources}

    visualizer = make_visualizer(discr, discr.order + 3
                                 if discr.dim == 2 else discr.order)
    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
#    cantera_soln.equilibrate("UV")
#    can_eq_t, can_eq_rho, can_eq_y = cantera_soln.TDY
#    can_eq_p = cantera_soln.P
    #    can_eq_e = cantera_soln.int_energy_mass
    #    can_eq_k = cantera_soln.forward_rate_constants
    #    can_eq_c = cantera_soln.concentrations
    #    can_eq_r = cantera_soln.net_rates_of_progress
    #    can_eq_omega = cantera_soln.net_production_rates
    if rank == 0:
        logger.info(init_message)
        #        logger.info(f"Expected (p,T,rho, y) = ({can_eq_p}, {can_eq_t},"
        #                    f" {can_eq_rho}, {can_eq_y})")

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_rhs(t, state):
        return inviscid_operator(discr, q=state, t=t,
                                 boundaries=boundaries, eos=eos,
                                 sources=sources)

    def my_checkpoint(step, t, dt, state):
        global checkpoint_t
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
        error_state = 1
        current_step = ex.step
        current_t = ex.t
        current_state = ex.state

    if current_t != checkpoint_t:  # This check because !overwrite
        if rank == 0:
            logger.info("Checkpointing final state ...")
        my_checkpoint(current_step, t=current_t,
                      dt=(current_t - checkpoint_t),
                      state=current_state)

    if current_t - t_final < 0:
        error_state = 1

    if error_state:
        raise ValueError("Simulation did not complete successfully.")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

# vim: foldmethod=marker
