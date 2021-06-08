"""Provide some utilities for building simulation applications.

General utilities
-----------------

.. autofunction:: check_step
.. autofunction:: inviscid_sim_timestep

Diagnostic callbacks
--------------------

.. autofunction:: sim_visualization
.. autofunction:: sim_checkpoint
.. autofunction:: sim_healthcheck
.. autofunction:: compare_with_analytic_solution

Mesh utilities
--------------

.. autofunction:: generate_and_distribute_mesh
"""

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
from meshmode.dof_array import thaw
from mirgecom.io import make_status_message
from mirgecom.inviscid import get_inviscid_timestep  # bad smell?
from mirgecom.exceptions import SynchronizedError, SimulationHealthError

logger = logging.getLogger(__name__)


def check_step(step, interval):
    """
    Check step number against a user-specified interval.

    Utility is used typically for visualization.

    - Negative numbers mean 'never visualize'.
    - Zero means 'always visualize'.

    Useful for checking whether the current step is an output step,
    or anyting else that occurs on fixed intervals.
    """
    if interval == 0:
        return True
    elif interval < 0:
        return False
    elif step % interval == 0:
        return True
    return False


def inviscid_sim_timestep(discr, state, t, dt, cfl, eos,
                          t_final, constant_cfl=False):
    """Return the maximum stable dt."""
    mydt = dt
    if constant_cfl is True:
        mydt = get_inviscid_timestep(discr=discr, q=state,
                                     cfl=cfl, eos=eos)
    if (t + mydt) > t_final:
        mydt = t_final - t
    return mydt


def sim_visualization(discr, eos, state, visualizer, vizname,
                      step=0, t=0,
                      exact_soln=None, viz_fields=None,
                      overwrite=False, vis_timer=None):
    """Visualize the simulation fields.

    Write VTK output of the conserved state and and specified derived
    quantities *viz_fields*.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state.
    state
        State array which expects at least the conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.
    visualizer:
        A :class:`meshmode.discretization.visualization.Visualizer`
        VTK output object.
    """
    from contextlib import nullcontext
    from mirgecom.fluid import split_conserved
    from mirgecom.io import make_rank_fname, make_par_fname

    cv = split_conserved(discr.dim, state)
    dependent_vars = eos.dependent_vars(cv)

    io_fields = [
        ("cv", cv),
        ("dv", dependent_vars)
    ]
    if exact_soln is not None:
        actx = cv.mass.array_context
        nodes = thaw(actx, discr.nodes())
        expected_state = exact_soln(x_vec=nodes, t=t, eos=eos)
        exact_list = [
            ("exact_soln", expected_state),
        ]
        io_fields.extend(exact_list)

    if viz_fields is not None:
        io_fields.extend(viz_fields)

    comm = discr.mpi_communicator
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    rank_fn = make_rank_fname(basename=vizname, rank=rank, step=step, t=t)

    if vis_timer:
        ctm = vis_timer.start_sub_timer()
    else:
        ctm = nullcontext()

    with ctm:
        visualizer.write_parallel_vtk_file(
            comm, rank_fn, io_fields,
            overwrite=overwrite,
            par_manifest_filename=make_par_fname(
                basename=vizname, step=step, t=t
            )
        )


def sim_checkpoint(discr, visualizer, eos, q, vizname, exact_soln=None,
                   step=0, t=0, dt=0, cfl=1.0, nstatus=-1, nviz=-1, exittol=1e-16,
                   constant_cfl=False, viz_fields=None, overwrite=False,
                   vis_timer=None):
    """Checkpoint the simulation status.

    Checkpoints the simulation status by reporting relevant diagnostic
    quantities, such as pressure/temperature, and visualization.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state *q*.
    q
        State array which expects at least the conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.
    nstatus: int
        An integer denoting the step frequency for performing status checks.
    nviz: int
        An integer denoting the step frequency for writing vtk output.
    """
    exception = None
    comm = discr.mpi_communicator
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    try:
        # Status checks
        if check_step(step=step, interval=nstatus):
            from mirgecom.fluid import split_conserved

            #        if constant_cfl is False:
            #            current_cfl = get_inviscid_cfl(discr=discr, q=q,
            #                                           eos=eos, dt=dt)

            cv = split_conserved(discr.dim, q)
            dependent_vars = eos.dependent_vars(cv)
            msg = make_status_message(discr=discr,
                                    t=t, step=step, dt=dt, cfl=cfl,
                                    dependent_vars=dependent_vars)
            if rank == 0:
                logger.info(msg)

            # Check the health of the simulation
            sim_healthcheck(discr, eos, q, step=step, t=t)

            # Compare with exact solution, if provided
            if exact_soln is not None:
                compare_with_analytic_solution(
                    discr, eos, q, exact_soln,
                    step=step, t=t, exittol=exittol
                )

        # Visualization
        if check_step(step=step, interval=nviz):
            sim_visualization(
                discr, eos, q, visualizer, vizname,
                step=step, t=t,
                exact_soln=exact_soln, viz_fields=viz_fields,
                overwrite=overwrite, vis_timer=vis_timer
            )
    except SynchronizedError as err:
        exception = err

    terminate = True if exception is not None else False

    if comm is not None:
        from mpi4py import MPI

        terminate = comm.allreduce(terminate, MPI.LOR)

    if terminate:
        if rank == 0:
            logger.info("Visualizing crashed state ...")

        # Write out crashed field
        sim_visualization(discr, eos, q,
                          visualizer, vizname=vizname,
                          step=step, t=t,
                          viz_fields=viz_fields)
        raise exception


def sim_healthcheck(discr, eos, state, step=0, t=0):
    """Check the global health of the fluids state.

    Determine the health of a state by inspecting for unphyiscal
    values of pressure and temperature.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state.
    state
        State array which expects at least the conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.
    """
    from mirgecom.fluid import split_conserved
    import grudge.op as op

    # NOTE: Derived quantities are functions of the conserved variables.
    # Therefore is it sufficient to check for unphysical values of
    # temperature and pressure.
    cv = split_conserved(discr.dim, state)
    dependent_vars = eos.dependent_vars(cv)
    pressure = dependent_vars.pressure
    temperature = dependent_vars.temperature

    # Check for NaN
    if (np.isnan(op.nodal_sum_loc(discr, "vol", pressure))
            or np.isnan(op.nodal_sum_loc(discr, "vol", temperature))):
        raise SimulationHealthError(
            message=("Simulation exited abnormally; detected a NaN.")
        )

    # Check for non-positivity
    if (op.nodal_min_loc(discr, "vol", pressure) < 0
            or op.nodal_min_loc(discr, "vol", temperature) < 0):
        raise SimulationHealthError(
            message=("Simulation exited abnormally; "
                        "found non-positive values for pressure or temperature.")
        )

    # Check for blow-up
    if (op.nodal_sum_loc(discr, "vol", pressure) == np.inf
            or op.nodal_sum_loc(discr, "vol", temperature) == np.inf):
        raise SimulationHealthError(
            message=("Simulation exited abnormally; "
                        "derived quantities are not finite.")
        )


def compare_with_analytic_solution(discr, eos, state, exact_soln,
                                   step=0, t=0, exittol=None):
    """Compute the infinite norm of the problem residual.

    Computes the infinite norm of the residual with respect to a specified
    exact solution *exact_soln*. If the error is larger than *exittol*,
    raises a :class:`mirgecom.exceptions.SimulationHealthError`.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state.
    state
        State array which expects at least the conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.
    exact_soln:
        A callable for the exact solution with signature:
        ``exact_soln(x_vec, t, eos)`` where `x_vec` are the nodes,
        `t` is time, and `eos` is a :class:`mirgecom.eos.GasEOS`.
    """
    if exittol is None:
        exittol = 1e-16

    actx = discr._setup_actx
    nodes = thaw(actx, discr.nodes())
    expected_state = exact_soln(x_vec=nodes, t=t, eos=eos)
    exp_resid = state - expected_state
    norms = [discr.norm(v, np.inf) for v in exp_resid]

    comm = discr.mpi_communicator
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    statusmesg = (
        f"Errors: {step=} {t=}\n"
        f"------- errors = "
        + ", ".join("%.3g" % err_norm for err_norm in norms)
    )

    if rank == 0:
        logger.info(statusmesg)

    if max(norms) > exittol:
        raise SimulationHealthError(
            message=("Simulation exited abnormally; "
                        "solution doesn't agree with analytic result.")
        )


def generate_and_distribute_mesh(comm, generate_mesh):
    """Generate a mesh and distribute it among all ranks in *comm*.

    Generate the mesh with the user-supplied mesh generation function
    *generate_mesh*, partition the mesh, and distribute it to every
    rank in the provided MPI communicator *comm*.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the mesh
    generate_mesh:
        Callable of zero arguments returning a :class:`meshmode.mesh.Mesh`.
        Will only be called on one (undetermined) rank.

    Returns
    -------
    local_mesh : :class:`meshmode.mesh.Mesh`
        The local partition of the the mesh returned by *generate_mesh*.
    global_nelements : :class:`int`
        The number of elements in the serial mesh
    """
    from meshmode.distributed import (
        MPIMeshDistributor,
        get_partition_by_pymetis,
    )
    num_parts = comm.Get_size()
    mesh_dist = MPIMeshDistributor(comm)
    global_nelements = 0

    if mesh_dist.is_mananger_rank():

        mesh = generate_mesh()

        global_nelements = mesh.nelements

        part_per_element = get_partition_by_pymetis(mesh, num_parts)
        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    return local_mesh, global_nelements


def create_parallel_grid(comm, generate_grid):
    """Generate and distribute mesh compatibility interface."""
    from warnings import warn
    warn("Do not call create_parallel_grid; use generate_and_distribute_mesh "
         "instead. This function will disappear August 1, 2021",
         DeprecationWarning, stacklevel=2)
    return generate_and_distribute_mesh(comm=comm, generate_mesh=generate_grid)
