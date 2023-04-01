"""Demonstrate heat source example."""

__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
#import grudge.op as op
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from mirgecom.discretization import create_discretization_collection

#from mirgecom.integrators import rk4_step
from mirgecom.integrators.lsrk import euler_step

from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary,
    PrescribedFluxDiffusionBoundary
)

from mirgecom.simutil import (
    check_naninf_local,
    generate_and_distribute_mesh
)

from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage)

from logpyle import IntervalTimer, set_dt

import sys  # noqa

#########################################

@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_file=None):
    """Run the example."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename="heat-source.sqlite", mode="wo", mpi_comm=comm)

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

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    viz_path = "viz_data/"
    vizname = viz_path+casename

    dim = 2

    t = 0
    t_final = 60.0
    istep = 0

    order = 1
    dt = 0.002

####################################

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    )
    if restart_file:  # read the grid from restart data
        rst_filename = f"{restart_file}"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == num_parts

    else:  # generate the grid from scratch
        from functools import partial
        from meshmode.mesh.generation import generate_regular_rect_mesh
        box_ll = 0.00
        box_ur = 0.05
        nel_1d = 61
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(box_ll,)*dim, b=(box_ur,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={"prescribed": ["+y"],
                                  "neumann": ["-x", "+x", "-y"]}
        )
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    nodes = actx.thaw(dcoll.nodes())

    zeros = nodes[0]*0.0

#######################################   

#    # ablation workshop case #1.0
#    def my_presc_func(**kwargs):
#        time = kwargs['time']

#        if time >= 0.0 and time < 0.1:
#            surface_temperature = (1644-300)*(time/0.1) + 300.0

#        if time >= 0.1 and time < 60.1:
#            surface_temperature = 1644.0

#        return surface_temperature 

    # ablation workshop case #2.1
    def my_presc_func(u_tpair, kappa_tpair, grad_u_tpair, normal, **kwargs):
        time = kwargs['time']

        flux = actx.np.where(actx.np.less(time, 0.1),
            0.3*(time/0.1)*1.5e6*(time/0.1),
            0.3*1.5e6
        )

        #FIXME make emissivity a function of "tau"
        return flux - 0.8*5.567e-8*u_tpair.int**4

    boundaries = {
        BoundaryDomainTag("prescribed"): PrescribedFluxDiffusionBoundary(my_presc_func),
#        BoundaryDomainTag("prescribed"): DirichletDiffusionBoundary(my_presc_func),
        BoundaryDomainTag("neumann"): NeumannDiffusionBoundary(0.)
    }

#######################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    import mirgecom.phenolics.phenolics as wall
    import mirgecom.phenolics.tacot as my_composite
    from mirgecom.phenolics.gas import GasProperties
    my_gas = GasProperties()

    eos = wall.PhenolicsEOS(composite=my_composite, gas=my_gas)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    # soln setup and init
    solid_species_mass = np.empty((3,), dtype=object)
    solid_species_mass[0] = 30.0 + nodes[0]*0.0
    solid_species_mass[1] = 90.0 + nodes[0]*0.0
    solid_species_mass[2] = 160. + nodes[0]*0.0

    pressure = 0.0 + nodes[0]*0.0

    temperature = 300.0 + nodes[0]*0.0

    pressure = force_evaluation(actx, pressure)
    temperature = force_evaluation(actx, temperature)
    solid_species_mass = force_evaluation(actx, solid_species_mass)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    if restart_file:
        t = restart_data["t"]
        istep = restart_data["step"]
        wall_vars = restart_data["wall_vars"]
    else:
        # Set the current state from time 0
        wall_vars = wall.initializer(eos=eos,
            solid_species_mass=solid_species_mass,
            pressure=pressure, temperature=temperature, progress=0.0)

    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, istep, t)

    eos = wall.PhenolicsEOS(composite=my_composite, gas=my_gas)

    wdv = eos.dependent_vars(wv=wall_vars, temperature_seed=temperature)
       
    wall_vars = force_evaluation(actx, wall_vars)
    wdv = force_evaluation(actx, wdv)

#######################################

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.3e} s, "),
            ("t_sim.max", "sim time: {value:7.3f} s, "),
            ("t_step.max", "step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    vis = make_visualizer(dcoll)

#######################################

    pyrolysis = my_composite.Pyrolysis()
    def _rhs(t, state):

        wv, tseed = state
        wdv = eos.dependent_vars(wv=wv, temperature_seed=tseed)

        kappa = wdv.thermal_conductivity
        temperature = wdv.temperature

        #~~~~~
        energy_rhs = diffusion_operator(dcoll, kappa=kappa,
            boundaries=boundaries, u=temperature, time=t)

        viscous_rhs = wall.make_conserved(
            solid_species_mass=wv.solid_species_mass*0.0,
            gas_density=zeros, energy=energy_rhs)

        #~~~~~
        pyrolysis_rhs = pyrolysis.get_sources(temperature, wv.solid_species_mass)

        source_terms = wall.make_conserved(solid_species_mass=pyrolysis_rhs,
            gas_density=zeros, energy=zeros)

        #~~~~~
        return make_obj_array([viscous_rhs + source_terms, tseed*0.0])

    compiled_rhs = actx.compile(_rhs)

#######################################

    visualizer = make_visualizer(dcoll)

    from mirgecom.simutil import write_visfile
    def my_write_viz(step, t, wall_vars, dep_vars):

        viz_fields = [("gas_density", wall_vars.gas_density),
                      #("energy", wall_vars.energy),
                      ("DV", dep_vars),
                     ]

        viz_fields.extend((
            ("phase_1", wall_vars.solid_species_mass[0]),
            ("phase_2", wall_vars.solid_species_mass[1]),
            ("phase_3", wall_vars.solid_species_mass[2]),
        ))

        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
            step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)

    def my_write_restart(step, t, wall_vars, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "wall_vars": wall_vars,
                "tseed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

###############################################

    from warnings import warn
    warn("Running gc.collect() to work around memory growth issue ")
    import gc
    gc.collect()

    from pytools.obj_array import make_obj_array

    my_write_viz(step=istep, t=t, wall_vars=wall_vars, dep_vars=wdv)
    print('Solution file' + str(istep))

    while t < t_final:

        if logmgr:
            logmgr.tick_before()

        tseed = wdv.temperature
        state = make_obj_array([ wall_vars, tseed ])

#        state = rk4_step(state, t, dt, _rhs)
        state = euler_step(state, t, dt, compiled_rhs)
        state = force_evaluation(actx, state)

        wall_vars, tseed = state

        wdv = eos.dependent_vars(wv=wall_vars, temperature_seed=tseed)

        t += dt
        istep += 1

        if check_naninf_local(dcoll, "vol", wdv.temperature):
            logmgr.info(f"{rank=}: NANs/INFs in temperature data.")

        if istep%100 == 0:
            my_write_viz(step=istep, t=t, wall_vars=wall_vars, dep_vars=wdv)

        if istep%100 == 0:
            gc.collect()

#        if istep%1000 == 0:
#            my_write_restart(step=istep, t=t, wall_vars=wall_vars,
#                         tseed=wdv.temperature)

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    if logmgr:
        logmgr.close()

##############################################################################

if __name__ == "__main__":
    import argparse
    casename = "conduction"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    lazy = args.lazy
    if args.profiling:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename

    rst_filename = None
    if args.restart_file:
        rst_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {rst_filename}")

    main(actx_class, use_logmgr=args.log, use_leap=args.leap, lazy=lazy,
         use_profiling=args.profiling, casename=casename,
         restart_file=rst_filename)

# vim: foldmethod=marker
