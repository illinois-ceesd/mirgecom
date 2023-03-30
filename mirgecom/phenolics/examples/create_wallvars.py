"""."""

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
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer

from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh,
    check_step
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedFluidBoundary
from mirgecom.initializers import Vortex2D
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state
from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging

from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    set_sim_state
)

import sys

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    dim = 2
    nel_1d = 16
    box_ll = -5.0
    box_ur = 5.0

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                            b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
    local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                generate_mesh)
    local_nelements = local_mesh.nelements

    order = 3
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    #~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    from mirgecom.phenolics.gas import gas_properties
    my_gas = gas_properties()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    # soln setup and init
    import mirgecom.phenolics.phenolics as wall 
    
    solid_species_mass = np.empty((3,), dtype=object)
    solid_species_mass[0] =  30.0 + nodes[0]*0.0
    solid_species_mass[1] =  90.0 + nodes[0]*0.0
    solid_species_mass[2] = 160.0 + nodes[0]*0.0

    gas_density = 1.0 + nodes[0]*0.0

    temperature = 100*nodes[0] + 800.0 + 0.1

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    import mirgecom.phenolics.tacot as my_composite

    wall_vars = wall.initializer(composite=my_composite,
        solid_species_mass=solid_species_mass,
        gas_density=gas_density, temperature=temperature, progress=0.0)

    eos = wall.PhenolicsEOS(composite=my_composite, gas=my_gas)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    gas_data = my_gas._data
    bounds = gas_data[:,0]

    idx = temperature*0.0
    for i in range(bounds.shape[0]-1):
        aux = actx.np.where(
                actx.np.greater(temperature, bounds[i] + 1e-7),
                    actx.np.where(actx.np.less(temperature, bounds[i+1]),
                        i,
                        0),
                    0
                )

        idx = idx + aux

    print(idx)
    sys.exit()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    wdv = eos.dependent_vars(wv=wall_vars, temperature_seed=temperature-10.0, idx=idx)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    vis_timer = None
    visualizer = make_visualizer(dcoll)

    from mirgecom.simutil import write_visfile
    def my_write_viz(step, t, wall_vars, dep_vars):

        viz_fields = [("species_mass", wall_vars.gas_species_mass),
                      ("gas_density", wall_vars.gas_density),
                      ("energy", wall_vars.energy),
                      ("DV", dep_vars),
                     ]

        viz_fields.extend((
            ("phase_1", wall_vars.solid_species_mass[0]),
            ("phase_2", wall_vars.solid_species_mass[1]),
            ("phase_3", wall_vars.solid_species_mass[2]),
        ))

        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer,
                      comm=comm)

    my_write_viz(step=0, t=0.0, wall_vars=wall_vars, dep_vars=wdv)



if __name__ == "__main__":
    import argparse
    casename = "check_wallvars"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
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
        rst_filename = args.restart_file

    main(actx_class, use_logmgr=args.log, use_leap=args.leap, lazy=lazy,
         use_profiling=args.profiling, casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
