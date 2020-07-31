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
import os
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from mirgecom.euler import split_fields
from mirgecom.checkstate import get_field_stats


def make_io_fields(dim, state, dv, eos):
    r"""Makes fluid-flow-specific fields for restart or
    visualization io.

    Parameters
    ----------
    dim
        Dimensionality of solution
    state
        Solution state
    dv
        EOS-specific dependent quantities
        (e.g. pressure, temperature, for ideal monatomic gas)
    eos
        Equation of state utility for resolving the dependent
        fields.
    """
    io_fields = split_fields(dim, state)
    io_fields += eos.split_fields(dim, dv)
    return io_fields


def make_init_message(dim, order, nelements, dt, t_final,
                      nstatus, nviz, cfl, constant_cfl,
                      initname, eosname, casename):
    return(
        f"Initialization for Case({casename})\n"
        f"===\n"
        f"Num {dim}d order-{order} elements: nelements\n"
        f"Timestep:        {dt}\n"
        f"Final time:      {t_final}\n"
        f"CFL:             {cfl}\n"
        f"Constant CFL:    {constant_cfl}\n"
        f"Initialization:  {initname}\n"
        f"EOS:             {eosname}\n"
    )


def make_status_message(t, step, dt, cfl, dv):
    dvxt = get_field_stats(dv)
    statusmsg = (
        f"Status: Step({step}) Time({t})\n"
        f"------   P({dvxt[0]},{dvxt[2]})\n"
        f"------   T({dvxt[1]},{dvxt[3]})\n"
        f"------   dt,cfl = ({dt},{cfl})"
    )
    return statusmsg


def make_serial_fname(basename, step=0, t=0):
    return f"{basename}-{step:06d}.vtu"


def make_rank_fname(basename, rank=0, step=0, t=0):
    return f"{basename}-{step:06d}-{{rank:04d}}.vtu"


def make_par_fname(basename, step=0, t=0):
    return f"{basename}-{step:06d}.pvtu"


def make_output_dump(visualizer, basename, io_fields,
                     comm=None, step=0, t=0, overwrite=True):
    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.Get_rank()
        nproc = comm.Get_size()
    if nproc > 1:
        rank_fn = make_rank_fname(basename=basename, rank=rank, step=step, t=t)
        par_fn = make_par_fname(basename=basename, step=step, t=t)
        visualizer.write_parallel_vtk_file(comm, rank_fn, io_fields, overwrite=True)
        if rank == 0:
            os.rename(rank_fn.format(rank=rank).replace("vtu", "pvtu"), par_fn)
    else:
        fname = make_serial_fname(basename=basename, step=step, t=t)
        visualizer.write_vtk_file(fname, io_fields, overwrite=True)
