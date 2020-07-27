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


def make_visfile_name(basename, rank=0, step=0, t=0):
    nameform = basename + "-{iorank:04d}-{iostep:06d}.vtu"
    return nameform.format(iorank=rank, iostep=step)


def checkpoint(discr, logger, visualizer, nstatus, nviz, rank, basename,
               eos, state, dim, t, step, dt, cfl):
    do_status = False
    do_viz = False

    if nstatus > 0 and nstatus % step == 0:
        do_status = True
    if nviz > 0 and nviz % step == 0:
        do_viz = True

    if do_status is False and do_viz is False:
        return 0

    dim = discr.dim
    dv = eos(state)

    if do_status:
        statusmesg = make_status_message(t=t, step=step, dt=dt,
                                         cfl=cfl, dv=dv)
        if rank == 0:
            logger.info(statusmesg)

    if do_viz:
        visfilename = make_visfile_name(basename=basename, rank=rank,
                                        step=step, t=t)
        io_fields = make_io_fields(dim, state, dv, eos)
        visualizer.write_vtk_file(visfilename, io_fields)

    return 0
