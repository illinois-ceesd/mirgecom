"""I/O - related functions and utilities.

.. autofunction:: make_status_message
.. autofunction:: make_rank_fname
.. autofunction:: make_par_fname
.. autofunction:: read_and_distribute_yaml_data
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
from functools import partial
import grudge.op as op
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.dof_desc import DD_VOLUME_ALL


def make_init_message(*, dim, order, dt, t_final,
                      nstatus, nviz, cfl, constant_cfl,
                      initname, eosname, casename,
                      nelements=0, global_nelements=0):
    """Create a summary of some general simulation parameters and inputs."""
    return(
        f"Initialization for Case({casename})\n"
        f"===\n"
        f"Num {dim}d order-{order} elements: {nelements}\n"
        f"Num global elements: {global_nelements}\n"
        f"Timestep:        {dt}\n"
        f"Final time:      {t_final}\n"
        f"CFL:             {cfl}\n"
        f"Constant CFL:    {constant_cfl}\n"
        f"Initialization:  {initname}\n"
        f"EOS:             {eosname}\n"
    )


def make_status_message(
        *, discr, t, step, dt, cfl, dependent_vars, fluid_volume_dd=DD_VOLUME_ALL):
    r"""Make simulation status and health message."""
    dv = dependent_vars
    _min = partial(op.nodal_min, discr, fluid_volume_dd)
    _max = partial(op.nodal_max, discr, fluid_volume_dd)
    statusmsg = (
        f"Status: {step=} {t=}\n"
        f"------- P({_min(dv.pressure):.3g}, {_max(dv.pressure):.3g})\n"
        f"------- T({_min(dv.temperature):.3g}, {_max(dv.temperature):.3g})\n"
        f"------- {dt=} {cfl=}"
    )
    return statusmsg


def make_rank_fname(basename, rank=0, step=0, t=0):
    """Create a rank-specific filename."""
    return f"{basename}-{step:06d}-{{rank:04d}}.vtu"


def make_par_fname(basename, step=0, t=0):
    r"""Make parallel visualization filename."""
    return f"{basename}-{step:06d}.pvtu"


def read_and_distribute_yaml_data(mpi_comm, file_path):
    """Read a YAML file on one rank, broadcast result to world."""
    import yaml
    rank = mpi_comm.Get_rank()
    if rank == 0:
        with open(file_path) as f:
            input_data = yaml.load(f, Loader=yaml.FullLoader)
    else:
        input_data = None
    mpi_comm.bcast(input_data, root=0)
    return input_data
