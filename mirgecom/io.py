"""I/O - related functions and utilities.

.. autofunction:: make_status_message
.. autofunction:: make_rank_fname
.. autofunction:: make_par_fname
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


def make_init_message(*,
        casename, dim, order, nelements, global_nelements,
        extra_params_dict=None):
    """Create a summary of some general simulation parameters and inputs."""
    msg = (
        f"Initialization for Case({casename})\n"
        f"===\n"
        f"Num {dim}d order-{order} elements: {nelements}\n"
        f"Num global elements: {global_nelements}\n")
    if extra_params_dict is not None:
        for name, value in extra_params_dict.items():
            if value is not None:
                msg += (
                    f"{name}: {value}\n")
    return msg


def make_status_message(*, discr, t, step, dt, cfl, dependent_vars):
    r"""Make simulation status and health message."""
    dv = dependent_vars
    from functools import partial
    _min = partial(discr.nodal_min, "vol")
    _max = partial(discr.nodal_max, "vol")
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
