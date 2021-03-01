"""I/O - related functions and utilities.

.. autofunction:: make_init_message
.. autofunction:: make_rank_fname
.. autofunction:: make_par_fname
.. autofunction:: write_visualization_file
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

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa


def make_init_message(*, dim, order, casename, extra_init=None,
        nelements=0, global_nelements=0):
    """Create a summary of some general simulation parameters and inputs."""
    initmsg = (
        f"Initialization for Case({casename})\n"
        f"===\n"
        f"Num {dim}d order-{order} elements: {nelements}\n"
        f"Num global elements: {global_nelements}\n")
    if extra_init is not None:
        initmsg += extra_init
    return initmsg


def make_rank_fname(basename, rank=0, step=0, t=0):
    """Create a rank-specific filename."""
    return f"{basename}-{step:06d}-{{rank:04d}}.vtu"


def make_par_fname(basename, step=0, t=0):
    r"""Make parallel visualization filename."""
    return f"{basename}-{step:06d}.pvtu"


def write_visualization_file(visualizer, fields, basename, step, t, comm=None,
        overwrite=False, timer=None):
    """Write a VTK visualization file."""
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()

    from mirgecom.io import make_rank_fname, make_par_fname
    rank_fn = make_rank_fname(basename=basename, rank=rank, step=step, t=t)

    from contextlib import nullcontext
    with timer.start_sub_timer() if timer is not None else nullcontext():
        visualizer.write_parallel_vtk_file(
            comm, rank_fn, fields, overwrite=overwrite,
            par_manifest_filename=make_par_fname(basename=basename, step=step, t=t))
