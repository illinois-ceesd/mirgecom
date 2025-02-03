"""MPI helper functionality.

.. autofunction:: mpi_entry_point
.. autofunction:: pudb_remote_debug_on_single_rank
.. autofunction:: enable_rank_labeled_print
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

from functools import wraps
import sys

from contextlib import contextmanager
from typing import Callable, Any, Generator, TYPE_CHECKING

import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from mpi4py.MPI import Comm


@contextmanager
def shared_split_comm_world() -> Generator["Comm", None, None]:
    """Create a context manager for a MPI.COMM_TYPE_SHARED comm."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

    try:
        yield comm
    finally:
        comm.Free()


def _check_isl_version() -> None:
    """
    Check that we run with a non-GMP ISL version.

    In general, ISL can be built with 3 options, imath-32, GMP, and imath,
    in descending order of speed.
    Since https://github.com/conda-forge/isl-feedstock only offers imath-32
    or GMP, we can check for the presence of GMP-only symbols in the loaded
    library to determine if we are running with imath-32.
    """
    import ctypes
    import islpy  # type: ignore[import-untyped]

    try:
        ctypes.cdll.LoadLibrary(islpy._isl.__file__).isl_val_get_num_gmp
    except AttributeError:
        # We are running with imath or imath-32.
        pass
    else:
        from warnings import warn
        warn("Running with the GMP version of ISL, which is considerably "
             "slower than imath-32. Please install a faster ISL version with "
             "a command such as 'conda install \"isl * imath32_*\"' .")


def _check_mpi4py_version() -> None:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    import mpi4py

    if mpi4py.__version__ < "4":
        from warnings import warn
        warn(f"mpi4py version {mpi4py.__version__} does not support pkl5 "
              "scatter. This may lead to errors when distributing large meshes. "
              "Please upgrade to version 4+ of mpi4py.")

    else:
        logger.info(f"Using mpi4py version {mpi4py.__version__} with pkl5 "
                     "scatter support.")

    mpi_ver = MPI.Get_version()

    logger.info(f"mpi4py is using '{MPI.Get_library_version()}' "
                f"with MPI v{mpi_ver[0]}.{mpi_ver[1]}.")


def mpi_entry_point(func) -> Callable:
    """
    Return a decorator that designates a function as the "main" function for MPI.

    Declares that all MPI code that will be executed on the current process is
    contained within *func*. Calls `MPI_Init()`/`MPI_Init_thread()` and sets up a
    hook to call `MPI_Finalize()` on exit.
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs) -> None:
        # We enforce this so that an exception raised on one rank terminates
        # all ranks.
        if "mpi4py.run" not in sys.modules:
            raise RuntimeError("Must run MPI scripts via mpi4py (i.e., 'python -m "
                        "mpi4py <args>').")

        # We enforce this so that we can work around certain interoperability issues,
        # including the hwloc/mprobe ones below.
        if "mpi4py.MPI" in sys.modules:
            raise RuntimeError("mpi4py.MPI imported before designated MPI entry "
                        "point. Check for prior imports.")

        # Avoid hwloc version conflicts by forcing pocl to load before mpi4py
        # (don't ask). See https://github.com/illinois-ceesd/mirgecom/pull/169
        # for details.
        import pyopencl as cl
        cl.get_platforms()

        # Avoid https://github.com/illinois-ceesd/mirgecom/issues/132 on
        # some MPI runtimes. This must be set *before* the first import
        # of mpi4py.MPI.
        import mpi4py
        mpi4py.rc.recv_mprobe = False

        # Runs MPI_Init()/MPI_Init_thread() and sets up a hook for MPI_Finalize() on
        # exit
        from mpi4py import MPI  # noqa

        _check_isl_version()
        _check_mpi4py_version()

        func(*args, **kwargs)

    return wrapped_func


def pudb_remote_debug_on_single_rank(func: Callable) -> Callable:
    """
    Designate a function *func* to be debugged with ``pudb`` on rank 0.

    To use it, add this decorator to the main function that you want to debug,
    after the :func:`mpi_entry_point` decorator:

    .. code-block:: python

        @mpi_entry_point
        @pudb_remote_debug_on_single_rank
        def main(...)


    Then, you can connect to pudb on rank 0 by running
    ``telnet 127.0.0.1 6899`` in a separate terminal and continue to use pudb
    as normal.
    """
    @wraps(func)
    def wrapped_func(*args: Any, **kwargs: Any) -> None:
        # pylint: disable=import-error
        from pudb.remote import debug_remote_on_single_rank
        from mpi4py import MPI
        debug_remote_on_single_rank(MPI.COMM_WORLD, 0, func, *args, **kwargs)

    return wrapped_func


def enable_rank_labeled_print() -> None:
    """Enable prepending the rank number to every message printed with print()."""
    def rank_print(*args, **kwargs):
        """Prepends the rank number to the print function."""
        if "mpi4py.MPI" in sys.modules:
            from mpi4py import MPI
            out_str = f"[{MPI.COMM_WORLD.Get_rank()}]"
        else:
            out_str = "[ ]"

        __builtins__["oldprint"](out_str, *args, **kwargs)

    if "oldprint" not in __builtins__:  # type: ignore[operator]
        __builtins__["oldprint"] = __builtins__["print"]  # type: ignore[index]
    __builtins__["print"] = rank_print
