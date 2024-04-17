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
import os
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


def _check_cache_dirs_node() -> None:
    """Check whether multiple ranks share cache directories on the same node."""
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()

    if size <= 1:
        return

    with shared_split_comm_world() as node_comm:
        node_rank = node_comm.Get_rank()

        def _check_var(var: str) -> None:
            from warnings import warn

            try:
                my_path = os.environ[var]
            except KeyError:
                warn(f"Please set the '{var}' variable in your job script to "
                    "avoid file system overheads when running on large numbers of "
                    "ranks. See https://mirgecom.readthedocs.io/en/latest/running/large-systems.html "  # noqa: E501
                    "for more information.")
                # Create a fake path so there will not be a second warning below.
                my_path = f"no/such/path/rank{node_rank}"

            all_paths = node_comm.gather(my_path, root=0)

            if node_rank == 0:
                assert all_paths
                if len(all_paths) != len(set(all_paths)):
                    hostname = MPI.Get_processor_name()
                    dup = [path for path in set(all_paths)
                                if all_paths.count(path) > 1]

                    from warnings import warn
                    warn(f"Multiple ranks are sharing '{var}' on node '{hostname}'. "
                        f"Duplicate '{var}'s: {dup}.")

        _check_var("XDG_CACHE_HOME")
        _check_var("POCL_CACHE_DIR")

        # We haven't observed an issue yet that 'CUDA_CACHE_PATH' fixes,
        # so disable this check for now.
        # _check_var("CUDA_CACHE_PATH")


def _check_gpu_oversubscription() -> None:
    """
    Check whether multiple ranks are running on the same GPU on each node.

    Only works with CUDA devices currently due to the use of the
    PCI_DOMAIN_ID_NV extension.
    """
    from mpi4py import MPI
    import pyopencl as cl

    size = MPI.COMM_WORLD.Get_size()

    if size <= 1:
        return

    # This may unnecessarily require pyopencl in case we run with a
    # NumpyArrayContext or CupyArrayContext
    cl_ctx = cl.create_some_context()
    dev = cl_ctx.devices

    # No support for multi-device contexts
    if len(dev) > 1:
        raise NotImplementedError("multi-device contexts not yet supported")

    dev = dev[0]

    # Allow running multiple ranks on non-GPU devices
    if not (dev.type & cl.device_type.GPU):
        return

    with shared_split_comm_world() as node_comm:
        try:
            domain_id = hex(dev.pci_domain_id_nv)
        except (cl._cl.LogicError, AttributeError):
            from warnings import warn
            warn("Cannot detect whether multiple ranks are running on the"
                 " same GPU because it requires Nvidia GPUs running with"
                 " pyopencl>2021.1.1 and (Nvidia CL or pocl>1.6).")
            return

        node_rank = node_comm.Get_rank()

        bus_id = hex(dev.pci_bus_id_nv)
        slot_id = hex(dev.pci_slot_id_nv)

        dev_id = (domain_id, bus_id, slot_id)

        dev_ids = node_comm.gather(dev_id, root=0)

        if node_rank == 0:
            assert dev_ids
            if len(dev_ids) != len(set(dev_ids)):
                hostname = MPI.Get_processor_name()
                dup = [item for item in dev_ids if dev_ids.count(item) > 1]

                from warnings import warn
                warn(f"Multiple ranks are sharing GPUs on node '{hostname}'. "
                     f"Duplicate PCIe IDs: {dup}.")


def log_disk_cache_config() -> None:
    """Log the disk cache configuration."""
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    res = f"Rank {rank} disk cache config: "

    # This may unnecessarily require pyopencl in case we run with a
    # NumpyArrayContext or CupyArrayContext
    import pyopencl as cl
    from pyopencl.characterize import nv_compute_capability, get_pocl_version
    cl_ctx = cl.create_some_context()
    dev = cl_ctx.devices[0]

    # Variables set any to any value => cache is disabled
    loopy_cache_enabled = bool(os.getenv("LOOPY_NO_CACHE", True))
    pyopencl_cache_enabled = bool(os.getenv("PYOPENCL_NO_CACHE", True))

    loopy_cache_dir = ("(" + os.getenv("XDG_CACHE_HOME", "default dir") + ")"
                       if loopy_cache_enabled else "")
    pyopencl_cache_dir = ("(" + os.getenv("XDG_CACHE_HOME", "default dir") + ")"
                          if pyopencl_cache_enabled else "")

    res += f"loopy: {loopy_cache_enabled} {loopy_cache_dir}; "
    res += f"pyopencl: {pyopencl_cache_enabled} {pyopencl_cache_dir}; "

    if get_pocl_version(dev.platform) is not None:
        # Variable set to '0' => cache is disabled
        pocl_cache_enabled = os.getenv("POCL_KERNEL_CACHE", "1") != "0"
        pocl_cache_dir = ("(" + os.getenv("POCL_CACHE_DIR", "default dir") + ")"
                        if pocl_cache_enabled else "")

        res += f"pocl: {pocl_cache_enabled} {pocl_cache_dir}; "

    if nv_compute_capability(dev) is not None:
        # Variable set to '1' => cache is disabled
        cuda_cache_enabled = os.getenv("CUDA_CACHE_DISABLE", "0") != "1"
        cuda_cache_dir = ("(" + os.getenv("CUDA_CACHE_PATH", "default dir") + ")"
                          if cuda_cache_enabled else "")
        res += f"cuda: {cuda_cache_enabled} {cuda_cache_dir};"

    res += "\n"
    logger.info(res)


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
    import mpi4py

    if mpi4py.__version__ < "4":
        from warnings import warn
        warn(f"mpi4py version {mpi4py.__version__} does not support pkl5 "
              "scatter. This may lead to errors when distributing large meshes. "
              "Please upgrade to the git version of mpi4py.")

    else:
        logger.info(f"Using mpi4py version {mpi4py.__version__} with pkl5 "
                     "scatter support.")


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

        _check_gpu_oversubscription()
        _check_cache_dirs_node()
        _check_isl_version()
        _check_mpi4py_version()
        log_disk_cache_config()

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
