"""Support for time series logging."""

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

__doc__ = """
.. autoclass:: StateConsumer
.. autoclass:: DiscretizationBasedQuantity
.. autoclass:: KernelProfile
.. autoclass:: PythonMemoryUsage
.. autoclass:: DeviceMemoryUsageCUDA
.. autoclass:: DeviceMemoryUsageAMD
.. autofunction:: initialize_logmgr
.. autofunction:: logmgr_add_cl_device_info
.. autofunction:: logmgr_add_simulation_info
.. autofunction:: logmgr_add_device_memory_usage
.. autofunction:: logmgr_add_many_discretization_quantities
.. autofunction:: logmgr_add_mempool_usage
.. autofunction:: add_package_versions
.. autofunction:: set_sim_state
.. autofunction:: logmgr_set_time
"""

from collections.abc import Collection
import logging

from logpyle import (LogQuantity, PostLogQuantity, LogManager,
    MultiPostLogQuantity, add_run_info,
    add_general_quantities, add_simulation_quantities)
from arraycontext.container import get_container_context_recursively
from meshmode.array_context import PyOpenCLArrayContext
from grudge.discretization import DiscretizationCollection
import pyopencl as cl

from typing import Optional, Callable, Union, Tuple, Dict, Any
import numpy as np

from grudge.dof_desc import DD_VOLUME_ALL
import grudge.op as oper
from typing import List


logger = logging.getLogger(__name__)

MemPoolType = Union[cl.tools.MemoryPool, cl.tools.SVMPool]


def initialize_logmgr(enable_logmgr: bool,
                      filename: Optional[str] = None, mode: str = "wu",
                      mpi_comm=None) -> Optional[LogManager]:
    """Create and initialize a mirgecom-specific :class:`logpyle.LogManager`."""
    if not enable_logmgr:
        return None

    logmgr = LogManager(filename=filename, mode=mode, mpi_comm=mpi_comm)

    logmgr.add_quantity(PythonInitTime())
    logmgr.enable_save_on_sigterm()

    add_run_info(logmgr)
    add_package_versions(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)

    try:
        from logpyle import GCStats
        logmgr.add_quantity(GCStats())
    except ImportError:
        from warnings import warn
        warn("GCStats not found, not collecting GC statistics. Please update your "
             "logpyle installation.")

    try:
        logmgr.add_quantity(PythonMemoryUsage())
    except ImportError:
        from warnings import warn
        warn("psutil module not found, not tracking memory consumption."
             "Install it with 'pip install psutil'")

    return logmgr


def logmgr_add_cl_device_info(logmgr: LogManager, queue: cl.CommandQueue) -> None:
    """Add information about the OpenCL device to the log."""
    if queue:
        dev = queue.device
        logmgr.set_constant("cl_device_name", str(dev))
        logmgr.set_constant("cl_device_version", dev.version)
        logmgr.set_constant("cl_platform_version", dev.platform.version)


def logmgr_add_simulation_info(logmgr: LogManager,
                               sim_info: Dict[str, Any]) -> None:
    """Add some user-defined information to the logpyle output."""
    for field_name in sim_info:
        logmgr.set_constant(field_name, sim_info[field_name])


def logmgr_add_device_name(logmgr: LogManager, queue: cl.CommandQueue):  # noqa: D401
    """Deprecated. Do not use in new code."""
    from warnings import warn
    warn("logmgr_add_device_name is deprecated and will disappear in Q3 2021. "
         "Use logmgr_add_cl_device_info instead.", DeprecationWarning,
         stacklevel=2)
    logmgr_add_cl_device_info(logmgr, queue)


def logmgr_add_device_memory_usage(logmgr: LogManager, queue: cl.CommandQueue) \
        -> None:
    """Add the OpenCL device memory usage to the log."""
    if not queue:
        return

    if queue.device.vendor.lower().startswith("nvidia"):
        logmgr.add_quantity(DeviceMemoryUsageCUDA())
    elif queue.device.vendor.lower().startswith("advanced micro devices"):
        logmgr.add_quantity(DeviceMemoryUsageAMD(queue.device))


def logmgr_add_mempool_usage(logmgr: LogManager, pool: MemPoolType) -> None:
    """Add the memory pool usage to the log."""
    if (not isinstance(pool, cl.tools.MemoryPool)
            and not isinstance(pool, cl.tools.SVMPool)):
        return
    logmgr.add_quantity(MempoolMemoryUsage(pool))


def logmgr_add_many_discretization_quantities(logmgr: LogManager, dcoll, dim,
        extract_vars_for_logging, units_for_logging, dd=DD_VOLUME_ALL) -> None:
    """Add default discretization quantities to the logmgr."""
    if dd != DD_VOLUME_ALL:
        suffix = f"_{dd.domain_tag.tag}"
    else:
        suffix = ""

    for reduction_op in ["min", "max", "L2_norm"]:
        for quantity in ["pressure"+suffix, "temperature"+suffix]:
            logmgr.add_quantity(DiscretizationBasedQuantity(
                dcoll, quantity, reduction_op, extract_vars_for_logging,
                units_for_logging, dd=dd))

        for quantity in ["mass"+suffix, "energy"+suffix]:
            logmgr.add_quantity(DiscretizationBasedQuantity(
                dcoll, quantity, reduction_op, extract_vars_for_logging,
                units_for_logging, dd=dd))

        for d in range(dim):
            logmgr.add_quantity(DiscretizationBasedQuantity(
                dcoll, "momentum"+suffix, reduction_op, extract_vars_for_logging,
                units_for_logging, axis=d, dd=dd))


# {{{ Package versions

def add_package_versions(mgr: LogManager, path_to_version_sh: Optional[str] = None) \
        -> None:
    """Add the output of the emirge version.sh script to the log.

    Parameters
    ----------
    mgr
        The :class:`logpyle.LogManager` to add the versions to.

    path_to_version_sh
        Path to emirge's version.sh script. The function will attempt to find this
        script automatically if this argument is not specified.

    """
    import subprocess
    from warnings import warn

    output = None

    # Find emirge's version.sh in any parent directory
    if path_to_version_sh is None:
        import pathlib
        import mirgecom

        p = pathlib.Path(mirgecom.__file__).resolve()

        for d in p.parents:
            candidate = pathlib.Path(d).joinpath("version.sh")
            if candidate.is_file():
                with open(candidate) as f:
                    if "emirge" in f.read():
                        path_to_version_sh = str(candidate)
                        break

    if path_to_version_sh is None:
        warn("Could not find emirge's version.sh.")

    else:
        from pytools import ProcessLogger
        try:
            with ProcessLogger(logger, "emirge's version.sh"):
                output = subprocess.check_output(path_to_version_sh)
        except OSError as e:
            warn("Could not record emirge's package versions: " + str(e))

    mgr.set_constant("emirge_package_versions", output)

# }}}


# {{{ State handling

def set_sim_state(mgr: LogManager, dim, state, eos) -> None:
    """Update the simulation state of all :class:`StateConsumer` of the log manager.

    Parameters
    ----------
    mgr
        The :class:`logpyle.LogManager` whose :class:`StateConsumer` quantities
        will receive *state*.
    """
    state_vars = {}

    for gd_lst in [mgr.before_gather_descriptors,
            mgr.after_gather_descriptors]:
        for gd in gd_lst:
            if isinstance(gd.quantity, StateConsumer):
                extract_state_vars_func = gd.quantity.extract_state_vars
                if extract_state_vars_func not in state_vars:
                    state_vars[extract_state_vars_func] = \
                        extract_state_vars_func(dim, state, eos)

                gd.quantity.set_state_vars(state_vars[extract_state_vars_func])


def logmgr_set_time(mgr: LogManager, steps: int, time: float) -> None:
    """Set the (current/initial) time/step count explicitly (e.g., for restart)."""
    from logpyle import TimestepCounter, SimulationTime

    for gd_lst in [mgr.before_gather_descriptors,
            mgr.after_gather_descriptors]:
        for gd in gd_lst:
            if isinstance(gd.quantity, TimestepCounter | TimeStepProfile):
                gd.quantity.steps = steps
            if isinstance(gd.quantity, SimulationTime):
                gd.quantity.t = time


class StateConsumer:
    """Base class for quantities that require a state for logging.

    .. automethod:: __init__
    .. automethod:: set_state_vars
    """

    def __init__(self, extract_vars_for_logging: Callable):
        """Store the function to extract state variables.

        Parameters
        ----------
        extract_vars_for_logging(dim, state, eos)
            Returns a dict(quantity_name: values) of the state vars for a particular
            state.
        """
        self.extract_state_vars = extract_vars_for_logging
        self.state_vars: Optional[np.ndarray] = None

    def set_state_vars(self, state_vars: np.ndarray) -> None:
        """Update the state vector of the object."""
        self.state_vars = state_vars

# }}}

# {{{ Discretization-based quantities


class DiscretizationBasedQuantity(PostLogQuantity, StateConsumer):
    """Logging support for physical quantities.

    Possible rank aggregation operations (``op``) are: min, max, L2_norm.
    """

    def __init__(self, dcoll: DiscretizationCollection, quantity: str, op: str,
                 extract_vars_for_logging, units_logging, name: Optional[str] = None,
                 axis: Optional[int] = None, dd=DD_VOLUME_ALL):
        unit = units_logging(quantity)

        if name is None:
            name = f"{op}_{quantity}" + (str(axis) if axis is not None else "")

        LogQuantity.__init__(self, name, unit)
        StateConsumer.__init__(self, extract_vars_for_logging)

        self.dcoll = dcoll

        self.quantity = quantity
        self.axis = axis

        from functools import partial

        if op == "min":
            self._discr_reduction = partial(oper.nodal_min, self.dcoll, dd)
            self.rank_aggr = min
        elif op == "max":
            self._discr_reduction = partial(oper.nodal_max, self.dcoll, dd)
            self.rank_aggr = max
        elif op == "L2_norm":
            self._discr_reduction = partial(oper.norm, self.dcoll, p=2, dd=dd)
            self.rank_aggr = max
        else:
            raise ValueError(f"unknown operation {op}")

    @property
    def default_aggregator(self):
        """Rank aggregator to use."""
        return self.rank_aggr

    def __call__(self):
        """Return the requested quantity."""
        if self.state_vars is None:
            return None

        quantity = self.state_vars[self.quantity]

        actx = get_container_context_recursively(quantity)

        if self.axis is not None:  # e.g. momentum
            quantity = quantity[self.axis]

        return actx.to_numpy(self._discr_reduction(quantity))[()]

# }}}


# {{{ Kernel profile quantities

class KernelProfile(MultiPostLogQuantity):
    """Logging support for statistics of the OpenCL kernel profiling (num_calls, \
    time, flops, bytes_accessed, footprint).

    All statistics except num_calls are averages.

    Parameters
    ----------
    actx
        The array context from which to collect statistics. Must have profiling
        enabled in the OpenCL command queue.

    kernel_name
        Name of the kernel to profile.
    """

    def __init__(self, actx: PyOpenCLArrayContext,
                 kernel_name: str) -> None:
        from mirgecom.profiling import PyOpenCLProfilingArrayContext
        assert isinstance(actx, PyOpenCLProfilingArrayContext)

        from dataclasses import fields
        from mirgecom.profiling import MultiCallKernelProfile

        units_default = {"num_calls": "1",  "time": "s", "flops": "GFlops",
                         "bytes_accessed": "GByte", "footprint_bytes": "GByte"}

        names = [f"{kernel_name}_{f.name}" for f in fields(MultiCallKernelProfile)]
        units = [units_default[f.name] for f in fields(MultiCallKernelProfile)]

        super().__init__(names, units)

        self.kernel_name = kernel_name
        self.actx = actx

    def __call__(self) -> List[Optional[float]]:
        """Return the requested kernel profile quantity."""
        r = self.actx.get_profiling_data_for_kernel(self.kernel_name)
        self.actx.reset_profiling_data_for_kernel(self.kernel_name)

        return [r.num_calls, r.time.mean(), r.flops.mean(), r.bytes_accessed.mean(),
                r.footprint_bytes.mean()]

# }}}


# {{{ Memory profiling

class PythonMemoryUsage(PostLogQuantity):
    """Logging support for Python memory usage (RSS, host).

    Uses :mod:`psutil` to track memory usage. Virtually no overhead.
    """

    def __init__(self, name: Optional[str] = None):

        if name is None:
            name = "memory_usage_python"

        super().__init__(name, "MByte", description="Memory usage (RSS, host)")

        import psutil  # pylint: disable=import-error
        self.process = psutil.Process()

    def __call__(self) -> float:
        """Return the memory usage in MByte."""
        return self.process.memory_info()[0] / 1024 / 1024


class DeviceMemoryUsageCUDA(PostLogQuantity):
    """Logging support for Nvidia CUDA GPU memory usage."""

    def __init__(self, name: Optional[str] = None) -> None:

        if name is None:
            name = "memory_usage_gpu"

        super().__init__(name, "MByte", description="Memory usage (GPU)")

        import ctypes

        try:
            # See https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549#gistcomment-3654335  # noqa
            # on why this calls cuMemGetInfo_v2 and not cuMemGetInfo
            libcuda = ctypes.cdll.LoadLibrary("libcuda.so")
            self.mem_func: Optional[Callable] = libcuda.cuMemGetInfo_v2
        except OSError:
            self.mem_func = None

    def __call__(self) -> Optional[float]:
        """Return the memory usage in MByte."""
        if self.mem_func is None:
            return None

        import ctypes
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        ret = self.mem_func(ctypes.byref(free), ctypes.byref(total))

        if ret != 0:
            from warnings import warn
            warn(f"cuMemGetInfo failed with error {ret}.")
            return None
        else:
            if free.value / total.value < 0.1:
                from warnings import warn
                warn(
                    "The memory usage on the GPU is approaching the memory "
                    f"size, with less than 10% free of "
                    f"{total.value // 1024 // 1024} MByte total. "
                    "This may lead to slowdowns or crashes.")
            return (total.value - free.value) / 1024 / 1024


class DeviceMemoryUsageAMD(PostLogQuantity):
    """Logging support for AMD GPU memory usage."""

    def __init__(self, dev: cl.Device, name: Optional[str] = None) -> None:

        if name is None:
            name = "memory_usage_gpu"

        super().__init__(name, "MByte", description="Memory usage (GPU)")

        self.dev = dev
        self.global_mem_size_mbyte = dev.global_mem_size / 1024 / 1024

    def __call__(self) -> Optional[float]:
        """Return the memory usage in MByte."""
        # NB: dev.global_mem_size is in Bytes,
        #     dev.global_free_memory_amd is in KByte,
        #     the actual granularity of the returned values appears to be MByte
        #     (like in CUDA)

        return self.global_mem_size_mbyte - self.dev.global_free_memory_amd[0] / 1024


class MempoolMemoryUsage(MultiPostLogQuantity):
    """Logging support for memory pool usage."""

    def __init__(self, pool: MemPoolType, names: Optional[List[str]] = None) -> None:
        if names is None:
            names = ["memory_usage_mempool_managed", "memory_usage_mempool_active"]

        descs = ["Memory pool managed", "Memory pool active"]

        super().__init__(names, ["MByte", "MByte"], descriptions=descs)

        self.pool = pool

    def __call__(self) -> Tuple[float, float]:
        """Return the memory pool usage in MByte."""
        return (self.pool.managed_bytes/1024/1024,
                self.pool.active_bytes/1024/1024)


class PythonInitTime(PostLogQuantity):
    """Stores the Python startup time.

    Measures the time from process start to when this quantity is initialized.
    """

    def __init__(self, name: str = "t_python_init") -> None:
        LogQuantity.__init__(self, name, "s", "Python init time")

        try:
            import psutil
        except ModuleNotFoundError:
            from warnings import warn
            warn("Measuring the Python init time requires the 'psutil' module.")
            self.done = True
        else:
            from time import time
            self.python_init_time = time() - psutil.Process().create_time()
            self.done = False

    def __call__(self) -> Optional[float]:
        """Return the Python init time in seconds."""
        if self.done:
            return None

        self.done = True
        return self.python_init_time


class TimeStepProfile(MultiPostLogQuantity):
    def __init__(self, timesteps: Collection[int]) -> None:
        # Could add support for MPI profile, kernel profile, etc. in the future
        import pyinstrument
        names = ["pyinstrument_profile"]
        units = ["1"]
        descriptions = ["Pyinstrument profile"]

        super().__init__(names, units, descriptions)

        self.steps = 0
        self.timesteps_to_profile = frozenset(timesteps)
        self.profiler = pyinstrument.Profiler()

    def prepare_for_tick(self) -> None:
        step = self.steps
        if step in self.timesteps_to_profile:
            self.profiler.start()
        self.steps += 1

    def __call__(self):
        if self.steps-1 not in self.timesteps_to_profile:
            return None

        self.profiler.stop()
        s = self.profiler.output_html()
        self.profiler.reset()
        return s


# }}}
