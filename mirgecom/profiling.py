"""An array context with kernel profiling capabilities."""

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

from meshmode.array_context import PyOpenCLArrayContext
import pyopencl as cl
from pytools.py_codegen import PythonFunctionGenerator
import loopy as lp
import numpy as np
from dataclasses import dataclass
import pytools
from logpyle import LogManager
from mirgecom.logging_quantities import KernelProfile
from mirgecom.utils import StatisticsAccumulator

__doc__ = """
.. autoclass:: PyOpenCLProfilingArrayContext
.. autoclass:: SingleCallKernelProfile
.. autoclass:: MultiCallKernelProfile
"""


@dataclass
class SingleCallKernelProfile:
    """Class to hold the results of a single kernel execution."""

    time: int
    flops: int
    bytes_accessed: int
    footprint_bytes: int


@dataclass
class MultiCallKernelProfile:
    """Class to hold the results of multiple kernel executions."""

    num_calls: int
    time: StatisticsAccumulator
    flops: StatisticsAccumulator
    bytes_accessed: StatisticsAccumulator
    footprint_bytes: StatisticsAccumulator


@dataclass
class ProfileEvent:
    """Holds a profile event that has not been collected by the profiler yet."""

    cl_event: cl._cl.Event
    translation_unit: lp.TranslationUnit
    args_tuple: tuple


class PyOpenCLProfilingArrayContext(PyOpenCLArrayContext):
    """An array context that profiles OpenCL kernel executions.

    .. automethod:: tabulate_profiling_data
    .. automethod:: call_loopy
    .. automethod:: get_profiling_data_for_kernel
    .. automethod:: reset_profiling_data_for_kernel

    Inherits from :class:`arraycontext.PyOpenCLArrayContext`.

    .. note::

       Profiling of :mod:`pyopencl` kernels (that is, kernels that do not get
       called through :meth:`call_loopy`) is restricted to a single instance of
       this class. If there are multiple instances, only the first one created
       will be able to profile these kernels.
    """

    def __init__(self, queue, allocator=None, logmgr: LogManager = None) -> None:
        super().__init__(queue, allocator)

        if not queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            raise RuntimeError("Profiling was not enabled in the command queue. "
                 "Please create the queue with "
                 "cl.command_queue_properties.PROFILING_ENABLE.")

        # list of ProfileEvents that haven't been transferred to profiled results yet
        self.profile_events = []

        # dict of kernel name -> SingleCallKernelProfile results
        self.profile_results = {}

        # dict of (Kernel, args_tuple) -> calculated number of flops, bytes
        self.kernel_stats = {}
        self.logmgr = logmgr

        # Only store the first kernel exec hook for elwise kernels
        if cl.array.ARRAY_KERNEL_EXEC_HOOK is None:
            cl.array.ARRAY_KERNEL_EXEC_HOOK = self.array_kernel_exec_hook

    def clone(self):
        """Return a semantically equivalent but distinct version of *self*."""
        from warnings import warn
        warn("Cloned PyOpenCLProfilingArrayContexts can not "
             "profile elementwise PyOpenCL kernels.")
        return type(self)(self.queue, self.allocator, self.logmgr)

    def __del__(self):
        """Release resources and undo monkey patching."""
        del self.profile_events[:]
        self.profile_results.clear()
        self.kernel_stats.clear()

    def array_kernel_exec_hook(self, knl, queue, gs, ls, *actual_args, wait_for):
        """Extract data from the elementwise array kernel."""
        evt = knl(queue, gs, ls, *actual_args, wait_for=wait_for)

        name = knl.function_name

        args_tuple = tuple(
            (arg.size)
            for arg in actual_args if isinstance(arg, cl.array.Array))

        try:
            self.kernel_stats[knl][args_tuple]
        except KeyError:
            nbytes = 0
            nops = 0
            for arg in actual_args:
                if isinstance(arg, cl.array.Array):
                    nbytes += arg.size * arg.dtype.itemsize
                    nops += arg.size
            res = SingleCallKernelProfile(time=0, flops=nops, bytes_accessed=nbytes,
                                footprint_bytes=nbytes)
            self.kernel_stats.setdefault(knl, {})[args_tuple] = res

        if self.logmgr and f"{name}_time" not in self.logmgr.quantity_data:
            self.logmgr.add_quantity(KernelProfile(self, name))

        self.profile_events.append(ProfileEvent(evt, knl, args_tuple))

        return evt

    def _wait_and_transfer_profile_events(self) -> None:
        # First, wait for completion of all events
        if self.profile_events:
            cl.wait_for_events([pevt.cl_event for pevt in self.profile_events])

        # Then, collect all events and store them
        for t in self.profile_events:
            t_unit = t.translation_unit
            if isinstance(t_unit, lp.TranslationUnit):
                name = t_unit.default_entrypoint.name
            else:
                # It's actually a cl.Kernel
                name = t_unit.function_name

            r = self._get_kernel_stats(t_unit, t.args_tuple)
            time = t.cl_event.profile.end - t.cl_event.profile.start

            new = SingleCallKernelProfile(time, r.flops, r.bytes_accessed,
                                          r.footprint_bytes)

            self.profile_results.setdefault(name, []).append(new)

        self.profile_events = []

    def get_profiling_data_for_kernel(self, kernel_name: str) \
          -> MultiCallKernelProfile:
        """Return profiling data for kernel `kernel_name`."""
        self._wait_and_transfer_profile_events()

        time = StatisticsAccumulator(scale_factor=1e-9)
        gflops = StatisticsAccumulator(scale_factor=1e-9)
        gbytes_accessed = StatisticsAccumulator(scale_factor=1e-9)
        fprint_gbytes = StatisticsAccumulator(scale_factor=1e-9)
        num_calls = 0

        if kernel_name in self.profile_results:
            knl_results = self.profile_results[kernel_name]

            num_calls = len(knl_results)

            for r in knl_results:
                time.add_value(r.time)
                gflops.add_value(r.flops)
                gbytes_accessed.add_value(r.bytes_accessed)
                fprint_gbytes.add_value(r.footprint_bytes)

        return MultiCallKernelProfile(num_calls, time, gflops, gbytes_accessed,
                                      fprint_gbytes)

    def reset_profiling_data_for_kernel(self, kernel_name: str) -> None:
        """Reset profiling data for kernel `kernel_name`."""
        self.profile_results.pop(kernel_name, None)

    def tabulate_profiling_data(self) -> pytools.Table:
        """Return a :class:`pytools.Table` with the profiling results."""
        self._wait_and_transfer_profile_events()

        tbl = pytools.Table()

        # Table header
        tbl.add_row(["Function", "Calls",
            "Time_sum [s]", "Time_min [s]", "Time_avg [s]", "Time_max [s]",
            "GFlops/s_min", "GFlops/s_avg", "GFlops/s_max",
            "BWAcc_min [GByte/s]", "BWAcc_mean [GByte/s]", "BWAcc_max [GByte/s]",
            "BWFoot_min [GByte/s]", "BWFoot_mean [GByte/s]", "BWFoot_max [GByte/s]",
            "Intensity (flops/byte)"])

        # Precision of results
        g = ".4g"

        total_calls = 0
        total_time = 0

        for knl in self.profile_results.keys():
            r = self.get_profiling_data_for_kernel(knl)

            # Extra statistics that are derived from the main values returned by
            # self.get_profiling_data_for_kernel(). These are already GFlops/s and
            # GBytes/s respectively, so no need to scale them.
            flops_per_sec = StatisticsAccumulator()
            bandwidth_access = StatisticsAccumulator()

            knl_results = self.profile_results[knl]
            for knl_res in knl_results:
                flops_per_sec.add_value(knl_res.flops/knl_res.time)
                bandwidth_access.add_value(knl_res.bytes_accessed/knl_res.time)

            total_calls += r.num_calls

            total_time += r.time.sum()

            time_sum = f"{r.time.sum():{g}}"
            time_min = f"{r.time.min():{g}}"
            time_avg = f"{r.time.mean():{g}}"
            time_max = f"{r.time.max():{g}}"

            if r.footprint_bytes.sum() is not None:
                fprint_mean = f"{r.footprint_bytes.mean():{g}}"
                fprint_min = f"{r.footprint_bytes.min():{g}}"
                fprint_max = f"{r.footprint_bytes.max():{g}}"
            else:
                fprint_mean = "--"
                fprint_min = "--"
                fprint_max = "--"

            if r.flops.sum() > 0:
                bytes_per_flop_mean = f"{r.bytes_accessed.sum() / r.flops.sum():{g}}"
                flops_per_sec_min = f"{flops_per_sec.min():{g}}"
                flops_per_sec_mean = f"{flops_per_sec.mean():{g}}"
                flops_per_sec_max = f"{flops_per_sec.max():{g}}"
            else:
                bytes_per_flop_mean = "--"
                flops_per_sec_min = "--"
                flops_per_sec_mean = "--"
                flops_per_sec_max = "--"

            bandwidth_access_min = f"{bandwidth_access.min():{g}}"
            bandwidth_access_mean = f"{bandwidth_access.sum():{g}}"
            bandwidth_access_max = f"{bandwidth_access.max():{g}}"

            tbl.add_row([knl, r.num_calls, time_sum,
                time_min, time_avg, time_max,
                flops_per_sec_min, flops_per_sec_mean, flops_per_sec_max,
                bandwidth_access_min, bandwidth_access_mean, bandwidth_access_max,
                fprint_min, fprint_mean, fprint_max,
                bytes_per_flop_mean])

        tbl.add_row(["Total", total_calls, f"{total_time:{g}}"] + ["--"] * 13)

        return tbl

    def _get_kernel_stats(self, t_unit: lp.TranslationUnit, args_tuple: tuple) \
      -> SingleCallKernelProfile:
        return self.kernel_stats[t_unit][args_tuple]

    def _cache_kernel_stats(self, t_unit: lp.TranslationUnit, kwargs: dict) \
      -> tuple:
        """Generate the kernel stats for a program with its args."""
        args_tuple = tuple(
            (key, value.shape) if hasattr(value, "shape") else (key, value)
            for key, value in kwargs.items())

        # Are kernel stats already in the cache?
        try:
            self.kernel_stats[t_unit][args_tuple]
            return args_tuple
        except KeyError:
            # If not, calculate and cache the stats
            ep_name = t_unit.default_entrypoint.name
            executor = t_unit.target.get_kernel_executor(t_unit, self.queue,
                    entrypoint=ep_name)
            info = executor.translation_unit_info(
                ep_name, executor.arg_to_dtype_set(kwargs))

            typed_t_unit = executor.get_typed_and_scheduled_translation_unit(
                ep_name, executor.arg_to_dtype_set(kwargs))
            kernel = typed_t_unit[ep_name]

            idi = info.implemented_data_info

            param_dict = kwargs.copy()
            param_dict.update({k: None for k in kernel.arg_dict.keys()
                if k not in param_dict})

            param_dict.update(
                {d.name: None for d in idi if d.name not in param_dict})

            # Generate the wrapper code
            wrapper = executor.get_wrapper_generator()

            gen = PythonFunctionGenerator("_mcom_gen_args_profile", list(param_dict))

            wrapper.generate_integer_arg_finding_from_shapes(gen, kernel, idi)
            wrapper.generate_integer_arg_finding_from_offsets(gen, kernel, idi)
            wrapper.generate_integer_arg_finding_from_strides(gen, kernel, idi)

            param_names = kernel.all_params()
            gen("return {%s}" % ", ".join(
                f"{repr(name)}: {name}" for name in param_names))

            # Run the wrapper code, save argument values in domain_params
            domain_params = gen.get_picklable_function()(**param_dict)

            # Get flops/memory statistics
            op_map = lp.get_op_map(typed_t_unit, subgroup_size="guess")
            bytes_accessed = lp.get_mem_access_map(
                typed_t_unit, subgroup_size="guess") \
                            .to_bytes().eval_and_sum(domain_params)

            flops = op_map.filter_by(dtype=[np.float32, np.float64]).eval_and_sum(
                domain_params)

            # Footprint gathering is not yet available in loopy with
            # kernel callables:
            # https://github.com/inducer/loopy/issues/399
            if 0:
                try:
                    footprint = lp.gather_access_footprint_bytes(typed_t_unit)
                    footprint_bytes = sum(footprint[k].eval_with_dict(domain_params)
                        for k in footprint)

                except lp.symbolic.UnableToDetermineAccessRange:
                    footprint_bytes = None
            else:
                footprint_bytes = None

            res = SingleCallKernelProfile(
                time=0, flops=flops, bytes_accessed=bytes_accessed,
                footprint_bytes=footprint_bytes)

            self.kernel_stats.setdefault(t_unit, {})[args_tuple] = res

            if self.logmgr:
                if f"{ep_name}_time" not in self.logmgr.quantity_data:
                    self.logmgr.add_quantity(KernelProfile(self, ep_name))

            return args_tuple

    def call_loopy(self, t_unit, **kwargs) -> dict:
        """Execute the loopy kernel and profile it."""
        try:
            t_unit = self._loopy_transform_cache[t_unit]
        except KeyError:
            orig_t_unit = t_unit
            t_unit = self.transform_loopy_program(t_unit)
            self._loopy_transform_cache[orig_t_unit] = t_unit
            del orig_t_unit

        evt, result = t_unit(self.queue, **kwargs, allocator=self.allocator)

        if self._wait_event_queue_length is not False:
            prg_name = t_unit.default_entrypoint.name
            wait_event_queue = self._kernel_name_to_wait_event_queue.setdefault(
                prg_name, [])

            wait_event_queue.append(evt)
            if len(wait_event_queue) > self._wait_event_queue_length:
                wait_event_queue.pop(0).wait()

        # Generate the stats here so we don't need to carry around the kwargs
        args_tuple = self._cache_kernel_stats(t_unit, kwargs)

        self.profile_events.append(ProfileEvent(evt, t_unit, args_tuple))

        return result
