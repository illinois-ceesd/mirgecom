"""An array context with profiling capabilities."""

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

__doc__ = """
.. autoclass:: PyOpenCLProfilingArrayContext
"""


# {{{ Support for non-Loopy results (e.g., pyopencl kernels)

nonloopy_profile_events = []


def init_pyopencl_array_monkey_patch():
    """Add the :mod:`pyopencl` monkey patching."""
    cl.array.ARRAY_KERNEL_EXEC_HOOK = array_kernel_exec_hook


def del_pyopencl_array_monkey_patch():
    """Remove the :mod:`pyopencl` monkey patching."""
    cl.array.ARRAY_KERNEL_EXEC_HOOK = None


@dataclass(eq=True, frozen=True)
class NonLoopyProfilekernel:
    """Class to hold the name for a non-loopy profile result.

    This is necessary so that :meth:`tabulate_profiling_data` and
    :meth:`get_profiling_data_for_kernel` can
    access the 'name' field of the non-Loopy kernel.
    """

    name: str


@dataclass
class ProfileResult:
    """Class to hold the results of a single kernel execution."""

    time: int
    flops: int
    bytes_accessed: int
    footprint_bytes: int


# For pyopencl.Array's elementwise kernels.
elwise_knl = NonLoopyProfilekernel("pyopencl_array")


def array_kernel_exec_hook(knl, queue, gs, ls, *actual_args, wait_for):
    """Initialize the :mod:`pyopencl` monkey patching."""
    evt = knl(queue, gs, ls, *actual_args, wait_for=wait_for)
    nonloopy_profile_events.append(evt)

    return evt

# }}}


@dataclass
class ProfileEvent:
    """Class to hold a profile event that has not been seen by the profiler yet."""

    cl_event: cl._cl.Event
    program: lp.kernel.LoopKernel
    args_tuple: tuple


class PyOpenCLProfilingArrayContext(PyOpenCLArrayContext):
    """An array context that profiles kernel executions.

    .. automethod:: tabulate_profiling_data
    .. automethod:: call_loopy

    Inherits from :class:`meshmode.array_context.PyOpenCLArrayContext`.
    """

    def __init__(self, queue, allocator=None) -> None:
        super().__init__(queue, allocator)

        if not queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            raise RuntimeError("Profiling was not enabled in the command queue. "
                 "Please create the queue with "
                 "cl.command_queue_properties.PROFILING_ENABLE.")

        self.profile_events = []
        self.profile_results = {}
        self.kernel_stats = {}

        init_pyopencl_array_monkey_patch()

    def __del__(self):
        """Release resources and undo monkey patching."""
        del self.profile_events[:]
        self.profile_results.clear()
        self.kernel_stats.clear()

        del_pyopencl_array_monkey_patch()

    def _finish_profile_events(self) -> None:
        # First, wait for completion of all events
        if self.profile_events:
            cl.wait_for_events([pevt.cl_event for pevt in self.profile_events])

        global nonloopy_profile_events
        if nonloopy_profile_events:
            cl.wait_for_events(nonloopy_profile_events)

        # Then, collect all events and store them
        for t in self.profile_events:
            program = t.program
            r = self._get_kernel_stats(program, t.args_tuple)
            time = t.cl_event.profile.end - t.cl_event.profile.start

            new = ProfileResult(time, r.flops, r.bytes_accessed, r.footprint_bytes)

            self.profile_results.setdefault(program, []).append(new)

        for t in nonloopy_profile_events:
            program = elwise_knl
            time = t.profile.end - t.profile.start
            new = ProfileResult(time, None, None, None)
            self.profile_results.setdefault(program, []).append(new)

        self.profile_events = []
        nonloopy_profile_events = []

    def tabulate_profiling_data(self) -> pytools.Table:
        """Return a :class:`pytools.Table` with the profiling results."""
        self._finish_profile_events()

        tbl = pytools.Table()

        tbl.add_row(["Function", "Calls",
            "Time_sum [s]", "Time_min [s]", "Time_avg [s]", "Time_max [s]",
            "GFlops/s_min", "GFlops/s_avg", "GFlops/s_max",
            "BWAcc_min [GByte/s]", "BWAcc_mean [GByte/s]", "BWAcc_max [GByte/s]",
            "BWFoot_min [GByte/s]", "BWFoot_mean [GByte/s]", "BWFoot_max [GByte/s]",
            "Intensity (flops/byte)"])

        # Precision of results
        g = ".4g"

        from statistics import mean

        total_calls = 0
        total_time = 0

        for key, value in self.profile_results.items():
            num_values = len(value)

            total_calls += num_values

            times = [v.time / 1e9 for v in value]

            total_time += sum(times)

            flops = [v.flops / 1e9 if v.flops is not None and v.flops > 0 else None
                     for v in value]
            flops_per_sec = [f / t if f is not None else None
                              for f, t in zip(flops, times)]

            bytes_accessed = [v.bytes_accessed / 1e9
                             if v.bytes_accessed is not None else None
                             for v in value]
            bandwidth_access = [b / t if b is not None else None
                                 for b, t in zip(bytes_accessed, times)]

            fprint_bytes = np.ma.masked_equal([v.footprint_bytes for v in value],
                None)
            fprint_mean = np.mean(fprint_bytes) / 1e9

            # pylint: disable=E1101
            if len(fprint_bytes.compressed()) > 0:
                fprint_min = f"{np.min(fprint_bytes.compressed() / 1e9):{g}}"
                fprint_max = f"{np.max(fprint_bytes.compressed() / 1e9):{g}}"
            else:
                fprint_min = "--"
                fprint_max = "--"

            bytes_per_flop = [f / b if b is not None and f is not None and b > 0
                               else None
                               for f, b in zip(flops, bytes_accessed)]
            bytes_per_flop_mean = f"{mean(bytes_per_flop):{g}}" \
                                    if None not in bytes_per_flop else "--"

            flops_per_sec_min = f"{min(flops_per_sec):{g}}" \
                                  if None not in flops_per_sec else "--"
            flops_per_sec_mean = f"{mean(flops_per_sec):{g}}" \
                                   if None not in flops_per_sec else "--"
            flops_per_sec_max = f"{max(flops_per_sec):{g}}" \
                                  if None not in flops_per_sec else "--"

            bandwidth_access_min = f"{min(bandwidth_access):{g}}" \
                                     if None not in bandwidth_access else "--"
            bandwidth_access_mean = f"{mean(bandwidth_access):{g}}" \
                                      if None not in bandwidth_access else "--"
            bandwidth_access_max = f"{max(bandwidth_access):{g}}" \
                                     if None not in bandwidth_access else "--"

            tbl.add_row([key.name, num_values, f"{sum(times):{g}}",
                f"{min(times):{g}}", f"{mean(times):{g}}", f"{max(times):{g}}",
                flops_per_sec_min, flops_per_sec_mean, flops_per_sec_max,
                bandwidth_access_min, bandwidth_access_mean, bandwidth_access_max,
                fprint_min, f"{fprint_mean:{g}}", fprint_max,
                bytes_per_flop_mean])

        tbl.add_row(["Total", total_calls, f"{total_time:{g}}"] + ["--"] * 13)

        return tbl

    def _get_kernel_stats(self, program: lp.kernel.LoopKernel, args_tuple: tuple) \
      -> ProfileResult:
        return self.kernel_stats[program][args_tuple]

    def _cache_kernel_stats(self, program: lp.kernel.LoopKernel, kwargs: dict) \
      -> tuple:
        """Generate the kernel stats for a program with its args."""
        args_tuple = tuple(
            (key, value.shape) if hasattr(value, "shape") else (key, value)
            for key, value in kwargs.items())

        # Are kernel stats already in the cache?
        try:
            x = self.kernel_stats[program][args_tuple]  # noqa
            return args_tuple
        except KeyError:
            # If not, calculate and cache the stats
            executor = program.target.get_kernel_executor(program, self.queue)
            info = executor.kernel_info(executor.arg_to_dtype_set(kwargs))

            kernel = executor.get_typed_and_scheduled_kernel(
                executor.arg_to_dtype_set(kwargs))

            idi = info.implemented_data_info

            types = {k: v for k, v in kwargs.items()
                if hasattr(v, "dtype") and not v.dtype == object}

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

            param_names = program.all_params()
            gen("return {%s}" % ", ".join(
                f"{repr(name)}: {name}" for name in param_names))

            # Run the wrapper code, save argument values in domain_params
            domain_params = gen.get_picklable_function()(**param_dict)

            # Get flops/memory statistics
            kernel = lp.add_and_infer_dtypes(kernel, types)
            op_map = lp.get_op_map(kernel, subgroup_size="guess")
            bytes_accessed = lp.get_mem_access_map(kernel, subgroup_size="guess") \
              .to_bytes().eval_and_sum(domain_params)

            flops = op_map.filter_by(dtype=[np.float32, np.float64]).eval_and_sum(
                domain_params)

            try:
                footprint = lp.gather_access_footprint_bytes(kernel)
                footprint_bytes = sum(footprint[k].eval_with_dict(domain_params)
                    for k in footprint)

            except lp.symbolic.UnableToDetermineAccessRange:
                footprint_bytes = None

            res = ProfileResult(
                time=0, flops=flops, bytes_accessed=bytes_accessed,
                footprint_bytes=footprint_bytes)

            self.kernel_stats.setdefault(program, {})[args_tuple] = res
            return args_tuple

    def call_loopy(self, program, **kwargs) -> dict:
        """Execute the loopy kernel."""
        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        # Generate the stats here so we don't need to carry around the kwargs
        args_tuple = self._cache_kernel_stats(program, kwargs)

        self.profile_events.append(ProfileEvent(evt, program, args_tuple))

        return result
