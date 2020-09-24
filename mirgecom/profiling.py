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


@dataclass
class ProfileResult:
    """Class to hold the results of a single kernel execution."""

    time: int
    flops: int
    bytes_accessed: int
    footprint_bytes: int


@dataclass
class ProfileEvent:
    """Class to hold a profile event that has not been seen by the profiler yet."""

    cl_event: cl._cl.Event
    program: lp.kernel.LoopKernel
    kwargs: dict


class PyOpenCLProfilingArrayContext(PyOpenCLArrayContext):
    """An array context that profiles kernel executions.

    .. automethod:: clear_profiling_data
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

    def _finish_profile_events(self) -> None:
        # First, wait for completion of all events
        if self.profile_events:
            cl.wait_for_events([pevt.cl_event for pevt in self.profile_events])

        # Then, collect all events and store them
        for t in self.profile_events:
            program = t.program
            r = self._get_kernel_stats(program, t.kwargs)
            time = t.cl_event.profile.end - t.cl_event.profile.start

            new = ProfileResult(time, r.flops, r.bytes_accessed, r.footprint_bytes)

            self.profile_results.setdefault(program, []).append(new)

        self.profile_events = []

    def clear_profiling_data(self) -> None:
        """Clear all profiling data."""
        self._finish_profile_events()
        self.profile_results = {}

    def tabulate_profiling_data(self) -> pytools.Table:
        """Return a :class:`pytools.Table` with the profiling results."""
        self._finish_profile_events()

        tbl = pytools.Table()

        tbl.add_row(["Function", "Calls",
            "Time_min [s]", "Time_avg [s]", "Time_max [s]",
            "GFlops/s_min", "GFlops/s_avg", "GFlops/s_max",
            "BWAcc_min [GByte/s]", "BWAcc_mean [GByte/s]", "BWAcc_max [GByte/s]",
            "BWFoot_min [GByte/s]", "BWFoot_mean [GByte/s]", "BWFoot_max [GByte/s]",
            "Intensity (flops/byte)"])

        # Precision of results
        g = ".4g"

        from statistics import mean

        for key, value in self.profile_results.items():
            num_values = len(value)

            times = [v.time / 1e9 for v in value]

            flops = [v.flops / 1e9 for v in value]
            flops_per_sec = [f / t for f, t in zip(flops, times)]

            bytes_accessed = [v.bytes_accessed / 1e9 for v in value]
            bandwidth_access = [b / t for b, t in zip(bytes_accessed, times)]

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

            bytes_per_flop = [f / b for f, b in zip(flops, bytes_accessed)]

            tbl.add_row([key.name, num_values,
                f"{min(times):{g}}", f"{mean(times):{g}}", f"{max(times):{g}}",
                f"{min(flops_per_sec):{g}}", f"{mean(flops_per_sec):{g}}",
                f"{max(flops_per_sec):{g}}",
                f"{min(bandwidth_access):{g}}", f"{mean(bandwidth_access):{g}}",
                f"{max(bandwidth_access):{g}}",
                fprint_min, f"{fprint_mean:{g}}", fprint_max,
                f"{mean(bytes_per_flop):{g}}"])

        return tbl

    def _get_kernel_stats(self, program, kwargs: dict) -> ProfileResult:
        # We need a tuple to index the cache
        args_tuple = tuple(
            (key, value.shape) if hasattr(value, "shape") else (key, value)
            for key, value in kwargs.items())

        # Are kernel stats already in the cache?
        try:
            return self.kernel_stats[program][args_tuple]
        except KeyError:
            # If not, calculate and cache the stats
            kwargs = dict(kwargs)
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
            return res

    def call_loopy(self, program, **kwargs) -> dict:
        """Execute the loopy kernel."""
        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        self.profile_events.append(ProfileEvent(evt, program, kwargs))

        return result
