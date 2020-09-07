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

    .. automethod:: tabulate_profiling_data
    .. automethod:: call_loopy

    Inherits from :class:`meshmode.array_context.PyOpenCLArrayContext`."""

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
        # First, wait for event completion
        if self.profile_events:
            cl.wait_for_events([pevt.cl_event for pevt in self.profile_events])

        for t in self.profile_events:
            program = t.program
            res = self._get_kernel_stats(program, t.kwargs)
            res.time = t.cl_event.profile.end - t.cl_event.profile.start

            self.profile_results.setdefault(program, []).append(res)

        self.profile_events = []

    def tabulate_profiling_data(self) -> pytools.Table:
        """Returns a :class:`pytools.Table` with the profiling results."""
        self._finish_profile_events()

        tbl = pytools.Table()

        tbl.add_row(["Function", "Calls", "Time_min (s)", "Time_avg (s)",
            "Time_max (s)", "GFlops_min", "GFlops_avg", "GFlops_max", "GMemAcc_min",
            "GMemAcc_avg", "GMemAcc_max", "Bandwidth (GByte/s)",
            "Footprint (GByte)"])

        from statistics import mean

        for key, value in self.profile_results.items():
            num_values = len(value)

            times = [v.time / 1e9 for v in value]
            flops = [v.flops / 1e9 for v in value]
            bytes_accessed = [v.bytes_accessed / 1e9 for v in value]
            footprint_bytes = [v.footprint_bytes / 1e9
                if v.footprint_bytes is not None else 0 for v in value]

            tbl.add_row([key.name, num_values, min(times),
                mean(times), max(times), min(flops), mean(flops),
                max(flops), min(bytes_accessed), mean(bytes_accessed),
                max(bytes_accessed), round(mean(bytes_accessed)/mean(times), 3),
                mean(footprint_bytes)])

        return tbl

    def _get_kernel_stats(self, program, kwargs: dict) -> ProfileResult:
        # We need a tuple to index the cache
        args_tuple = tuple(
            (key, value.shape) if hasattr(value, "shape") else (key, value)
            for key, value in kwargs.items())

        # Are kernel stats already in the cache?
        if program in self.kernel_stats and args_tuple in self.kernel_stats[program]:
            return self.kernel_stats[program][args_tuple]

        # If not, calculate and cache the stats
        kwargs = dict(kwargs)
        executor = program.target.get_kernel_executor(program, self.queue)
        info = executor.kernel_info(executor.arg_to_dtype_set(kwargs))

        kernel = executor.get_typed_and_scheduled_kernel(
            executor.arg_to_dtype_set(kwargs))

        idi = info.implemented_data_info

        types = {k: v for k, v in kwargs.items()
            if hasattr(v, "dtype") and not v.dtype == object}

        param_dict = {**kwargs}
        param_dict.update({k: None for k, value in kernel.arg_dict.items()
            if k not in param_dict})

        param_dict.update(
            {d.name: None for d in idi if d.name not in param_dict})

        # Generate the wrapper code
        wrapper = executor.get_wrapper_generator()

        gen = PythonFunctionGenerator("_my_gen_args_profiling", ["param_dict"])

        # Unpack dict items to local variables
        for k, v in param_dict.items():
            gen(f"{k}=param_dict['{k}']")

        wrapper.generate_integer_arg_finding_from_shapes(gen, kernel, idi)
        wrapper.generate_integer_arg_finding_from_offsets(gen, kernel, idi)
        wrapper.generate_integer_arg_finding_from_strides(gen, kernel, idi)

        # Pack modified variables back into dict
        for k, v in param_dict.items():
            gen(f"param_dict['{k}']={k}")

        # Run the wrapper code, save argument values in param_dict
        gen.get_picklable_function()(param_dict)

        # Get flops/memory statistics
        kernel = lp.add_and_infer_dtypes(kernel, types)
        op_map = lp.get_op_map(kernel, subgroup_size="guess")
        bytes_accessed = lp.get_mem_access_map(kernel, subgroup_size="guess") \
          .to_bytes().eval_and_sum(param_dict)

        flops = op_map.filter_by(dtype=[np.float32, np.float64]).eval_and_sum(
            param_dict)

        footprint_bytes = 0
        try:
            footprint = lp.gather_access_footprint_bytes(kernel)
            for k, v in footprint.items():
                footprint_bytes += footprint[k].eval_with_dict(param_dict)

        except lp.symbolic.UnableToDetermineAccessRange:
            footprint_bytes = None

        res = ProfileResult(
            time=0, flops=flops, bytes_accessed=bytes_accessed,
            footprint_bytes=footprint_bytes)

        self.kernel_stats.setdefault(program, {})[args_tuple] = res
        return res

    def call_loopy(self, program, **kwargs) -> dict:
        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        self.profile_events.append(ProfileEvent(evt, program, kwargs))

        return result
