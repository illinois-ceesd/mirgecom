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
from logpyle import LogManager
from mirgecom.logging_quantities import KernelProfile
from statistics import mean

__doc__ = """
.. autoclass:: PyOpenCLProfilingArrayContext
.. autofunction:: add_nonloopy_profiling_result
"""

# {{{ Support for non-Loopy results (e.g., pyopencl kernels)

nonloopy_profile_results = {}


def pyopencl_monkey_del(self):
    """Monkey patches pyopencl.array destructor to grab profile data."""
    if self.events and self.queue and \
          self.queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
        cl.wait_for_events(self.events)
        times = []
        for evt in self.events:
            time = evt.profile.end - evt.profile.start
            times.append(time)

        add_nonloopy_profiling_result("pyopencl_array", mean(times))

        del self.events[:]


cl.array.Array.__del__ = pyopencl_monkey_del


@dataclass(eq=True, frozen=True)
class NonLoopyProfilekernel:
    """Class to hold the name for a non-loopy profile result.

    This is necessary so that :meth:`tabulate_profiling_data` etc. can
    access the 'name' field of the non-Loopy kernel.
    """

    name: str


def add_nonloopy_profiling_result(name: str, time: float, flops: float = None,
                bytes_accessed: float = None, footprint_bytes: float = None) -> None:
    """Add a non-loopy profile result to the profiling framework."""
    knl = NonLoopyProfilekernel(name)
    new = ProfileResult(time, flops, bytes_accessed, footprint_bytes)
    global nonloopy_profile_results
    nonloopy_profile_results.setdefault(knl, []).append(new)

# }}}


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
    .. automethod:: get_kernel_names
    .. automethod:: get_profiling_data_for_kernel

    Inherits from :class:`meshmode.array_context.PyOpenCLArrayContext`.
    """

    def __init__(self, queue, allocator=None, logmgr: LogManager = None) -> None:
        super().__init__(queue, allocator)

        if not queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            raise RuntimeError("Profiling was not enabled in the command queue. "
                 "Please create the queue with "
                 "cl.command_queue_properties.PROFILING_ENABLE.")

        self.profile_events = []
        self.profile_results = {}
        self.kernel_stats = {}
        self.logmgr = logmgr

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

    def get_kernel_names(self, wait_for_events=True) -> list:
        """Return a list of all kernel names."""
        if wait_for_events:
            self._finish_profile_events()

        res = []

        for key, value in self.profile_results.items():
            res += key.name

        return res

    def get_profiling_data_for_kernel(self, kernel_name: str,
                                      wait_for_events=True) -> float:
        """Return value of profiling result for kernel `kernel_name`."""
        if wait_for_events:
            self._finish_profile_events()

        def _gather_data(results_list: list):
            results = [key for key in results_list if key.name == kernel_name]
            num_calls = 0
            times = []
            flops = []
            bytes_accessed = []
            # fprint_bytes = []

            for key in results:
                value = results_list[key]

                num_calls += len(value)

                times += [v.time / 1e9 for v in value]
                flops += [v.flops / 1e9 if v.flops is not None else 0
                         for v in value]

                bytes_accessed += [v.bytes_accessed / 1e9
                              if v.bytes_accessed is not None else 0 for v in value]
                # fprint_bytes += np.ma.masked_equal(
                #   [v.footprint_bytes for v in value], None)

                del results_list[key]

            return num_calls, times, flops, bytes_accessed

        num_calls, times, flops, bytes_accessed = \
            _gather_data(self.profile_results)

        if num_calls == 0:
            num_calls, times, flops, bytes_accessed = \
                _gather_data(nonloopy_profile_results)

        if num_calls == 0:
            return [0, 0, 0, 0]

        return [mean(times), mean(flops), num_calls, mean(bytes_accessed)]

    def tabulate_profiling_data(self, wait_for_events=True) -> pytools.Table:
        """Return a :class:`pytools.Table` with the profiling results."""
        if wait_for_events:
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

        all_res = {**self.profile_results, **nonloopy_profile_results}

        for key, value in all_res.items():
            num_values = len(value)

            times = [v.time / 1e9 for v in value]

            flops = [v.flops / 1e9 if v.flops is not None and v.flops > 0 else None
                     for v in value]
            flops_per_sec = [f / t if f is not None else None
                             for f, t in zip(flops, times)]

            bytes_accessed = [v.bytes_accessed / 1e9
                            if v.bytes_accessed is not None else None for v in value]
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

            tbl.add_row([key.name, num_values,
                f"{min(times):{g}}", f"{mean(times):{g}}", f"{max(times):{g}}",
                flops_per_sec_min, flops_per_sec_mean, flops_per_sec_max,
                bandwidth_access_min, bandwidth_access_mean, bandwidth_access_max,
                fprint_min, f"{fprint_mean:{g}}", fprint_max,
                bytes_per_flop_mean])

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

            if self.logmgr:
                if "pyopencl_array_time" not in self.logmgr.quantity_data:
                    self.logmgr.add_quantity(
                        KernelProfile(self, "pyopencl_array"))

                # Since kernel names are not unique, find the next free ID
                # to append to the logging name
                i = 0
                while True:
                    if (f"{program.name}_{i}_time" not in
                            self.logmgr.quantity_data):
                        name = f"{program.name}_{i}"
                        break
                    i += 1
                self.logmgr.add_quantity(
                    KernelProfile(self, program.name, name))

            return res

    def call_loopy(self, program, **kwargs) -> dict:
        """Execute the loopy kernel."""
        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        self.profile_events.append(ProfileEvent(evt, program, kwargs))

        if self.logmgr and not self.kernel_stats:
            # Call _get_kernel_stats() once to add
            # profiling quantities to the logmgr
            self._get_kernel_stats(program, kwargs)

        return result
