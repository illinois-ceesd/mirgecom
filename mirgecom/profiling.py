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
from pytools.py_codegen import PythonCodeGenerator
import loopy as lp
import numpy as np
from dataclasses import dataclass

__doc__ = """
.. autoclass:: PyOpenCLProfilingArrayContext
"""


@dataclass
class ProfileEvent:
    """Class to hold a profile event that (potentially) has not completed yet."""
    cl_event: cl._cl.Event
    program: lp.kernel.LoopKernel
    args_tuple: tuple


class PyOpenCLProfilingArrayContext(PyOpenCLArrayContext):
    """An array context that profiles kernel executions.

    .. automethod:: finish_profile_events
    .. automethod:: print_profiling_data
    .. automethod:: call_loopy
    """

    def __init__(self, queue, allocator=None) -> None:
        super().__init__(queue, allocator)

        if not queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            raise RuntimeError("Profiling was not enabled in the command queue."
                 "Please create the queue with "
                 "cl.command_queue_properties.PROFILING_ENABLE.")

        self.profile_events = []
        self.profile_results = {}
        self.kernel_stats = {}

    def finish_profile_events(self) -> None:
        # First, wait for event completion
        if self.profile_events:
            cl.wait_for_events([t.cl_event for t in self.profile_events])

        for t in self.profile_events:
            program = t.program
            (flops, bytes_accessed) = self.kernel_stats[program][t.args_tuple]
            time = t.cl_event.profile.end - t.cl_event.profile.start

            self.profile_results.setdefault(program, []).append(
                dict(time=time, flops=flops, bytes_accessed=bytes_accessed))

        self.profile_events = []

    def print_profiling_data(self) -> None:
        self.finish_profile_events()

        import pytools

        tbl = pytools.Table()

        tbl.add_row(['Function', 'Calls', 'T_min', 'T_avg', 'T_max',
            'F_min', 'F_avg', 'F_max', 'M_min', 'M_avg', 'M_max', 'BW_avg'])

        from statistics import mean

        for key, value in self.profile_results.items():
            num_values = len(value)

            times = [v['time'] for v in value]
            flops = [v['flops'] for v in value]
            bytes_accessed = [v['bytes_accessed'] for v in value]

            tbl.add_row([key.name, num_values, min(times),
                int(mean(times)), max(times), min(flops), int(mean(flops)),
                max(flops), min(bytes_accessed), int(mean(bytes_accessed)),
                max(bytes_accessed), round(mean(bytes_accessed)/mean(times), 3)])

        if hasattr(pytools.Table, 'github_markdown'):
            print(tbl.github_markdown())
        else:
            print(tbl)

    def call_loopy(self, program, **kwargs) -> dict:
        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        args_tuple = tuple(
            (key, value.shape) if hasattr(value, 'shape') else (key, value)
            for key, value in kwargs.items())

        # Check if we need to get the invoker code to generate integer arguments
        # N.B.: The invoker code might be different for the same program with
        # different kwargs
        if (program not in self.kernel_stats
                or args_tuple not in self.kernel_stats[program]):
            executor = program.target.get_kernel_executor(program, self.queue)
            info = executor.kernel_info(executor.arg_to_dtype_set(kwargs))

            kernel = executor.get_typed_and_scheduled_kernel(
                executor.arg_to_dtype_set(kwargs))

            data = info.implemented_data_info

            # Generate the wrapper code
            wrapper = executor.get_wrapper_generator()

            gen = PythonCodeGenerator()

            wrapper.generate_integer_arg_finding_from_shapes(gen, kernel, data)
            wrapper.generate_integer_arg_finding_from_offsets(gen, kernel, data)
            wrapper.generate_integer_arg_finding_from_strides(gen, kernel, data)

            types = {}
            param_dict = {}

            for key, value in kwargs.items():
                if hasattr(value, 'dtype') and not value.dtype == object:
                    types[key] = value.dtype
                param_dict[key] = value

            for key, value in kernel.arg_dict.items():
                if key not in param_dict:
                    param_dict[key] = None

            for d in data:
                if d.name not in param_dict:
                    param_dict[d.name] = None

            # Run the wrapper code, save argument values in param_dict
            exec(gen.get(), param_dict)

            # Get flops/memory statistics
            kernel = lp.add_and_infer_dtypes(kernel, types)
            op_map = lp.get_op_map(kernel, subgroup_size='guess')
            bytes_accessed = lp.get_mem_access_map(kernel, subgroup_size='guess') \
              .to_bytes().eval_and_sum(param_dict)

            flops = op_map.filter_by(dtype=[np.float32, np.float64]).eval_and_sum(
                param_dict)

            self.kernel_stats.setdefault(program, {})[args_tuple] = (
                flops, bytes_accessed)

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        self.profile_events.append(ProfileEvent(evt, program, args_tuple))

        return result
