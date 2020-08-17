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


class ProfileData:
    def __init__(self, time, flops, mem_access) -> None:
        self.time = time
        self.flops = flops
        self.mem_access = mem_access


class TimingEvent:
    def __init__(self, event, program, flops, bytes_accessed) -> None:
        self.event = event
        self.program = program
        self.flops = flops
        self.bytes_accessed = bytes_accessed


class PyOpenCLProfilingArrayContext(PyOpenCLArrayContext):
    def __init__(self, queue, allocator=None) -> None:
        super().__init__(queue, allocator)

        if not queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            from warnings import warn
            warn("Profiling was not enabled in the command queue."
                 "Timing data will not be collected.")
            self.profiling_enabled = False
        else:
            self.profiling_enabled = True

        self.events = []
        self.profiling_data = {}
        self.kernel_stats = {}

    def finish_profile_events(self) -> None:
        if not self.profiling_enabled:
            return

        if self.events:
            cl.wait_for_events([t.event for t in self.events])

        for t in self.events:
            program = t.program
            flops = t.flops
            bytes_accessed = t.bytes_accessed
            evt = t.event

            time = evt.profile.end - evt.profile.start

            if program in self.profiling_data:
                self.profiling_data[program].append(
                    ProfileData(time, flops, bytes_accessed))
            else:
                self.profiling_data[program] = [
                    ProfileData(time, flops, bytes_accessed)]

        self.events = []

    def print_profiling_data(self) -> None:
        if not self.profiling_enabled:
            return

        self.finish_profile_events()

        max_name_len = max(
            [len(key.name) for key, value in self.profiling_data.items()])
        max_name_len = max(max_name_len, len('Function'))

        format_str = ("{:<" + str(max_name_len)
            + "} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}")

        print(format_str.format('='*20, '='*6, '='*8, '='*8, '='*8, '='*8, '='*8,
            '='*8, '='*8, '='*8, '='*8, '='*8))
        print(format_str.format('Function', 'Calls', 'T_min', 'T_avg', 'T_max',
            'F_min', 'F_avg', 'F_max', 'M_min', 'M_avg', 'M_max', 'BW_avg'))
        print(format_str.format('='*20, '='*6, '='*8, '='*8, '='*8,
            '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8))

        from statistics import mean

        for key, value in self.profiling_data.items():
            num_values = len(value)

            times = [v.time for v in value]
            flops = [v.flops for v in value]
            bytes_accessed = [v.bytes_accessed for v in value]

            print(format_str.format(key.name, num_values, min(times),
                int(mean(times)), max(times), min(flops), int(mean(flops)),
                max(flops), min(bytes_accessed), int(mean(bytes_accessed)),
                max(bytes_accessed), round(mean(bytes_accessed)/mean(times), 3)))

        print(format_str.format('='*20, '='*6, '='*8, '='*8, '='*8, '='*8, '='*8,
            '='*8, '='*8, '='*8, '='*8, '='*8))

    def call_loopy(self, program, **kwargs) -> dict:
        from warnings import resetwarnings, filterwarnings
        resetwarnings()
        filterwarnings('ignore', category=Warning)

        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        if self.profiling_enabled:
            args_tuple = ()

            for key, value in kwargs.items():
                if hasattr(value, 'shape'):
                    args_tuple = args_tuple + (key, value.shape)
                else:
                    args_tuple = args_tuple + (key, value)

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
                kernel.options.ignore_boostable_into = True
                kernel = lp.add_and_infer_dtypes(kernel, types)
                op_map = lp.get_op_map(kernel, subgroup_size='guess')
                mem_map = lp.get_mem_access_map(kernel, subgroup_size='guess')

                f32op = op_map.filter_by(dtype=[np.float32]).eval_and_sum(param_dict)
                f64op = op_map.filter_by(dtype=[np.float64]).eval_and_sum(param_dict)

                mem32 = mem_map.filter_by(dtype=[np.float32]).eval_and_sum(
                    param_dict)
                mem64 = mem_map.filter_by(dtype=[np.float64]).eval_and_sum(
                    param_dict)

                flops = f32op + f64op
                bytes_accessed = mem32*4 + mem64*8

                if program not in self.kernel_stats:
                    self.kernel_stats[program] = {}

                self.kernel_stats[program][args_tuple] = (flops, bytes_accessed)

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        if self.profiling_enabled:
            flops = self.kernel_stats[program][args_tuple][0]
            bytes_accessed = self.kernel_stats[program][args_tuple][1]
            self.events.append(TimingEvent(evt, program, flops, bytes_accessed))

        return result
