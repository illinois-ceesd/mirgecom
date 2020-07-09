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


class ProfileData:
    time = 0
    flops = 0
    mem_access = 0

    def __init__(self, time=0, flops=0, mem_access=0):
        self.time = time
        self.flops = flops
        self.mem_access = mem_access

    def __repr__(self):
        return "(time={0}, flops={1}, mem_access={2})".format(self.time, self.flops, self.mem_access)

    def __str__(self):
        return self.__repr__()


class TimingEvent:
    def __init__(self, event, program, kwargs):
        self.event = event
        self.program = program
        self.kwargs = kwargs


class PyOpenCLProfilingArrayContext(PyOpenCLArrayContext):

    def __init__(self, queue, allocator=None):
        super().__init__(queue, allocator)

        if not queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            from warnings import warn
            warn("Profiling was not enabled in the command queue. Timing data will not be collected.")
            self.profiling_enabled = False
        else:
            self.profiling_enabled = True

        self.events = []
        self.profiling_data = {}
        self.invoker_codes = {}

    def finish_profile_events(self):

        if not self.profiling_enabled:
            return

        if self.events:
            cl.wait_for_events([t.event for t in self.events])

        for t in self.events:

            kwargs = t.kwargs
            program = t.program
            invoker_code = self.invoker_codes[program.name][tuple(kwargs)]
            evt = t.event

            types = {}
            param_dict = {}

            for key, value in kwargs.items():
                types[key] = value.dtype
                param_dict[key] = value

            # extract integer argument generation code from wrapper
            code = ""
            import textwrap
            for o in ["shapes", "strides"]: #"offsets",
                subs = "# {{{ find integer arguments from " + o
                start=invoker_code.find(subs) + len(subs)
                end=invoker_code.find("# }}}", start)
                code = code + textwrap.dedent(invoker_code[start:end])


            for key, value in program.arg_dict.items():
                if key not in param_dict:
                    param_dict[key] = None

            # execute integer argument generation code from wrapper
            exec(code, param_dict)

            # get statistics
            program = lp.add_and_infer_dtypes(program, types)
            op_map = lp.get_op_map(program, subgroup_size='guess')
            mem_map = lp.get_mem_access_map(program, subgroup_size='guess')

            f32op_count = op_map.filter_by(dtype=[np.float32]).eval_and_sum(param_dict)
            f64op_count = op_map.filter_by(dtype=[np.float64]).eval_and_sum(param_dict)

            mem32_count = mem_map.filter_by(dtype=[np.float32]).eval_and_sum(param_dict)
            mem64_count = mem_map.filter_by(dtype=[np.float64]).eval_and_sum(param_dict)

            flops = f32op_count + f64op_count
            time = evt.profile.end - evt.profile.start
            mem_access = mem32_count + mem64_count

            if program.name in self.profiling_data:
                self.profiling_data[program.name].append(ProfileData(time, flops, mem_access))
            else:
                self.profiling_data[program.name] = [ProfileData(time, flops, mem_access)]

        self.events = []

    def print_profiling_data(self):

        if not self.profiling_enabled:
            return

        self.finish_profile_events()

        max_name_len = max([len(key) for key, value in self.profiling_data.items()])
        max_name_len = max(max_name_len, len('Function'))

        format_str = "{:<" + str(max_name_len) + "} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"

        print(format_str.format('='*20, '='*6, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8))
        print(format_str.format('Function', 'Calls', 'T_min', 'T_avg', 'T_max', 'F_min', 'F_avg', 'F_max', 'M_min', 'M_avg', 'M_max', 'BW_avg'))
        print(format_str.format('='*20, '='*6, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8))

        from statistics import mean

        for key, value in self.profiling_data.items():
            num_values = len(value)

            times = [v.time for v in value]
            flops = [v.flops for v in value]
            mem_access = [v.mem_access for v in value]

            print(format_str.format(key, num_values, min(times), int(mean(times)), max(times), min(flops), int(mean(flops)), max(flops), min(mem_access), int(mean(mem_access)), max(mem_access), round(mean(mem_access)/mean(times),3) ))

        print(format_str.format('='*20, '='*6, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8, '='*8))

    def __del__(self):
        self.print_profiling_data()

    def call_loopy(self, program, **kwargs):

        from warnings import resetwarnings, filterwarnings
        resetwarnings()
        filterwarnings('ignore', category=Warning)

        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        # Determine if we need to get the invoker code (for integer argument generation).
        # N.B.: The invoker code might be different for the same program with different kwargs
        if program.name not in self.invoker_codes or tuple(kwargs) not in self.invoker_codes[program.name]:
            executor=program.target.get_kernel_executor(program, self.queue)
            info = executor.kernel_info(executor.arg_to_dtype_set(kwargs))
            invoker_code = info.invoker.get()
            self.invoker_codes[program.name] = {}
            self.invoker_codes[program.name][tuple(kwargs)] = invoker_code

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        if self.profiling_enabled:
            self.events.append(TimingEvent(evt, program, kwargs))

        # self.print_profiling_data()

        return result
