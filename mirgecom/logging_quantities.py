from logpyle import LogQuantity, MultiLogQuantity


class MinPressure(LogQuantity):
    def __init__(self, discr, eos):
        LogQuantity.__init__(self, "min_pressure", "P")

        self.discr = discr
        self.eos = eos
        from functools import partial
        self._min = partial(self.discr.nodal_min, "vol")

    def __call__(self):
        from mirgecom.steppers import get_current_state

        if get_current_state() is None:
            return 0

        from mirgecom.euler import split_conserved

        cv = split_conserved(self.discr.dim, get_current_state())
        dv = self.eos.dependent_vars(cv)

        return self._min(dv.pressure)


class MinTemperature(LogQuantity):
    def __init__(self, discr, eos):
        LogQuantity.__init__(self, "min_temperature", "K")

        self.discr = discr
        self.eos = eos

        from functools import partial
        self._min = partial(self.discr.nodal_min, "vol")

    def __call__(self):
        from mirgecom.steppers import get_current_state

        if get_current_state() is None:
            return 0

        from mirgecom.euler import split_conserved

        cv = split_conserved(self.discr.dim, get_current_state())
        dv = self.eos.dependent_vars(cv)

        return self._min(dv.temperature)


class KernelProfile(LogQuantity):
    def __init__(self, actx, kernel_name):
        LogQuantity.__init__(self, kernel_name, "s")
        self.kernel_name = kernel_name
        self.actx = actx

    def __call__(self):
        return(self.actx.get_profiling_data_for_kernel(self.kernel_name))
