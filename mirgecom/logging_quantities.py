from logpyle import LogQuantity


class PhysicalQuantity(LogQuantity):
    def __init__(self, discr, eos, quantity, unit, op):
        LogQuantity.__init__(self, f"{op}_{quantity}", unit)

        self.discr = discr
        self.eos = eos
        self.quantity = quantity
        from functools import partial

        if op == "min":
            self._myop = partial(self.discr.nodal_min, "vol")
        elif op == "max":
            self._myop = partial(self.discr.nodal_max, "vol")
        else:
            raise RuntimeError("unknown operation {op}")

    def __call__(self):
        from mirgecom.steppers import get_current_state

        if get_current_state() is None:
            return 0

        from mirgecom.euler import split_conserved

        cv = split_conserved(self.discr.dim, get_current_state())
        dv = self.eos.dependent_vars(cv)

        return self._myop(getattr(dv, self.quantity))


class KernelProfile(LogQuantity):
    def __init__(self, actx, kernel_name, stat):
        if stat == "time":
            unit = "s"
        else:
            unit = ""
        LogQuantity.__init__(self, f"{kernel_name}_{stat}",
                             unit, f"{stat} of '{kernel_name}'")
        self.kernel_name = kernel_name
        self.actx = actx
        self.stat = stat

    def __call__(self):
        return(self.actx.get_profiling_data_for_kernel(self.kernel_name, self.stat))
