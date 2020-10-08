from logpyle import LogQuantity


def set_state(mgr, state):
    for gd_lst in [mgr.before_gather_descriptors,
            mgr.after_gather_descriptors]:
        for gd in gd_lst:
            if isinstance(gd.quantity, StateConsumer):
                gd.quantity.set_state(state)


class StateConsumer:
    def __init__(self, state):
        self.state = state

    def set_state(self, state) -> None:
        self.state = state


class PhysicalQuantity(LogQuantity, StateConsumer):
    def __init__(self, discr, eos, quantity, unit, op):
        LogQuantity.__init__(self, f"{op}_{quantity}", unit)
        StateConsumer.__init__(self, None)

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
        if self.state is None:
            return 0

        from mirgecom.euler import split_conserved

        cv = split_conserved(self.discr.dim, self.state)
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
