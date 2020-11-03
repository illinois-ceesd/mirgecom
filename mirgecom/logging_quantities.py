"""Support for time series logging."""

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

__doc__ = """
.. autoclass:: StateConsumer
.. autoclass:: PhysicalQuantity
.. autoclass:: KernelProfile
.. autofunction:: set_state
"""

from logpyle import LogQuantity
from numpy import ndarray
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from meshmode.discretization import Discretization
from mirgecom.eos import GasEOS


def set_state(mgr, state: ndarray) -> None:
    """Update the state of all :class:`StateConsumer` of the log manager `mgr`."""
    for gd_lst in [mgr.before_gather_descriptors,
            mgr.after_gather_descriptors]:
        for gd in gd_lst:
            if isinstance(gd.quantity, StateConsumer):
                gd.quantity.set_state(state)


class StateConsumer:
    """Base class for quantities that require a state for logging."""

    def __init__(self):
        self.state = None

    def set_state(self, state: ndarray) -> None:
        """Set the state of the object."""
        self.state = state


class PhysicalQuantity(LogQuantity, StateConsumer):
    """Logging support for physical quantities."""

    def __init__(self, discr: Discretization, quantity: str, unit: str, op: str,
                 name=None):
        if name:
            LogQuantity.__init__(self, name, unit)
        else:
            LogQuantity.__init__(self, f"{op}_{quantity}", unit)
        StateConsumer.__init__(self)

        self.discr = discr

        self.quantity = quantity

        from functools import partial

        if op == "min":
            self._myop = partial(self.discr.nodal_min, "vol")
        elif op == "max":
            self._myop = partial(self.discr.nodal_max, "vol")
        elif op == "sum":
            self._myop = partial(self.discr.nodal_sum, "vol")
        else:
            raise RuntimeError("unknown operation {op}")

    def __call__(self):
        """Return the requested quantity."""
        raise NotImplementedError


class ConservedQuantity(PhysicalQuantity):
    """Logging support for conserved quantities."""

    def __init__(self, discr: Discretization, quantity: str, unit: str, op: str,
                 dim: int = None, name: str = None):
        if name:
            PhysicalQuantity.__init__(self, discr, quantity, unit, op, name)
        elif dim:
            PhysicalQuantity.__init__(self, discr, quantity, unit, op,
                                      name=f"{op}_{quantity}{dim}")
        else:
            PhysicalQuantity.__init__(self, discr, quantity, unit, op)
        self.dim = dim

    def __call__(self):
        """Return the requested conserved quantity."""
        if self.state is None:
            return None

        from mirgecom.euler import split_conserved

        cv = split_conserved(self.discr.dim, self.state)

        self.state = None

        cq = getattr(cv, self.quantity)

        if isinstance(cq, ndarray):
            if self.dim is None:
                raise RuntimeError("Missing 'dim' parameter for dimensional "
                                   f"ConservedQuantity '{self.quantity}'.")
            return self._myop(cq[0])
        else:
            return self._myop(cq)


class DependentQuantity(PhysicalQuantity):

    def __init__(self, discr: Discretization, eos: GasEOS,
                 quantity: str, unit: str, op: str, name: str = None):
        if name:
            PhysicalQuantity.__init__(self, discr, quantity, unit, op, name)
        else:
            PhysicalQuantity.__init__(self, discr, quantity, unit, op)
        self.eos = eos

    def __call__(self):
        """Return the requested dependent quantity."""
        if self.state is None:
            return None

        from mirgecom.euler import split_conserved

        cv = split_conserved(self.discr.dim, self.state)
        dv = self.eos.dependent_vars(cv)

        self.state = None

        return self._myop(getattr(dv, self.quantity))


class KernelProfile(LogQuantity):
    """Logging support for results of the OpenCL kernel profiling."""

    def __init__(self, actx: PyOpenCLProfilingArrayContext,
                 kernel_name: str, stat: str, name: str = None):
        if stat == "time":
            unit = "s"
        else:
            unit = ""

        if name:
            LogQuantity.__init__(self, name,
                                 unit, f"{stat} of '{kernel_name}'")
        else:
            LogQuantity.__init__(self, f"{kernel_name}_{stat}",
                                 unit, f"{stat} of '{kernel_name}'")
        self.kernel_name = kernel_name
        self.actx = actx
        self.stat = stat

    def __call__(self):
        """Return the requested quantity."""
        return(self.actx.get_profiling_data_for_kernel(self.kernel_name, self.stat))
