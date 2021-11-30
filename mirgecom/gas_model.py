""":mod:`mirgecom.gas_model` provides utilities to deal with gases.

Fluid State Handling
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FluidState
.. autofunction: make_fluid_state
.. autofunction: make_fluid_state_on_boundary
.. autofunction: make_fluid_state_trace_pairs
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
import numpy as np  # noqa
from meshmode.dof_array import DOFArray  # noqa
from dataclasses import dataclass
from arraycontext import dataclass_array_container
from mirgecom.fluid import ConservedVars
from mirgecom.eos import EOSDependentVars


@dataclass_array_container
@dataclass(frozen=True)
class FluidState:
    r"""Gas model-consistent fluid state.

    .. attribute:: cv

        :class:`~mirgecom.fluid.ConservedVars` for the fluid conserved state

    .. attribute:: dv

        :class:`~mirgecom.eos.EOSDependentVars` for the fluid state-dependent
        quantities corresponding to the chosen equation of state.
    """

    # FIXME: autoattribute for array_context, dim, nspecies

    #    .. attribute:: tv
    #
    #        :class:`~mirgecom.transport.TransportDependentVars` for the fluid
    #        state-dependent transport properties.

    cv: ConservedVars
    dv: EOSDependentVars
    # tv: TransportDependentVars

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return self.cv.array_context

    @property
    def dim(self):
        """Return the number of physical dimensions."""
        return self.cv.dim

    @property
    def nspecies(self):
        """Return the number of physical dimensions."""
        return self.cv.nspecies


def make_fluid_state(cv, eos, temperature_seed=None):
    """Create a fluid state from the conserved vars and equation of state.
    """
    return FluidState(
        cv=cv, dv=eos.dependent_vars(cv, temperature_seed=temperature_seed)
    )


def project_fluid_state(discr, btag, fluid_state, eos):
    """Create a fluid state from volume :class:`FluidState` *fluid_state*
    by projection onto the boundary and ensuring thermal consistency.
    """
    cv_sd = discr.project("vol", btag, fluid_state.cv)
    temperature_seed = None
    if fluid_state.cv.nspecies > 0:
        temperature_seed = discr.project("vol", btag, fluid_state.dv.temperature)
    return make_fluid_state(cv=cv_sd, eos=eos, temperature_seed=temperature_seed)


def _getattr_ish(obj, name):
    if obj is None:
        return None
    else:
        return getattr(obj, name)


def make_fluid_state_trace_pairs(cv_pairs, eos, temperature_seed_pairs=None):
    """Create a fluid state from the conserved vars and equation of state."""
    from grudge.trace_pair import TracePair
    if temperature_seed_pairs is None:
        temperature_seed_pairs = [None] * len(cv_pairs)

    return [TracePair(
        dd=cv_pair.dd,
        interior=make_fluid_state(cv_pair.int, eos,
                                  temperature_seed=_getattr_ish(tseed_pair, "int")),
        exterior=make_fluid_state(cv_pair.ext, eos,
                                  temperature_seed=_getattr_ish(tseed_pair, "ext")))
        for cv_pair, tseed_pair in zip(cv_pairs, temperature_seed_pairs)]


def make_fluid_state_interior_trace_pair(discr, state, eos):
    """Create a fluid state from the conserved vars and equation of state."""
    from grudge.eager import interior_trace_pair
    from grudge.trace_pair import TracePair
    cv_tpair = interior_trace_pair(discr, state.cv)
    # FIXME As above?
    if state.nspecies > 0:
        tseed_pair = interior_trace_pair(discr, state.dv.temperature)
        return TracePair(
            cv_tpair.dd,
            interior=make_fluid_state(cv_tpair.int, eos, tseed_pair.int),
            exterior=make_fluid_state(cv_tpair.ext, eos, tseed_pair.ext))
    return TracePair(cv_tpair.dd,
                     interior=make_fluid_state(cv_tpair.int, eos),
                     exterior=make_fluid_state(cv_tpair.ext, eos))
