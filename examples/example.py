"""Example a bugzample bugzy."""

import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.discretization import create_discretization_collection

from mirgecom.simutil import (
    generate_and_distribute_mesh,
)
from mirgecom.mpi import mpi_entry_point
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel
from mirgecom.utils import force_evaluation


class MyRuntimeError(RuntimeError):
    """Simple exception for fatal driver errors."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, lazy=False):
    """Drive example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # Some discretization parameters
    dim = 2
    nel_1d = 8
    order = 1

    from meshmode.mesh.generation import generate_regular_rect_mesh
    box_ll = -0.005
    box_ur = 0.005
    generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                            b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
    local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                generate_mesh)

    discr = create_discretization_collection(actx, local_mesh, order=order,
                                             mpi_communicator=comm)

    ones = discr.zeros(actx) + 1.0
    eos = IdealSingleGas()
    nodes = actx.thaw(discr.nodes())

    from mirgecom.transport import SimpleTransport
    transport_model = None
    nspecies = 7
    kappa = 1e-5
    spec_diffusivity = 1e-5 * np.ones(nspecies)
    sigma = 1e-5
    transport_model = SimpleTransport(viscosity=sigma,
                                      thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    gas_model = GasModel(eos=eos, transport=transport_model)

    from mirgecom.navierstokes import grad_cv_operator

    def get_grad_cv(state, time):
        return grad_cv_operator(discr, gas_model, boundaries, state,
                                time=time)

    compute_grad_cv = actx.compile(get_grad_cv)

    from mirgecom.gas_model import make_fluid_state

    def get_fluid_state(cv, tseed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=tseed)

    construct_fluid_state = actx.compile(get_fluid_state)

    my_boundary = AdiabaticSlipBoundary()
    boundaries = {BTAG_ALL: my_boundary}

    # Set the current state from time 0
    nspecies = 7
    can_y = np.ones(nspecies)
    velocity = np.zeros(dim)
    initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                     pressure=1e6, temperature=1500.0,
                                     massfractions=can_y, velocity=velocity)
    current_cv = initializer(eos=gas_model.eos, x_vec=nodes)
    temperature_seed = 1500.0 * ones
    current_cv = force_evaluation(actx, current_cv)

    # The temperature_seed going into this function is:
    # - At time 0: the initial temperature input data (maybe from Cantera)
    # - On restart: the restarted temperature seed from restart file (saving
    #               the *seed* allows restarts to be deterministic
    current_fluid_state = construct_fluid_state(current_cv, temperature_seed)
    grad_cv = compute_grad_cv(current_fluid_state, time=0.0)
    print(f"{grad_cv=}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com bugzample")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    args = parser.parse_args()
    lazy = args.lazy
    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    main(actx_class, lazy=lazy)

# vim: foldmethod=marker
