"""Reproduce a restart bug."""
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from grudge.eager import EagerDGDiscretization


def main(ctx_factory=cl.create_some_context):
    """Run restart bug."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nel_1d = 8
    order = 1
    box_ll = -0.005
    box_ur = 0.005

    from meshmode.mesh.generation import generate_regular_rect_mesh
    local_mesh = generate_regular_rect_mesh(a=(box_ll,)*dim,
                                            b=(box_ur,) * dim,
                                            nelements_per_axis=(nel_1d,)*dim)
    discr = EagerDGDiscretization(actx, local_mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    rst_filename = "restart_data/test.pkl"
    from mirgecom.initializers import Uniform
    initializer = Uniform(dim=dim)

    # Set the current state from time 0
    from mirgecom.eos import IdealSingleGas
    test_state = initializer(eos=IdealSingleGas(), x_vec=nodes, t=0)

    rst_data = {"state": test_state}
    from mirgecom.restart import write_restart_file
    write_restart_file(actx, rst_data, rst_filename)

    from mirgecom.restart import read_restart_data
    restart_data = read_restart_data(actx, rst_filename)

    resid = test_state - restart_data["state"]
    assert discr.norm(resid, np.inf) < 1e-14


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
