"""Reproduce bug that hits us in parallel."""

from mirgecom.mpi import mpi_entry_point
from functools import partial
import pyopencl as cl
import pyopencl.tools as cl_tools

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from grudge.eager import EagerDGDiscretization
from grudge import sym as grudge_sym

from mirgecom.simutil import create_parallel_grid


@mpi_entry_point
def main(ctx_factory=cl.create_some_context):
    """Drive bug example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 3
    nel_1d = 17
    order = 1
    box_ll = -0.5
    box_ur = 0.5

    bnd_tags = {
        "left": ["-x", ],
        "right": ["+x", ],
        "bottom": ["-y", ],
        "top": ["+y", ],
        "back": ["-z", ],
        "front": ["+z", ],
    }

    boundaries = {
        grudge_sym.DTAG_BOUNDARY("left"): -1,
        grudge_sym.DTAG_BOUNDARY("right"): 1,
        grudge_sym.DTAG_BOUNDARY("bottom"): -2,
        grudge_sym.DTAG_BOUNDARY("top"): 2,
        grudge_sym.DTAG_BOUNDARY("back"): -3,
        grudge_sym.DTAG_BOUNDARY("front"): 3
    }

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    num_parts = nproc

    from meshmode.mesh.generation import generate_regular_rect_mesh
    if num_parts > 1:
        generate_grid = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                                b=(box_ur,) * dim, n=(nel_1d,) * dim,
                                boundary_tag_to_face=bnd_tags)
        local_mesh, global_nelements = create_parallel_grid(comm, generate_grid)
    else:
        local_mesh = generate_regular_rect_mesh(
            a=(box_ll,) * dim, b=(box_ur,) * dim, n=(nel_1d,) * dim,
            boundary_tag_to_face=bnd_tags
        )
        global_nelements = local_mesh.nelements
    local_nelements = local_mesh.nelements
    print(f"{local_nelements=},{global_nelements=}")

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )

    local_boundaries = {}
    nonlocal_boundaries = {}
    for btag in boundaries:
        bnd_discr = discr.discr_from_dd(btag)
        bnd_nodes = thaw(actx, bnd_discr.nodes())
        bnd_normal = thaw(actx, discr.normal(btag))
        num_bnd_nodes = bnd_nodes[0][0].shape[0]

        if num_bnd_nodes > 0:
            local_boundaries[btag] = boundaries[btag]
        else:
            nonlocal_boundaries[btag] = boundaries[btag]
            print(f"{rank=},{btag=}")
            print(f"{bnd_nodes=}")
            print(f"{bnd_normal=}")

        # next line reproduces issue
        result = bnd_nodes @ bnd_normal
        print(f"{rank=},{btag=},{result=}")

    print(f"{rank=},{nonlocal_boundaries=}")


if __name__ == "__main__":
    main()
