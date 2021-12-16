"""Demonstrate simple MPI use."""

from mirgecom.mpi import mpi_entry_point


@mpi_entry_point
def main(dist_ctx):
    """Run the demo."""

    dist_ctx.barrier()

    # This is here to document in the CI examples log how the MPI examples
    # are being run.
    print(f"Hello MPI World! My rank is {dist_ctx.rank} / {dist_ctx.size}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
