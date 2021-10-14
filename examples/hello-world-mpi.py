"""Demonstrate simple MPI use."""

from mirgecom.mpi import mpi_entry_point


@mpi_entry_point
def main():
    """Run the demo."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    comm.Barrier()

    # This is here to document in the CI examples log how the MPI examples
    # are being run.
    print(f"Hello MPI World! My rank is {comm.Get_rank()} / {comm.Get_size()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hello World MPI demo")
    parser.parse_args()

    main()
