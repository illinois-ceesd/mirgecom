"""Demonstrate simple MPI use."""

from mpi4py import MPI

comm = MPI.COMM_WORLD

comm.Barrier()

# This is here to document in the CI examples log how the MPI examples
# are being run.
print(f"Hello MPI World! My rank is {comm.Get_rank()} / {comm.Get_size()}")
