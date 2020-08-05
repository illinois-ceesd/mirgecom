from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

comm.Barrier()
print(f'Hello MPI World! My rank is {myrank} / {nproc}')
