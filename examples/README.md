# *MIRGE-Com* examples

This directory has a collection of examples to demonstrate and test *MIRGE-Com*
capabilities.  Examples using the "-mpi" naming convention are MPI parallel and
are able to run on mulitple GPUs or CPUs in a suitable MPI environemnt. All of
the example exercise some unique feature of *MIRGE-Com*.  The examples and the
unique features they exercise are as follows:

- `autoignition-mpi.py`: Chemistry verification case with Pyrometheus
- `heat-source-mpi.py`: Diffusion operator
- `lump-mpi.py`: Lump advection, advection verification case
- `mixture-mpi.py`: Mixture EOS with Pyrometheus
- `scalar-lump-mpi.py`: Scalar component lump advection verification case
- `pulse-mpi.py`: Acoustic pulse in a box, wall boundary test case
- `sod-mpi.py`: Sod's shock case: Fluid test case with strong shock
- `vortex-mpi.py`: Isentropic vortex advection: outflow boundaries, verification
- `hotplate-mpi.py`: Isothermal BC verification (prescribed exact soln)
- `doublemach-mpi.py`: AV test case
- `nsmix-mpi.py`: Viscous mixture w/Pyrometheus-based EOS
- `poiseuille-mpi.py`: Poiseuille flow verification case
- `poiseuille-multispecies-mpi.py`: Poiseuille flow with passive scalars
- `scalar-advdiff-mpi.py`: Scalar advection-diffusion verification case
