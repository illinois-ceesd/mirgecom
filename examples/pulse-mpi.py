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
import logging
from mpi4py import MPI
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.euler import (
    inviscid_operator,
    #    split_conserved
)
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedBoundary,
    AdiabaticSlipBoundary
)
from mirgecom.initializers import (
    Lump,
    make_pulse
)
from mirgecom.eos import IdealSingleGas
from time import perf_counter as gettime
import sys
from contextlib import contextmanager


class Profiler:
    def resettime(self):
        self.t0 = gettime()

    def setbarrier(self, inoption):
        self.timerbarrier = inoption

    def setmpicommunicator(self, inmpicommunicator):
        self.mpicommobj = inmpicommunicator
        self.myrank = self.mpicommobj.Get_rank()
        self.numproc = self.mpicommobj.Get_size()

    def __init__(self, inmpicommunicator):
        self.setmpicommunicator(inmpicommunicator)
        self.resettime()
        self.opensections = []
        self.sectiontimes = []
        self.timerbarrier = True

    def starttimer(self, sectionname="MAIN"):

        if sectionname == "MAIN":
            self.resettime()

        numopensections = len(self.opensections)

        opensection = [numopensections, sectionname, 1, 0, 0]

        if self.timerbarrier:
            self.mpicommobj.Barrier()

        opensection[3] = gettime()

        self.opensections.append(opensection)

    def endtimer(self, sectionname="MAIN"):
        numopensections = len(self.opensections)

        if numopensections <= 0:
            print(
                "Error: EndTimer(",
                sectionname,
                ") called with no matching StartTimer.",
            )
            1 / 0

        sectiontime = gettime()

        if self.timerbarrier:
            self.mpicommobj.Barrier()

        opensectionindex = numopensections - 1

        if sectionname != self.opensections[opensectionindex][1]:
            print(
                "SectionName: Expected(",
                self.opensections[opensectionindex][1],
                ")",
                ", Got(",
                sectionname,
                ")",
            )
            1 / 0

        opensection = self.opensections.pop()
        sectiontime = sectiontime - opensection[3]
        opensection[3] = sectiontime
        opensectionindex = opensectionindex - 1

        # Update parent's sub-timers
        if opensectionindex >= 0:
            self.opensections[opensectionindex][4] += sectiontime

        # Update section if it exists
        numsections = len(self.sectiontimes)
        match = False

        for i in range(numsections):
            if self.sectiontimes[i][1] == sectionname:
                existingsection = self.sectiontimes[i]
                existingsection[2] += 1
                existingsection[3] += sectiontime
                existingsection[4] += opensection[4]
                match = True
                break

        # Create new section if it didn't exist
        if not match:
            self.sectiontimes.append(opensection)

    @contextmanager
    def contexttimer(self, contextname=""):
        self.starttimer(contextname)
        yield contextname
        self.endtimer(contextname)

    def writeserialprofile(self, filename=""):

        # copy the timers to avoid destructing the list when printing
        sectiontimers = list(self.sectiontimes)

        numsections = len(sectiontimers)
        numcurrentsections = numsections
        minlevel = 0

        profilefile = sys.stdout

        if filename != "":
            profilefile = open(filename, "w")

        if numcurrentsections > 0:
            print(
                "# SectionName   NumCalls  TotalTime   ChildTime",
                file=profilefile,
            )

        while numcurrentsections > 0:
            match = False
            for i in range(numcurrentsections):
                if sectiontimers[i][0] == minlevel:
                    sectiontimer = sectiontimers.pop()
                    # print out SectionName NumCalls TotalTime ChildTime
                    print(
                        sectiontimer[1],
                        sectiontimer[2],
                        sectiontimer[3],
                        sectiontimer[4],
                        file=profilefile,
                    )
                    match = True
                    break

            if match is False:
                minlevel += 1

            numcurrentsections = len(sectiontimers)

        if filename != "":
            profilefile.close()

    # WriteParallelProfile is a collective call, must be called on all procs
    def writeparallelprofile(self, filename=""):

        sectiontimers = list(self.sectiontimes)

        numsections = len(sectiontimers)
        mynumsections = np.zeros(1, dtype=int)
        mycheck = np.zeros(1, dtype=int)

        self.mpicommobj.Barrier()
        numproc = self.mpicommobj.Get_size()

        if self.myrank == 0:
            mynumsections[0] = numsections

        self.mpicommobj.Bcast(mynumsections, root=0)

        if numsections == mynumsections[0]:
            mynumsections[0] = 0
        else:
            mynumsections[0] = 1
            print(
                "(",
                self.myrank,
                "): ",
                numsections,
                " != ",
                mynumsections[0],
            )
            1 / 0

        self.mpicommobj.Reduce(mynumsections, mycheck, MPI.MAX, 0)

        if mycheck > 0:
            print(
                "ReduceTimers:Error: Disparate number of sections ",
                "across processors.",
            )
            1 / 0

        mysectiontimes = np.zeros(numsections, dtype="float")
        mintimes = np.zeros(numsections, dtype="float")
        maxtimes = np.zeros(numsections, dtype="float")
        sumtimes = np.zeros(numsections, dtype="float")

        for i in range(numsections):
            mysectiontimes[i] = sectiontimers[i][3]

        self.mpicommobj.Reduce(mysectiontimes, mintimes, MPI.MIN, 0)
        self.mpicommobj.Reduce(mysectiontimes, maxtimes, MPI.MAX, 0)
        self.mpicommobj.Reduce(mysectiontimes, sumtimes, MPI.SUM, 0)

        if self.myrank == 0:

            profilefile = sys.stdout
            if filename != "":
                profilefile = open(filename, "w")
            print("# NumProcs: ", numproc, file=profilefile)
            print(
                "# SectionName   MinTime   MaxTime   MeanTime",
                file=profilefile,
            )
            for i in range(numsections):
                sectiontime = sectiontimers[i]
                print(
                    sectiontime[1],
                    mintimes[i],
                    maxtimes[i],
                    sumtimes[i] / float(self.numproc),
                    file=profilefile,
                )

            if filename != "":
                profilefile.close()

        self.mpicommobj.Barrier()

    def makeparallelfilename(self, rootname=""):

        myrootname = rootname
        if myrootname == "":
            myrootname = "Profiler"

        numproc = self.mpicommobj.Get_size()

        profilefilename = myrootname + "_ParTimes_" + str(numproc)

        return profilefilename


# Surrogate for the currently non-functioning Uniform class
def set_uniform_solution(t, x_vec, eos=IdealSingleGas()):

    dim = len(x_vec)
    _rho = 1.0
    _p = 1.0
    _velocity = np.zeros(shape=(dim,))
    _gamma = 1.4

    mom0 = _rho * _velocity
    e0 = _p / (_gamma - 1.0)
    ke = 0.5 * np.dot(_velocity, _velocity) / _rho
    x_rel = x_vec[0]
    zeros = 0.0*x_rel
    ones = zeros + 1.0

    mass = zeros + _rho
    mom = make_obj_array([mom0 * ones for i in range(dim)])
    energy = e0 + ke + zeros

    return flat_obj_array(mass, energy, mom)


comm = MPI.COMM_WORLD
myprofiler = Profiler(comm)

myprofiler.starttimer()


def main(ctx_factory=cl.create_some_context):

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    logger = logging.getLogger(__name__)
    myprofiler.starttimer("Setup")

    dim = 2
    nel_1d = 1024
    order = 1
    exittol = 2e-2
    exittol = 100.0
    t_final = .0001
    current_cfl = 1.0
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    #    vel[:dim] = 1.0
    current_dt = .00001
    current_t = 0
    eos = IdealSingleGas()
    initializer = Lump(center=orig, velocity=vel, rhoamp=0.0)
    casename = 'pulse'
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    wall = AdiabaticSlipBoundary()
    boundaries = {BTAG_ALL: wall}
    constant_cfl = False
    nstatus = 10
    nviz = 10
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    box_ll = -0.5
    box_ur = 0.5

    nproc = comm.Get_size()
    rank = comm.Get_rank()
    num_parts = nproc

    myprofiler.starttimer("Mesh")
    global_nelements = 0
    local_nelements = 0
    if num_parts > 1:
        from meshmode.distributed import (
            MPIMeshDistributor,
            get_partition_by_pymetis,
        )

        mesh_dist = MPIMeshDistributor(comm)
        if mesh_dist.is_mananger_rank():
            from meshmode.mesh.generation import generate_regular_rect_mesh

            mesh = generate_regular_rect_mesh(
                a=(box_ll,) * dim, b=(box_ur,) * dim, n=(nel_1d,) * dim
            )
            global_nelements = mesh.nelements
            logging.info(f"Total {dim}d elements: {global_nelements}")

            part_per_element = get_partition_by_pymetis(mesh, num_parts)

            myprofiler.starttimer("MeshComm")
            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
            myprofiler.endtimer("MeshComm")
            del mesh

        else:
            myprofiler.starttimer("MeshComm")
            local_mesh = mesh_dist.receive_mesh_part()
            myprofiler.endtimer("MeshComm")
    else:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        local_mesh = generate_regular_rect_mesh(
            a=(box_ll,) * dim, b=(box_ur,) * dim, n=(nel_1d,) * dim
        )

    local_nelements = local_mesh.nelements
    myprofiler.endtimer("Mesh")

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    current_state = initializer(0, nodes)

    #    qs = split_conserved(dim, current_state)
    #    qs.energy = qs.energy + make_pulse(amp=1.0, w=.2, r0=orig, r=nodes)
    current_state[1] = current_state[1] + make_pulse(amp=1.0, w=.1, r0=orig, r=nodes)
    visualizer = make_visualizer(discr, discr.order + 3
                                 if discr.dim == 2 else discr.order)
    #    initname = initializer.__class__.__name__
    initname = "pulse"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_rhs(t, state):
        return inviscid_operator(discr, q=state, t=t,
                                 boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        return sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                              logger=logger, q=state, vizname=casename, step=step,
                              t=t, dt=dt, nstatus=nstatus, nviz=nviz,
                              exittol=exittol, constant_cfl=constant_cfl,
                              profiler=myprofiler)

    myprofiler.endtimer("Setup")
    myprofiler.starttimer("Stepping")
    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper, checkpoint=my_checkpoint,
                    get_timestep=get_timestep, state=current_state,
                      t=current_t, t_final=t_final, profiler=myprofiler)
    myprofiler.endtimer("Stepping")

    #    if current_t != checkpoint_t:
    if rank == 0:
        logger.info("Checkpointing final state ...")
    my_checkpoint(current_step, t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state)

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    myrank = comm.Get_rank()
    numproc = comm.Get_size()

    myprofiler.starttimer("mirgecom")
    main()
    myprofiler.endtimer("mirgecom")

    myprofiler.endtimer()

    if myrank == 0:
        myprofiler.writeserialprofile("pulse_rank0_profile_"+str(numproc))

    if numproc > 1:
        parallelprofilefilename = myprofiler.makeparallelfilename("pulse")
        myprofiler.writeparallelprofile(parallelprofilefilename)

# vim: foldmethod=marker
