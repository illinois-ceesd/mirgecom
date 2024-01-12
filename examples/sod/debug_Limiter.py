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
import yaml
import logging
import sys
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze
from meshmode.array_context import (
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
from mirgecom.integrators import (
    euler_step,
#    rk2_step,
#    rk4_step,
#    ssprk32_step,
#    ssprk43_step,
#    lsrk54_step
)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary
)
#from mirgecom.boundary_new import OutflowBoundary
from mirgecom.fluid import make_conserved#, species_mass_fraction_gradient
#from mirgecom.initializers import PlanarDiscontinuity
#from mirgecom.transport import SimpleTransport
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.viscous import get_viscous_timestep, get_viscous_cfl

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time,
)

from pytools.obj_array import make_obj_array

#from mirgecom.artificial_viscosity import \
#    av_laplacian_operator, smoothness_indicator

from mirgecom.limiter import limiter_liu_osher, neighbor_list

############################################################################

class PlanarDiscontinuity:
    r"""Solution initializer for flow with a discontinuity.

    This initializer creates a physics-consistent flow solution
    given an initial thermal state (pressure, temperature) and an EOS.

    The solution varies across a planar interface defined by a tanh function
    located at disc_location for pressure, temperature, velocity, and mass fraction

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=3, normal_dir=0, disc_location=0, nspecies=0,
            temperature_left=None, temperature_right=None,
            mass_left=None, mass_right=None,
            pressure_left, pressure_right,
            velocity_left=None, velocity_right=None,
            species_mass_left=None, species_mass_right=None,
            convective_velocity=None, sigma=0.5
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        normal_dir: int
            specifies the direction (plane) the discontinuity is applied in
        disc_location: float or Callable
            fixed location of discontinuity or optionally a function that
            returns the time-dependent location.
        nspecies: int
            specifies the number of mixture species
        pressure_left: float
            pressure to the left of the discontinuity
        temperature_left: float
            temperature to the left of the discontinuity
        velocity_left: numpy.ndarray
            velocity (vector) to the left of the discontinuity
        species_mass_left: numpy.ndarray
            species mass fractions to the left of the discontinuity
        pressure_right: float
            pressure to the right of the discontinuity
        temperature_right: float
            temperaure to the right of the discontinuity
        velocity_right: numpy.ndarray
            velocity (vector) to the right of the discontinuity
        species_mass_right: numpy.ndarray
            species mass fractions to the right of the discontinuity
        sigma: float
           sharpness parameter
        """
        if velocity_left is None:
            velocity_left = np.zeros(shape=(dim,))
        if velocity_right is None:
            velocity_right = np.zeros(shape=(dim,))

        if nspecies is not None:
            self._nspecies = nspecies
            if species_mass_left is None:
                species_mass_left = np.zeros(shape=(nspecies,))
            if species_mass_right is None:
                species_mass_right = np.zeros(shape=(nspecies,))
        else:
            self._nspecies = nspecies

        self._dim = dim
        self._disc_location = disc_location
        self._sigma = sigma
        self._ul = velocity_left
        self._ur = velocity_right
        self._uc = convective_velocity
        self._pl = pressure_left
        self._pr = pressure_right
        self._rl = mass_left
        self._rr = mass_right
        self._tl = temperature_left
        self._tr = temperature_right
        self._yl = species_mass_left
        self._yr = species_mass_right
        self._xdir = normal_dir
        if self._xdir >= self._dim:
            self._xdir = self._dim - 1

    def __call__(self, x_vec, eos, *, time=0.0):
        """Create the mixture state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        x = x_vec[self._xdir]
        actx = x.array_context
        x0 = self._disc_location        
#        if isinstance(self._disc_location, Number):
#            x0 = self._disc_location
#        else:
#            x0 = self._disc_location(time)

        xtanh = 1.0/self._sigma*(x0 - x)
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        pressure = self._pl + (self._pr - self._pl)*weight
        if self._tr == None:
          mass = self._rl + (self._rr - self._rl)*weight        
        if self._rr == None:
          temperature = self._tl + (self._tr - self._tl)*weight
        velocity = self._ul + (self._ur - self._ul)*weight

        if self._nspecies:
            y = self._yl + (self._yr - self._yl)*weight
            mass = eos.get_density(pressure, temperature,
                                   species_mass_fractions=y)
            specmass = mass * y
        else:
            y = None
            if self._rr == None:
              mass = pressure/temperature/eos.gas_const()
            specmass = None

        mom = mass * velocity
        if self._tr == None:
          internal_energy = pressure/(eos.gamma() - 1.0)
        if self._rr == None:
          internal_energy = mass * eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)

        kinetic_energy = 0.5 * mass * np.dot(velocity, velocity)
        energy = (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, casename="debug",
         user_input_file=None, restart_file=None, use_profiling=False,
         use_logmgr=False, use_lazy_eval=False):
         
    """Drive the 1D Flame example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    restart_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    snapshot_pattern = restart_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
    cl_ctx = ctx_factory()
    if use_profiling:
        if use_lazy_eval:
            raise RuntimeError("Cannot run lazy with profiling.")
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        if use_lazy_eval:
            actx = PytatoPyOpenCLArrayContext(queue)
        else:
            actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # discretization and model control
    order = 2

    # default i/o frequencies
    nviz = 100
    nrestart = 10000
    nhealth = 1
    nstatus = 100 

    # default timestepping control
    #integrator = "ssprk43"
    integrator = "euler"

    storeCFL = True
    do_not_kill = False
        
    use_AV = False
    use_limiter = True
    use_overintegration = False
    use_filter = False

    constant_cfl = False    
    current_dt = 0.25e-3/2
    
    t_final = 2.0
    
    niter = int(t_final/current_dt)
    
##############################################################################

    ##################################################

    # param sanity check
    allowed_integrators = ["euler", "rk2", "rk4", "ssprk32", "ssprk43"]
    if(integrator not in allowed_integrators):
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if integrator == "euler":
        timestepper = euler_step
    if integrator == "rk2":
        timestepper = rk2_step
    if integrator == "rk4":
        timestepper = rk4_step
    if integrator == "ssprk32":
        timestepper = ssprk32_step
    if integrator == "ssprk43":
        timestepper = ssprk43_step
                 
    ##################################################

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")
        print(f"\tuse_AV = {use_AV}")
        print(f"\tuse_limiter = {use_limiter}")
        print(f"\tuse_overintegration = {use_overintegration}")
        print(f"\tuse_filter = {use_filter}")

###############################################################################

    dim = 2
    current_cfl = 1.0
    current_t = 0
    current_step = 0

    eos = IdealSingleGas(gamma=1.4,gas_const=1.0)

    mu = None
    kappa = None
    Pr = None

    gas_model = GasModel(eos=eos)

#######################################

    flow_init = PlanarDiscontinuity(dim=dim,
                                    disc_location=0.0,
                                    sigma=0.001, nspecies=None,
                                    mass_right=0.125,
                                    mass_left=1.0,
                                    pressure_right=0.1,
                                    pressure_left=1.0)

    def _inflow_func(nodes, cv, eos, **kwargs):
        return flow_init(x_vec=nodes, eos=eos, time=0.)

    def _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus, init_func,
                             **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = thaw(bnd_discr.nodes(), actx)
        return make_fluid_state(init_func(nodes=nodes, eos=gas_model.eos,
                                          cv=state_minus.cv, **kwargs),
                                gas_model=gas_model)

    def _inflow_boundary_state(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus,
                                    _inflow_func, **kwargs)
   
    inflow_boundary = PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state)
    outflow_boundary = PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state)

    boundaries = {DTAG_BOUNDARY("inflow"): inflow_boundary,
                  DTAG_BOUNDARY("outflow"): outflow_boundary}                  

    ##################################################

    restart_step = None
    if restart_file is None:
    
        char_len = 0.04
        box_ll = (-4.0, -0.04)
        box_ur = (+4.0, +0.04)
        num_elements = ( int(np.rint((box_ur[0]-box_ll[0])/char_len))+1, \
                         int(np.rint((box_ur[1]-box_ll[1])/char_len))+1 )

        print(num_elements)

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                                a=box_ll,
                                b=box_ur,
                                n=num_elements,
                                periodic=(False, True),
                                # mesh_type="X",
                                boundary_tag_to_face={
                                    "inflow": ["-x"],
                                    "outflow": ["+x"]})

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

############################################################

    #overintegration
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        default_simplex_group_factory, QuadratureSimplexGroupFactory
    if rank == 0:
        print('Making discretization')
        logging.info("Making discretization")
    
    dcoll = EagerDGDiscretization(
            actx, local_mesh,
            discr_tag_to_group_factory={
                 DISCR_TAG_BASE: default_simplex_group_factory(
                     base_dim=local_mesh.dim, order=order),
                 DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
            },
            mpi_communicator=comm
    )

    nodes = thaw(dcoll.nodes(), actx)
    
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None
        
    if (use_filter == True):  
        vol_discr = dcoll.discr_from_dd("vol")
        group = vol_discr.groups
        ModeRespFunc = make_spectral_filter(actx, group, cutoff, frfunc)
        
#################################################################

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        #current_cv = _flow_bnd(nodes, eos)
        current_cv = flow_init(x_vec=nodes, eos=gas_model.eos, time=0.)
    else:
        current_t = restart_data["t"]
        current_step = restart_step

        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol"))

            current_cv = connection(restart_data["state"])
            #temperature_seed = connection(restart_data["temperature_seed"])
        else:
            current_cv = restart_data["state"]
            #temperature_seed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model)   

    ##################################################

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)
        #logmgr_add_package_versions(logmgr)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(dcoll)

    initname = "debug"
    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

################################################################################################

    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(dcoll, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(dcoll, "vol", x))[()]

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]

    def get_dt(state):
        return make_obj_array([get_viscous_timestep(dcoll, state=state)])
    compute_dt = actx.compile(get_dt)

    def get_cfl(state, dt):
        return make_obj_array([get_viscous_cfl(dcoll, dt, state=state)])
    compute_cfl = actx.compile(get_cfl)

################################################################################################

    def my_write_viz(step, t, cv, dv,
                     ts_field=None, alpha_field=None,
                     rhs=None,
                     grad_cv=None, grad_t=None,grad_y=None,
                     flux=None,
                     ref_cv=None,sponge_sigma=None,gas_model=None):

#        filtered = None
#        avg = None
#        theta = None
#        minRatio = None
#        maxRatio = None        
        if (use_limiter == True):
            neighbors = neighbor_list(dim, local_mesh)
            filtered, theta, avg, minRatio, maxRatio = \
                         limiter_liu_osher(dcoll,neighbors,dv.pressure,vizdata=True)
        
        viz_fields = [("CV", cv),
                      ("CV_U", cv.momentum[0]/cv.mass),
                      ("CV_V", cv.momentum[1]/cv.mass),
                      ("DV_P", dv.pressure),
                      ("DV_T", dv.temperature),
                      ("Lim_Filt_Field", filtered),                      
                      ("Lim_Avg", avg),
                      ("Lim_theta", theta),
                      ("Lim_minRatio", minRatio),
                      ("Lim_maxRatio", maxRatio),
                      #("tagged_cells", tagged_cells),
                      #("alpha", alpha_field),
                      ("dt" if constant_cfl else "cfl", ts_field)
                      ]
                      
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)
        
        ###############
        
#        file_name = "adjacency.vtu"

#        import os        
#        if os.path.exists("geometry.vtu"):
#            os.remove("geometry.vtu")
#        visualizer.write_vtk_file("geometry.vtu", [
#            ("f", thaw(dcoll.nodes()[0], actx)),
#            ])

#        from pyvisfile.vtk import (
#                UnstructuredGrid, DataArray,
#                AppendedDataXMLGenerator,
#                VTK_LINE,
#                VF_LIST_OF_COMPONENTS)                      

#        centroids, neighbors, connections = neighbor_list(local_mesh)
#        
#        nconnections = connections.shape[0]              
#        grid = UnstructuredGrid(
#                (local_mesh.nelements,
#                    DataArray("points",
#                        centroids,
#                        vector_format=VF_LIST_OF_COMPONENTS)),
#                cells=connections.reshape(-1),
#                cell_types=np.asarray([VTK_LINE] * nconnections,
#                    dtype=np.uint8))

#        if os.path.exists(file_name):
#            os.remove(file_name)

#        compressor = None
#        with open(file_name, "w") as outf:
#            AppendedDataXMLGenerator(compressor)(grid).write(outf)

        ###############
            
        return

    def my_write_restart(step, t, cv):
        rst_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

################################################################################################

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            cfl = current_cfl
            ts_field = cfl * compute_dt(state)[0]
            ts_field = thaw(freeze(ts_field, actx), actx)
            dt = global_reduce(vol_min_loc(ts_field), op="min", comm=comm)
        else:
            ts_field = compute_cfl(state, current_dt)[0]
            cfl = global_reduce(vol_max_loc(ts_field), op="max", comm=comm)

        return ts_field, cfl, min(t_remaining, dt)

################################################################################################

    def my_health_check(cv, dv, ts_field):

        health_error = False
        pressure = thaw(freeze(dv.pressure, actx), actx)
        temperature = thaw(freeze(dv.temperature, actx), actx)

        if global_reduce(check_naninf_local(dcoll, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

################################################################################################

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        cv = my_limiter(state) if use_limiter else state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)    

        cv = fluid_state.cv
        dv = fluid_state.dv

        try:

            if (use_AV == True):
                alpha_field = my_get_alpha(dcoll, fluid_state, alpha_sc)
            else:
                alpha_field = None

            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv, ts_field), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv)

            if do_viz:
                        
                if (storeCFL == False): ts_field = None
                my_write_viz(step=step, t=t, cv=cv, dv=dv,
                             ts_field=ts_field, alpha_field=alpha_field)
                
        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, cv=cv, dv=dv,
                             ts_field=ts_field, alpha_field=alpha_field)
            raise

        return cv, dt

    def my_post_step(step, t, dt, state):   
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
            
        return state, dt

#####################################################################   

#    def my_limiter(cv):
#    
#        actx = cv.array_context
#        neighbors = neighbor_list(dim, local_mesh)

#        mass_lim = limiter_liu_osher(discr,neighbors,cv.mass)
#        velc_lim = make_obj_array([
#               limiter_liu_osher(discr,neighbors,cv.momentum[0]/cv.mass),
#               limiter_liu_osher(discr,neighbors,cv.momentum[1]/cv.mass)
#               ])
#        ener_lim = limiter_liu_osher(discr,neighbors,cv.energy)
#        spec_y_lim = cv.species_mass_fractions

#        return make_conserved(dim=dim, mass=mass_lim, energy=ener_lim,
#                                     momentum=velc_lim*mass_lim,
#                                     species_mass=mass_lim*spec_y_lim)

    def my_limiter(cv,dv=None):
    
        #actx = cv.array_context
        neighbors = neighbor_list(dim, local_mesh)

        mass_lim = limiter_liu_osher(dcoll, neighbors, cv.mass)
        velc_lim = make_obj_array([
              limiter_liu_osher(dcoll, neighbors, cv.momentum[0]/cv.mass),
              limiter_liu_osher(dcoll, neighbors, cv.momentum[1]/cv.mass)
                    ])
        
        if (dv is not None):
          pres_lim = limiter_liu_osher(dcoll, neighbors, dv.pressure)
          temp_lim = pres_lim/(gas_model.eos.gas_const(cv)*mass_lim)
          int_energy = gas_model.eos.get_internal_energy(
                                              temp_lim, spec_lim)
          kin_energy = 0.5 * np.dot(velc_lim, velc_lim)
          ener_lim = mass_lim * (int_energy + kin_energy)

          return make_conserved(dim=dim, mass=mass_lim, energy=ener_lim,
                           momentum=velc_lim*mass_lim,
                           species_mass=mass_lim*spec_lim
                           ), \
                 temp_lim

        else:        
          pressure = (cv.energy - gas_model.eos.kinetic_energy(cv))*(gas_model.eos.gamma() - 1.0)
          pres_lim = limiter_liu_osher(dcoll,neighbors,pressure)

          ener_lim = pres_lim/(gas_model.eos.gamma() - 1.0) + \
                           0.5 * mass_lim * np.dot(velc_lim, velc_lim)

          return make_conserved(dim=dim, mass=mass_lim, energy=ener_lim,
                                momentum=velc_lim*mass_lim,
                                species_mass=cv.species_mass_fractions
                                )

    def my_rhs(t, state):
   
        #cv = my_limiter(state) if use_limiter else state
        #fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)               

        return euler_operator(dcoll, state=fluid_state, time=t,
                              boundaries=boundaries, gas_model=gas_model,
                              quadrature_tag=quadrature_tag)      

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_cv) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=current_state.cv,
                      dt=current_dt, t_final=t_final, t=current_t,
                      istep=current_step)
    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_state.dv
    my_write_viz(step=current_step, t=current_t,
                 cv=current_state.cv, dv=final_dv)
    my_write_restart(step=current_step, t=current_t,
                     cv=current_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "debug"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_file=restart_file, user_input_file=input_file,
         use_profiling=args.profile, use_lazy_eval=args.lazy, use_logmgr=args.log)
         
