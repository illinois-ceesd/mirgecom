"""Test the Navier-Stokes gas dynamics module with some manufactured solutions."""

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

import numpy as np
import numpy.random
import numpy.linalg as la  # noqa
import pyopencl.clmath  # noqa
import logging

from pytools.obj_array import (  # noqa
    flat_obj_array,
    make_obj_array,
)

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.navierstokes import ns_operator

from mirgecom.boundary import (  # noqa
   IsothermalNoSlipBoundary,  # noqa
    AdiabaticSlipBoundary,  # noqa
    PrescribedFluidBoundary  # noqa
)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport, PowerLawTransport  # noqa
from mirgecom.discretization import create_discretization_collection
import grudge.op as op
from grudge.dof_desc import BoundaryDomainTag
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from meshmode.dof_array import DOFArray
import pymbolic as pmbl
from mirgecom.symbolic import (
    diff as sym_diff,
    evaluate)
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from mirgecom.simutil import (
    compare_fluid_solutions,
    #    componentwise_norms
)
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from mirgecom.mpi import mpi_entry_point
# from functools import partial
import pyopencl as cl

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None, periodic=None):
    if periodic is None:
        periodic = (False,)*dim
    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t,
               periodic=periodic)


class _TestCommTag:
    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_overintegration=False, lazy=False, use_leap=False, use_profiling=False,
         casename=None, rst_filename=None):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()
    print(f"{rank=}/{num_parts=}")

    # from mirgecom.simutil import global_reduce as _global_reduce
    # global_reduce = partial(_global_reduce, comm=comm)

    # logmgr = initialize_logmgr(use_logmgr,
    #    filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    dim = 3
    order = 1
    sym_x = pmbl.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")

    gas_const = 287.
    gamma = 1.4
    # tn = 2./3.
    tn = 1.0
    alpha = 0.
    prandtl = .7
    temperature_ref = 300.0
    mu_ref = 1.852e-5
    beta = mu_ref / temperature_ref**tn
    # kappa_ref = gamma * gas_const * mu_ref / ((gamma - 1) * prandtl)
    sigma = gamma/prandtl

    nspecies = 0

    diffusivity = np.empty((0,), dtype=object)
    if nspecies > 0:
        diffusivity = np.zeros(nspecies)

    mu = 1.0
    kappa = gamma * gas_const * mu / ((gamma - 1) * prandtl)

    print(f"{gas_const=},{gamma=},mu{tn=},{alpha=}")
    print(f"{prandtl=},{mu_ref=},mu{beta=},{temperature_ref=}")
    print(f"{sigma=},{mu=},{kappa=}")

    eos = IdealSingleGas(gas_const=gas_const)
    # transport_model = SimpleTransport(viscosity=mu,
    #                                  thermal_conductivity=kappa)
    transport_model = PowerLawTransport(alpha=alpha, beta=beta, sigma=sigma,
                                        n=tn, species_diffusivity=diffusivity)
    gas_model = GasModel(eos=eos, transport=transport_model)

    from mms import MoserSolution
    man_soln = MoserSolution(dim=dim, lx=(4*np.pi, 2., 4*np.pi/3.),
                             alpha=beta, beta=alpha, sigma=sigma, mu=mu, kappa=kappa,
                             nspecies=nspecies, n=tn)

    sym_cv, sym_prs, sym_tmp = man_soln.get_solution(sym_x, sym_t)
    sym_mu, sym_kappa = man_soln.get_transport_properties(sym_cv, sym_prs, sym_tmp)

    # logger.info(f"{sym_cv=}\n"
    #            f"{sym_cv.mass=}\n"
    #            f"{sym_cv.energy=}\n"
    #            f"{sym_cv.momentum=}\n"
    #            f"{sym_cv.species_mass=}")

    dcv_dt = sym_diff(sym_t)(sym_cv)
    # print(f"{dcv_dt=}")

    from mirgecom.symbolic_fluid import sym_ns
    sym_ns_rhs = sym_ns(sym_cv, sym_prs, sym_tmp, mu=sym_mu, kappa=sym_kappa,
                        species_diffusivities=diffusivity)
    from pymbolic.mapper.analysis import get_num_nodes

    nnodes_mass = get_num_nodes(sym_ns_rhs.mass)
    nnodes_mom = get_num_nodes(sym_ns_rhs.momentum[0])
    nnodes_ener = get_num_nodes(sym_ns_rhs.energy)

    nnodes_dmassdt = get_num_nodes(dcv_dt.mass)
    nnodes_dmomdt = get_num_nodes(dcv_dt.momentum[0])
    nnodes_denerdt = get_num_nodes(dcv_dt.energy)

    print(f"{nnodes_mass=},{nnodes_mom=},{nnodes_ener=},{nnodes_dmassdt=},"
          f"{nnodes_dmomdt=},{nnodes_denerdt=}")

    # assert False

    sym_ns_source = dcv_dt - sym_ns_rhs

    tol = 1e-12

    sym_source = sym_ns_source

    # logger.info(f"{sym_source=}")

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    # n0 = 8

    def _evaluate_source(t, x):
        return evaluate(sym_source, t=t, x=x)

    def _evaluate_soln(t, x):
        return evaluate(sym_cv, t=t, x=x)

    # eval_source = actx.compile(_evaluate_source)
    # eval_soln = actx.compile(_evaluate_soln)
    periodic = (True, False, True)
    for n in [24, 36, 48]:

        mesh = man_soln.get_mesh(n, periodic=periodic)

        dcoll = create_discretization_collection(actx, mesh, order)
        nodes = actx.thaw(dcoll.nodes())

        from grudge.dt_utils import characteristic_lengthscales
        char_len = actx.to_numpy(
            op.norm(dcoll, characteristic_lengthscales(actx, dcoll), np.inf)
        )

        source_eval = evaluate(sym_source, t=0, x=nodes)
        # source_eval = eval_source(0, nodes)
        cv_exact = evaluate(sym_cv, t=0, x=nodes)
        # cv_exact = eval_soln(0, nodes)
        import time as perftime
        t0 = perftime.perf_counter()
        # print(f"{source_eval=}")
        t1 = perftime.perf_counter() - t0
        # print(f"{cv_exact=}")
        perftime.perf_counter()
        t2 = perftime.perf_counter() - t1
        print(f"{t1=},{t2=}")

        # Sanity check the dependent quantities
        tmp_exact = evaluate(sym_tmp, t=0, x=nodes)
        tmp_eos = eos.temperature(cv=cv_exact)
        prs_exact = evaluate(sym_prs, t=0, x=nodes)
        prs_eos = eos.pressure(cv=cv_exact)
        prs_resid = (prs_exact - prs_eos)/prs_exact
        tmp_resid = (tmp_exact - tmp_eos)/tmp_exact
        prs_err = actx.to_numpy(op.norm(dcoll, prs_resid, np.inf))
        tmp_err = actx.to_numpy(op.norm(dcoll, tmp_resid, np.inf))

        # print(f"{prs_exact=}\n{prs_eos=}")
        # print(f"{tmp_exact=}\n{tmp_eos=}")

        assert prs_err < tol
        assert tmp_err < tol

        if isinstance(source_eval.mass, DOFArray):
            from mirgecom.simutil import componentwise_norms
            source_norms = componentwise_norms(dcoll, source_eval)
        else:
            source_norms = source_eval

        logger.info(f"{source_norms=}")
        # logger.info(f"{source_eval=}")

        def _boundary_state_func(dcoll, dd_bdry, gas_model,
                                 state_minus, time=0, **kwargs):
            actx = state_minus.array_context
            bnd_discr = dcoll.discr_from_dd(dd_bdry)
            nodes = actx.thaw(bnd_discr.nodes())
            t0 = perftime.perf_counter()
            boundary_cv = evaluate(sym_cv, x=nodes, t=time)
            t_bnd = perftime.perf_counter() - t0
            print(f"{t_bnd=}")
            # boundary_cv = eval_soln(time, nodes)
            return make_fluid_state(boundary_cv, gas_model)

        boundaries = {
            BoundaryDomainTag("-2"): IsothermalNoSlipBoundary(wall_temperature=300.),
            BoundaryDomainTag("+2"): IsothermalNoSlipBoundary(wall_temperature=300.)
        }

        from mirgecom.simutil import max_component_norm
        err_scale = max_component_norm(dcoll, cv_exact)

        def get_rhs(t, cv):
            from mirgecom.gas_model import make_fluid_state
            fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
            # source = eval_source(t, nodes)
            t0 = perftime.perf_counter()
            mms_source = evaluate(sym_source, t=t, x=nodes)  # noqa
            t_src = perftime.perf_counter() - t0
            t0 = perftime.perf_counter()
            print(f"perf {t_src=}")
            # assert False
            ns_rhs = \
                ns_operator(dcoll, boundaries=boundaries, state=fluid_state,  # noqa
                            gas_model=gas_model, comm_tag=_TestCommTag)
            t_rhs = perftime.perf_counter() - t0
            print(f"perf {t_rhs=}")
            # print(f"{ns_rhs=}")
            # print(f"{mms_source=}")
            fluid_rhs = ns_rhs + mms_source
            print(f"{max_component_norm(dcoll, fluid_rhs/err_scale)=}")  # noqa            
            print(f"{max_component_norm(dcoll, ns_rhs/err_scale)=}")  # noqa
            print(f"{max_component_norm(dcoll, mms_source/err_scale)=}")  # noqa
            return fluid_rhs

        t = 0.
        from mirgecom.integrators import rk4_step
        dt = 1e-6
        nsteps = 10
        cv = cv_exact
        # print(f"{cv.dim=}")
        # print(f"{cv=}")

        for iloop in range(nsteps):
            print(f"{iloop=}")
            cv = rk4_step(cv, t, dt, get_rhs)
            t += dt

        cv_exact = evaluate(sym_cv, t=t, x=nodes)
        soln_resid = compare_fluid_solutions(dcoll, cv, cv_exact)
        cv_err_scales = componentwise_norms(dcoll, cv_exact)

        max_err = soln_resid[0]/cv_err_scales.mass
        max_err_mass = actx.to_numpy(max_err)
        print(f"mass{max_err_mass=}")
        max_err_ener = soln_resid[1]/cv_err_scales.energy
        max_err = max(max_err, max_err_ener)
        max_err_ener = actx.to_numpy(max_err_ener)
        print(f"ener{max_err_ener=}")
        for i in range(dim):
            max_err_mom = soln_resid[2+i]/cv_err_scales.momentum[i]
            max_err = max(max_err, max_err_mom)
            max_err_mom = actx.to_numpy(max_err_mom)
            print(f"mom{i=}{max_err=}")

        max_err = actx.to_numpy(max_err)
        print(f"{max_err=}")

        eoc_rec.add_data_point(char_len, max_err)

    logger.info(eoc_rec.pretty_print())

    assert (
        eoc_rec.order_estimate() >= order - 0.5
        or eoc_rec.max_error() < tol
    )


if __name__ == "__main__":
    import argparse
    casename = "pulse"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    lazy = args.lazy
    if args.profiling:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_logmgr=args.log, use_overintegration=args.overintegration,
         use_leap=args.leap, use_profiling=args.profiling, lazy=lazy,
         casename=casename, rst_filename=rst_filename)
