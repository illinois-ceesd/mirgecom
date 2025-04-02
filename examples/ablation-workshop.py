r"""Demonstrate Ablation Workshop case \#2.1."""

__copyright__ = "Copyright (C) 2023 University of Illinois Board of Trustees"

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
import gc
import numpy as np
import scipy  # type: ignore[import-untyped]
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from meshmode.dof_array import DOFArray
from meshmode.discretization.connection import FACE_RESTR_ALL

from grudge.trace_pair import (
    TracePair, interior_trace_pairs, tracepair_with_discr_tag
)
from grudge import op
import grudge.geometry as geo
from grudge.dof_desc import DD_VOLUME_ALL
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    BoundaryDomainTag,
    DISCR_TAG_BASE
)
from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import ssprk43_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    PrescribedFluxDiffusionBoundary,
    NeumannDiffusionBoundary
)
from mirgecom.simutil import (
    check_naninf_local,
    generate_and_distribute_mesh,
    write_visfile,
    check_step
)
from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage
)
from mirgecom.eos import (
    MixtureDependentVars,
    MixtureEOS,
    GasEOS,
    GasDependentVars
)
from mirgecom.transport import TransportModel
from mirgecom.gas_model import PorousFlowFluidState
from mirgecom.wall_model import (
    PorousWallVars,
    PorousFlowModel as BasePorousFlowModel,
    PorousWallTransport
)
from mirgecom.fluid import ConservedVars
from mirgecom.materials.tacot import TacotEOS as OriginalTacotEOS
from logpyle import IntervalTimer, set_dt
from typing import Optional, Union
from pytools.obj_array import make_obj_array

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class _MyGradTag_f:  # noqa N801
    pass


class _MyGradTag_u:  # noqa N801
    pass


class _PresDiffTag:
    pass


class _TempDiffTag:
    pass


def initializer(dim, gas_model, material_densities, temperature,
                gas_density=None, pressure=None):
    """Initialize state of composite material.

    Parameters
    ----------
    gas_model
        :class:`mirgecom.gas_model.GasModel`

    material_densities: numpy.ndarray
        The initial bulk density of each one of the resin constituents.
        It has shape ``(nphase,)``

    temperature: :class:`~meshmode.dof_array.DOFArray`
        The initial temperature of the gas+solid

    gas_density: :class:`~meshmode.dof_array.DOFArray`
        Optional argument with the gas density. If not provided, the pressure
        will be used to evaluate the density.

    pressure: :class:`~meshmode.dof_array.DOFArray`
        Optional argument with the gas pressure. It will be used to evaluate
        the gas density.

    Returns
    -------
    cv: :class:`mirgecom.fluid.ConservedVars`
        The conserved variables of the fluid permeating the porous wall.
    """
    if gas_density is None and pressure is None:
        raise ValueError("Must specify one of 'gas_density' or 'pressure'")

    if not isinstance(temperature, DOFArray):
        raise TypeError("Temperature does not have the proper shape")

    actx = temperature.array_context

    tau = gas_model.decomposition_progress(material_densities)

    gas_const = gas_model.eos.gas_const(cv=None, temperature=temperature)

    eps_rho_gas = 1
    if gas_density is None:
        eps_gas = gas_model.wall_eos.void_fraction(tau)
        eps_rho_gas = eps_gas*pressure/(gas_const*temperature)

    # internal energy (kinetic energy is neglected)
    eps_rho_solid = sum(material_densities)
    bulk_energy = (
        eps_rho_gas*gas_model.eos.get_internal_energy(temperature=temperature)
        + eps_rho_solid*gas_model.wall_eos.enthalpy(temperature, tau)
    )

    zeros = actx.np.zeros_like(tau)
    momentum = make_obj_array([zeros for i in range(dim)])

    return ConservedVars(mass=eps_rho_gas, energy=bulk_energy, momentum=momentum)


def eval_spline(x, x_bnds, coeffs) -> DOFArray:
    r"""Evaluate spline $a(x-x_i)^3 + b(x-x_i)^2 + c(x-x_i) + d$.

    Parameters
    ----------
    x: :class:`~meshmode.dof_array.DOFArray`
        The value where $f(x)$ will be evaluated.
    x_bnds: :class:`numpy.ndarray` with shape ``(m,)``
        The $m$ nodes $x_i$ for the different segments of the spline.
    coeffs: :class:`numpy.ndarray` with shape ``(4,m)``
        The 4 coefficients for each segment $i$ of the spline.
    """
    actx = x.array_context

    val = actx.np.zeros_like(x)
    for i in range(0, len(x_bnds)-1):
        val = (
            actx.np.where(actx.np.less(x, x_bnds[i+1]),
            actx.np.where(actx.np.greater_equal(x, x_bnds[i]),
                coeffs[0, i]*(x-x_bnds[i])**3 + coeffs[1, i]*(x-x_bnds[i])**2
                + coeffs[2, i]*(x-x_bnds[i]) + coeffs[3, i], 0.0), 0.0)) + val

    return val


def eval_spline_derivative(x, x_bnds, coeffs) -> DOFArray:
    """Evaluate analytical derivative of a spline $3a(x-x_i)^2 + 2b(x-x_i) + c$.

    Parameters
    ----------
    x: :class:`~meshmode.dof_array.DOFArray`
        The value where $f(x)$ will be evaluatead.
    x_bnds: :class:`numpy.ndarray` with shape ``(m,)``
        The $m$ nodes $x_i$ for the different segments of the spline.
    coeffs: :class:`numpy.ndarray` with shape ``(4,m)``
        The 4 coefficients for each segment $i$ of the spline.
    """
    actx = x.array_context

    val = actx.np.zeros_like(x)
    for i in range(0, len(x_bnds)-1):
        val = (
            actx.np.where(actx.np.less(x, x_bnds[i+1]),
            actx.np.where(actx.np.greater_equal(x, x_bnds[i]),
                3.0*coeffs[0, i]*(x-x_bnds[i])**2 + 2.0*coeffs[1, i]*(x-x_bnds[i])
                + coeffs[2, i], 0.0), 0.0)) + val

    return val


class BprimeTable:
    """Class containing the table for energy balance at the surface.

    This class is only required for uncoupled cases, where only the wall
    portion is evaluated. This is NOT used for fully-coupled cases.
    """

    T_bounds: np.ndarray
    B_bounds: np.ndarray
    Bc: np.ndarray
    Hw: np.ndarray
    Hw_cs: np.ndarray

    def __init__(self):

        import mirgecom
        path = mirgecom.__path__[0] + "/materials/aw_Bprime.dat"
        bprime_table = \
            (np.genfromtxt(path, skip_header=1)[:, 2:6]).reshape((25, 151, 4))

        # bprime contains: B_g, B_c, Temperature T, Wall enthalpy H_W
        self.T_bounds = bprime_table[0, :-1:6, 2]
        self.B_bounds = bprime_table[::-1, 0, 0]
        self.Bc = bprime_table[::-1, :, 1]
        self.Hw = bprime_table[::-1, :-1:6, 3]

        # create spline to interpolate the wall enthalpy
        self.Hw_cs = np.zeros((25, 4, 24))
        for i in range(0, 25):
            self.Hw_cs[i, :, :] = \
                scipy.interpolate.CubicSpline(self.T_bounds, self.Hw[i, :]).c


class GasTabulatedTransport(TransportModel):
    """Evaluate tabulated transport data for TACOT."""

    def __init__(self, prandtl=1.0, lewis=1.0):
        """Return gas tabulated data and interpolating functions.

        Parameters
        ----------
        prandtl: float
            the Prandtl number of the mixture. Defaults to 1.
        lewis: float
            the Lewis number of the mixture. Defaults to 1.
        """
        self._prandtl = prandtl
        self._lewis = lewis

        #    T     ,  viscosity
        gas_data = np.array([
            [200.00,  0.086881],
            [350.00,  0.144380],
            [500.00,  0.196150],
            [650.00,  0.243230],
            [700.00,  0.258610],
            [750.00,  0.274430],
            [800.00,  0.290920],
            [850.00,  0.307610],
            [900.00,  0.323490],
            [975.00,  0.344350],
            [1025.0,  0.356630],
            [1100.0,  0.373980],
            [1150.0,  0.385360],
            [1175.0,  0.391330],
            [1200.0,  0.397930],
            [1275.0,  0.421190],
            [1400.0,  0.458870],
            [1525.0,  0.483230],
            [1575.0,  0.487980],
            [1625.0,  0.491950],
            [1700.0,  0.502120],
            [1775.0,  0.516020],
            [1925.0,  0.545280],
            [2000.0,  0.559860],
            [2150.0,  0.588820],
            [2300.0,  0.617610],
            [2450.0,  0.646380],
            [2600.0,  0.675410],
            [2750.0,  0.705000],
            [2900.0,  0.735570],
            [3050.0,  0.767590],
            [3200.0,  0.801520],
            [3350.0,  0.837430],
        ])

        self._cs_viscosity = CubicSpline(gas_data[:, 0], gas_data[:, 1]*1e-4)

    def bulk_viscosity(self, cv: ConservedVars,  # type: ignore[override]
            dv: GasDependentVars, eos: GasEOS) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$."""
        actx = cv.mass.array_context
        return actx.np.zeros_like(cv.mass)

    def volume_viscosity(self, cv: ConservedVars,  # type: ignore[override]
            dv: GasDependentVars, eos: GasEOS) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$."""
        return (self.bulk_viscosity(cv, dv, eos)
                - 2./3.)*self.viscosity(cv, dv, eos)

    def viscosity(self, cv: ConservedVars,  # type: ignore[override]
            dv: GasDependentVars, eos: GasEOS) -> DOFArray:
        r"""Return the gas viscosity $\mu$."""
        coeffs = self._cs_viscosity.c
        bnds = self._cs_viscosity.x
        return eval_spline(dv.temperature, bnds, coeffs)

    def thermal_conductivity(self, cv: ConservedVars,  # type: ignore[override]
            dv: GasDependentVars, eos: GasEOS) -> DOFArray:
        r"""Return the gas thermal conductivity $\kappa_g$.

        .. math::
            \kappa = \frac{\mu C_p}{Pr}

        with gas viscosity $\mu$, heat capacity at constant pressure $C_p$
        and the Prandtl number $Pr$ (default to 1).
        """
        cp = eos.heat_capacity_cp(cv, dv.temperature)
        mu = self.viscosity(cv, dv, eos)
        return mu*cp/self._prandtl

    def species_diffusivity(self, cv: ConservedVars,  # type: ignore[override]
            dv: GasDependentVars, eos: GasEOS) -> DOFArray:
        """Return the (empty) species mass diffusivities."""
        return cv.species_mass  # type: ignore


class TabulatedGasEOS(MixtureEOS):
    """Simplified model of the pyrolysis gas using tabulated data.

    This section is to be used when species conservation is not employed and
    the output gas is assumed to be in chemical equilibrium.
    The table was extracted from the ablation workshop suplementary material.
    Some lines were removed to reduce the number of spline interpolation segments.
    """

    def __init__(self):
        """Return gas tabulated data and interpolating functions."""

        #    T     , M      ,  Cp    , gamma  ,  enthalpy
        gas_data = np.array([
            [200.00,  21.996,  1.5119,  1.3334,  -7246.50],
            [350.00,  21.995,  1.7259,  1.2807,  -7006.30],
            [500.00,  21.948,  2.2411,  1.2133,  -6715.20],
            [650.00,  21.418,  4.3012,  1.1440,  -6265.70],
            [700.00,  20.890,  6.3506,  1.1242,  -6004.60],
            [750.00,  19.990,  9.7476,  1.1131,  -5607.70],
            [800.00,  18.644,  14.029,  1.1116,  -5014.40],
            [850.00,  17.004,  17.437,  1.1171,  -4218.50],
            [900.00,  15.457,  17.009,  1.1283,  -3335.30],
            [975.00,  14.119,  8.5576,  1.1620,  -2352.90],
            [1025.0,  13.854,  4.7840,  1.1992,  -2034.20],
            [1100.0,  13.763,  3.5092,  1.2240,  -1741.20],
            [1150.0,  13.737,  3.9008,  1.2087,  -1560.90],
            [1175.0,  13.706,  4.8067,  1.1899,  -1453.50],
            [1200.0,  13.639,  6.2353,  1.1737,  -1315.90],
            [1275.0,  13.256,  8.4790,  1.1633,  -739.700],
            [1400.0,  12.580,  9.0239,  1.1583,  353.3100],
            [1525.0,  11.982,  11.516,  1.1377,  1608.400],
            [1575.0,  11.732,  12.531,  1.1349,  2214.000],
            [1625.0,  11.495,  11.514,  1.1444,  2826.800],
            [1700.0,  11.255,  7.3383,  1.1849,  3529.400],
            [1775.0,  11.139,  5.3118,  1.2195,  3991.000],
            [1925.0,  11.046,  4.2004,  1.2453,  4681.800],
            [2000.0,  11.024,  4.0784,  1.2467,  4991.300],
            [2150.0,  10.995,  4.1688,  1.2382,  5605.400],
            [2300.0,  10.963,  4.5727,  1.2214,  6257.300],
            [2450.0,  10.914,  5.3049,  1.2012,  6993.500],
            [2600.0,  10.832,  6.4546,  1.1815,  7869.600],
            [2750.0,  10.701,  8.1450,  1.1650,  8956.900],
            [2900.0,  10.503,  10.524,  1.1528,  10347.00],
            [3050.0,  10.221,  13.755,  1.1449,  12157.00],
            [3200.0,  9.8394,  17.957,  1.1408,  14523.00],
            [3350.0,  9.3574,  22.944,  1.1401,  17584.00],
        ])

        self._cs_molar_mass = CubicSpline(gas_data[:, 0], gas_data[:, 1])
        self._cs_gamma = CubicSpline(gas_data[:, 0], gas_data[:, 3])
        self._cs_enthalpy = CubicSpline(gas_data[:, 0], gas_data[:, 4]*1000.0)

    def enthalpy(self, temperature: DOFArray) -> DOFArray:
        r"""Return the gas enthalpy $h_g$."""
        coeffs = self._cs_enthalpy.c
        bnds = self._cs_enthalpy.x
        return eval_spline(temperature, bnds, coeffs)

    def get_internal_energy(self, temperature: DOFArray,
                            species_mass_fractions=None) -> DOFArray:
        r"""Evaluate the gas internal energy $e$.

        It is evaluated based on the tabulated enthalpy and molar mass as

        .. math::
            e(T) = h(T) - \frac{R}{M} T
        """
        gas_const = self.gas_const(cv=None, temperature=temperature)
        return self.enthalpy(temperature) - gas_const*temperature

    def heat_capacity_cp(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Return the gas heat capacity at constant pressure $C_{p_g}$.

        The heat capacity is the derivative of the enthalpy. Thus, to improve
        accuracy and avoid issues with Newton iteration, this is computed
        exactly as the analytical derivative of the spline for the enthalpy.
        """
        coeffs = self._cs_enthalpy.c
        bnds = self._cs_enthalpy.x
        return eval_spline_derivative(temperature, bnds, coeffs)

    def heat_capacity_cv(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Return the gas heat capacity at constant volume $C_{v_g}$."""
        mcp = self.heat_capacity_cp(cv, temperature)
        return mcp/self.gamma(cv, temperature)

    def gamma(self, cv: Optional[ConservedVars] = None,
              temperature: Optional[DOFArray] = None) -> DOFArray:
        r"""Return the heat of capacity ratios $\gamma$."""
        coeffs = self._cs_gamma.c
        bnds = self._cs_gamma.x
        return eval_spline(temperature, bnds, coeffs)

    def pressure(self, cv: ConservedVars, temperature: DOFArray) -> DOFArray:
        r"""Return the gas pressure.

        .. math::
            P = \frac{\epsilon_g \rho_g}{\epsilon_g} \frac{R}{M} T
        """
        gas_const = self.gas_const(cv, temperature)
        return cv.mass*gas_const*temperature

    def gas_const(self, cv: Optional[ConservedVars] = None,
                  temperature: Optional[DOFArray] = None,
                  species_mass_fractions: Optional[np.ndarray] = None) -> DOFArray:
        """Return the specific gas constant."""
        coeffs = self._cs_molar_mass.c
        bnds = self._cs_molar_mass.x
        molar_mass = eval_spline(temperature, bnds, coeffs)
        return 8314.46261815324/molar_mass

    def temperature(self, cv: ConservedVars, temperature_seed=None):
        raise NotImplementedError

    def sound_speed(self, cv: ConservedVars, temperature: DOFArray):
        raise NotImplementedError

    def internal_energy(self, cv: ConservedVars):
        raise NotImplementedError

    def total_energy(self, cv: ConservedVars, pressure: DOFArray,
                     temperature: DOFArray):
        raise NotImplementedError

    def kinetic_energy(self, cv: ConservedVars):
        raise NotImplementedError

    def get_temperature_seed(self, ary: Optional[DOFArray] = None,
            temperature_seed: Optional[Union[float, DOFArray]] = None) -> DOFArray:
        raise NotImplementedError

    def get_density(self, pressure, temperature, species_mass_fractions):
        raise NotImplementedError

    def get_species_molecular_weights(self):
        raise NotImplementedError

    def species_enthalpies(self, cv: ConservedVars, temperature: DOFArray):
        raise NotImplementedError

    def get_production_rates(self, cv: ConservedVars, temperature: DOFArray):
        raise NotImplementedError

    def get_species_source_terms(self, cv: ConservedVars, temperature: DOFArray):
        raise NotImplementedError

    def dependent_vars(self, cv: ConservedVars, temperature_seed=None,
            smoothness_mu=None, smoothness_kappa=None,
            smoothness_d=None, smoothness_beta=None):
        raise NotImplementedError


class PorousFlowModel(BasePorousFlowModel):
    """EOS for wall using tabulated data.

    Inherits :mod:`~mirgecom.wall_model.PorousFlowModel` and add an
    temperature-evaluation function exclusive for TACOT-tabulated data.
    """

    def get_temperature(self, cv, wv, tseed, niter=3):
        r"""Evaluate the temperature based on solid+gas properties.

        It uses the assumption of thermal equilibrium between solid and fluid.
        Newton iteration is used to get the temperature based on the internal
        energy/enthalpy and heat capacity for the bulk (solid+gas) material:

        .. math::
            T^{n+1} = T^n -
                \frac
                {\epsilon_g \rho_g e_g + \rho_s h_s - \rho e}
                {\epsilon_g \rho_g C_{v_g} + \epsilon_s \rho_s C_{p_s}}

        Note that kinetic energy is neglected is this case.

        Parameters
        ----------
        cv: ConservedVars

            The fluid conserved variables

        wv: PorousWallVars

            Wall properties as a function of decomposition progress

        tseed:

            Temperature to use as a seed for Netwon iteration

        Returns
        -------
        temperature: meshmode.dof_array.DOFArray

            The temperature of the gas+solid

        """
        if isinstance(tseed, DOFArray) is False:
            actx = cv.array_context
            temp = tseed + actx.np.zeros_like(cv.mass)
        else:
            temp = tseed

        for _ in range(0, niter):
            eps_rho_e = (cv.mass*self.eos.get_internal_energy(temperature=temp)
                         + wv.density*self.wall_eos.enthalpy(temp, wv.tau))
            bulk_cp = (cv.mass*self.eos.heat_capacity_cv(cv=cv, temperature=temp)
                       + wv.density*self.wall_eos.heat_capacity(temp, wv.tau))
            temp = temp - (eps_rho_e - cv.energy)/bulk_cp

        return temp

    def pressure_diffusivity(self, cv: ConservedVars, wv: PorousWallVars,
                             viscosity: DOFArray) -> DOFArray:
        r"""Return the pressure diffusivity for Darcy flow.

        .. math::
            d_{P} = \epsilon_g \rho_g \frac{\mathbf{K}}{\mu \epsilon_g}

        where $\mu$ is the gas viscosity, $\epsilon_g$ is the void fraction
        and $\mathbf{K}$ is the permeability matrix.
        """
        return cv.mass*wv.permeability/(viscosity*wv.void_fraction)


class TacotEOS(OriginalTacotEOS):
    """Inherits and modified the original TACOT material."""

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the composite material."""
        virgin = 1.6e-11
        char = 2.0e-11
        return virgin*tau + char*(1.0 - tau)


def binary_sum(ary):
    """Sum the elements of an array, creating a log-depth DAG instead of linear."""
    n = len(ary)
    if n == 1:
        return ary[0]
    return binary_sum(ary[:n//2]) + binary_sum(ary[n//2:])


@mpi_entry_point
def main(actx_class=None, use_logmgr=True, casename=None, restart_file=None):
    """Demonstrate the ablation workshop test case 2.1."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename="ablation.sqlite", mode="wo", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=True)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    viz_path = "viz_data/"
    vizname = viz_path+casename

    t_final = 2.0e-7

    dim = 1

    order = 3
    dt = 2.0e-8
    pressure_scaling_factor = 1.0  # noqa N806

    nviz = 200
    ngarbage = 50
    nrestart = 10000
    nhealth = 10

    current_dt = dt/pressure_scaling_factor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    rst_pattern = rst_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    if restart_file:  # read the grid from restart data
        rst_filename = f"{restart_file}"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts

    else:  # generate the grid from scratch
        from functools import partial
        nel_1d = 121

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(0.0,)*dim, b=(0.05,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={"prescribed": ["+x"], "neumann": ["-x"]})

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    quadrature_tag = DISCR_TAG_BASE

    nodes = actx.thaw(dcoll.nodes())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if rank == 0:
        print("----- Discretization info ----")
        #  print(f"Discr: {nodes.shape=}, {order=}, {h_min=}, {h_max=}")
    for i in range(nparts):
        if rank == i:
            print(f"{rank=},{local_nelements=},{global_nelements=}")
        comm.Barrier()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pressure_boundaries = {
        BoundaryDomainTag("prescribed"): DirichletDiffusionBoundary(101325.0),
        BoundaryDomainTag("neumann"): NeumannDiffusionBoundary(0.0)}

    def my_presc_bdry(u_minus):
        return +u_minus

    def my_wall_bdry(u_minus):
        return -u_minus

    velocity_boundaries = {BoundaryDomainTag("prescribed"): my_presc_bdry,
                           BoundaryDomainTag("neumann"): my_wall_bdry}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # {{{ Initialize flow model

    import mirgecom.materials.tacot as my_composite

    my_gas = TabulatedGasEOS()
    bprime_class = BprimeTable()
    my_material = TacotEOS(char_mass=220.0, virgin_mass=280.0)
    pyrolysis = my_composite.Pyrolysis()

    base_transport = GasTabulatedTransport()
    sample_transport = PorousWallTransport(base_transport=base_transport)
    gas_model = PorousFlowModel(eos=my_gas, wall_eos=my_material,
                                transport=sample_transport)

    # }}}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # soln setup and init
    material_densities = np.empty((3,), dtype=object)

    zeros = actx.np.zeros_like(nodes[0])
    material_densities[0] = 30.0 + zeros
    material_densities[1] = 90.0 + zeros
    material_densities[2] = 160. + zeros
    temperature = 300.0 + zeros
    pressure = 101325.0 + zeros

    pressure = force_evaluation(actx, pressure)
    temperature = force_evaluation(actx, temperature)
    material_densities = force_evaluation(actx, material_densities)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    if restart_file:
        current_t = restart_data["t"]
        istep = restart_data["step"]
        cv = restart_data["cv"]
        material_densities = restart_data["material_densities"]
        first_step = istep + 0
    else:
        current_t = 0.0
        istep = 0
        first_step = 0
        cv = initializer(dim=dim, gas_model=gas_model,
            material_densities=material_densities,
            pressure=pressure, temperature=temperature)

    # stand-alone version of the "gas_model" to bypass some unnecessary
    # variables for this particular case
    def make_state(cv, temperature_seed, material_densities):
        """Return the fluid+wall state for porous media flow.

        Ideally one would use the gas_model.make_fluid_state but, since this
        case use tabulated data and equilibrium gas assumption, it was
        implemented in this stand-alone function. Note that some functions
        have slightly different calls and the absence of species.
        """
        zeros = actx.np.zeros_like(cv.mass)

        tau = gas_model.decomposition_progress(material_densities)
        wv = PorousWallVars(
            material_densities=material_densities,
            tau=tau,
            density=gas_model.solid_density(material_densities),
            void_fraction=gas_model.wall_eos.void_fraction(tau=tau),
            permeability=gas_model.wall_eos.permeability(tau=tau),
            tortuosity=gas_model.wall_eos.tortuosity(tau=tau)
        )

        temperature = gas_model.get_temperature(cv=cv, wv=wv,
                                                tseed=temperature_seed)

        pressure = gas_model.get_pressure(cv=cv, wv=wv, temperature=temperature)

        dv = MixtureDependentVars(
            temperature=temperature,
            pressure=pressure,
            speed_of_sound=zeros,
            smoothness_mu=zeros,
            smoothness_kappa=zeros,
            smoothness_beta=zeros,
            smoothness_d=zeros,
            species_enthalpies=cv.species_mass,  # empty array
        )

        tv = gas_model.transport.transport_vars(
            cv=cv, dv=dv, wv=wv, eos=gas_model.eos, wall_eos=gas_model.wall_eos)

        return PorousFlowFluidState(cv=cv, dv=dv, tv=tv, wv=wv)

    compiled_make_state = actx.compile(make_state)

    cv = force_evaluation(actx, cv)
    temperature_seed = force_evaluation(actx,
                                        actx.np.zeros_like(nodes[0]) + 300.0)
    fluid_state = compiled_make_state(cv, temperature_seed, material_densities)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    gc_timer = None
    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, istep, current_t)

        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value:8d}, "),
            ("dt.max", "dt: {value:1.3e} s, "),
            ("t_sim.max", "sim time: {value:12.8f} s, "),
            ("t_step.max", "step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer_init = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer_init)
        gc_timer = gc_timer_init.get_sub_timer()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def compute_div(actx, dcoll, quadrature_tag, field, velocity,
                    boundaries, dd_vol):
        r"""Return divergence for the inviscid term in energy equation.

        .. math::
            \frac{\partial \rho u_i h}{\partial x_i}
        """
        dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

        f_quad = op.project(dcoll, dd_vol, dd_vol_quad, field)
        u_quad = op.project(dcoll, dd_vol, dd_vol_quad, velocity)
        flux_quad = f_quad*u_quad

        itp_f_quad = op.project(dcoll, dd_vol, dd_vol_quad,
                                interior_trace_pairs(dcoll, field, volume_dd=dd_vol,
                                                     comm_tag=_MyGradTag_f))

        itp_u_quad = op.project(dcoll, dd_vol, dd_vol_quad,
                                interior_trace_pairs(dcoll, velocity,
                                                     volume_dd=dd_vol,
                                                     comm_tag=_MyGradTag_u))

        def interior_flux(f_tpair, u_tpair):
            dd_trace_quad = f_tpair.dd.with_discr_tag(quadrature_tag)
            normal_quad = geo.normal(actx, dcoll, dd_trace_quad)

            bnd_u_tpair_quad = \
                tracepair_with_discr_tag(dcoll, quadrature_tag, u_tpair)
            bnd_f_tpair_quad = \
                tracepair_with_discr_tag(dcoll, quadrature_tag, f_tpair)

            wavespeed_int = \
                actx.np.sqrt(np.dot(bnd_u_tpair_quad.int, bnd_u_tpair_quad.int))
            wavespeed_ext = \
                actx.np.sqrt(np.dot(bnd_u_tpair_quad.ext, bnd_u_tpair_quad.ext))

            lmbda = actx.np.maximum(wavespeed_int, wavespeed_ext)
            jump = bnd_f_tpair_quad.int - bnd_f_tpair_quad.ext
            numerical_flux = (bnd_f_tpair_quad*bnd_u_tpair_quad).avg + 0.5*lmbda*jump

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad,
                              numerical_flux@normal_quad)

        def boundary_flux(bdtag, bdry_cond_function):
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = geo.normal(actx, dcoll, dd_bdry_quad)

            int_flux_quad = op.project(dcoll, dd_vol_quad, dd_bdry_quad, flux_quad)
            ext_flux_quad = bdry_cond_function(int_flux_quad)

            bnd_tpair = TracePair(dd_bdry_quad,
                interior=int_flux_quad, exterior=ext_flux_quad)

            return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad,
                              bnd_tpair.avg@normal_quad)

        # pylint: disable=invalid-unary-operand-type
        return -op.inverse_mass(
            dcoll, dd_vol_quad,
            op.weak_local_div(dcoll, dd_vol_quad, flux_quad)
            - op.face_mass(dcoll, dd_allfaces_quad,
                (sum(interior_flux(f_tpair, u_tpair) for f_tpair, u_tpair in
                    zip(itp_f_quad, itp_u_quad))
                + sum(boundary_flux(bdtag, bdry) for bdtag, bdry in
                    boundaries.items()))
            )
        )

    def ablation_workshop_flux(dcoll, state, gas_model, velocity, bprime_class,
                               quadrature_tag, dd_wall, time):
        """Evaluate the prescribed heat flux to be applied at the boundary.

        Function specific for verification against the ablation workshop test
        case 2.1.
        """
        cv = state.cv
        dv = state.dv
        wv = state.wv

        actx = cv.mass.array_context

        # restrict variables to the domain boundary
        dd_vol_quad = dd_wall.with_discr_tag(quadrature_tag)
        bdtag = dd_wall.trace("prescribed").domain_tag
        normal_vec = geo.normal(actx, dcoll, bdtag)
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)

        temperature_bc = op.project(dcoll, dd_wall, dd_bdry_quad, dv.temperature)
        momentum_bc = op.project(dcoll, dd_wall, dd_bdry_quad, cv.mass*velocity)
        m_dot_g = np.dot(momentum_bc, normal_vec)

        # time-dependent function
        weight = actx.np.where(actx.np.less(time, 0.1), (time/0.1)+1e-7, 1.0)

        h_e = 1.5e6*weight
        conv_coeff_0 = 0.3*weight

        # this is only valid for the non-ablative case (2.1)
        m_dot_c = 0.0
        m_dot = m_dot_g + m_dot_c + 1e-13

        # ~~~~
        # blowing correction: few iterations to converge the coefficient
        conv_coeff = conv_coeff_0*1.0
        lambda_corr = 0.5
        for _ in range(0, 3):
            phi = 2.0*lambda_corr*m_dot/conv_coeff
            blowing_correction = phi/(actx.np.exp(phi) - 1.0)
            conv_coeff = conv_coeff_0*blowing_correction

        Bsurf = m_dot_g/conv_coeff  # noqa N806

        # ~~~~
        # get the wall enthalpy using spline interpolation
        bnds_T = bprime_class.T_bounds  # noqa N806
        bnds_B = bprime_class.B_bounds  # noqa N806

        # using spline for temperature interpolation
        # while using "nearest neighbor" for the "B" parameter
        h_w_comps = make_obj_array([actx.zeros((), dtype=np.float64)]*24*15)
        i = 0
        for j in range(0, 24):
            for k in range(0, 15):
                h_w_comps[i] = \
                    actx.np.where(actx.np.greater_equal(temperature_bc, bnds_T[j]),
                    actx.np.where(actx.np.less(temperature_bc, bnds_T[j+1]),
                    actx.np.where(actx.np.greater_equal(Bsurf, bnds_B[k]),
                    actx.np.where(actx.np.less(Bsurf, bnds_B[k+1]),
                          bprime_class.Hw_cs[k, 0, j]*(temperature_bc-bnds_T[j])**3
                        + bprime_class.Hw_cs[k, 1, j]*(temperature_bc-bnds_T[j])**2
                        + bprime_class.Hw_cs[k, 2, j]*(temperature_bc-bnds_T[j])
                        + bprime_class.Hw_cs[k, 3, j], 0.0), 0.0), 0.0), 0.0)
                i += 1

        h_w = binary_sum(h_w_comps)

        h_g = gas_model.eos.enthalpy(temperature_bc)

        flux = -(conv_coeff*(h_e - h_w) - m_dot*h_w + m_dot_g*h_g)

        tau_bc = op.project(dcoll, dd_wall, dd_bdry_quad, wv.tau)
        emissivity = my_material.emissivity(tau=tau_bc)
        radiation = emissivity*5.67e-8*(temperature_bc**4 - 300**4)

        # this is the physical flux normal to the boundary
        return flux + radiation

    def phenolics_operator(dcoll, fluid_state, boundaries, gas_model, pyrolysis,
                           quadrature_tag, dd_wall=DD_VOLUME_ALL, time=0.0,
                           bprime_class=None, pressure_scaling_factor=1.0,
                           penalty_amount=1.0):
        """Return the RHS of the composite wall."""
        cv = fluid_state.cv
        dv = fluid_state.dv
        tv = fluid_state.tv
        wv = fluid_state.wv

        actx = cv.array_context
        zeros = actx.np.zeros_like(wv.tau)

        pressure_boundaries, velocity_boundaries = boundaries

        # pressure diffusivity for Darcy flow
        pressure_diffusivity = gas_model.pressure_diffusivity(cv, wv, tv.viscosity)

        # ~~~~~
        # viscous RHS
        pressure_viscous_rhs, grad_pressure = diffusion_operator(dcoll,
            kappa=pressure_diffusivity,
            boundaries=pressure_boundaries, u=dv.pressure,
            penalty_amount=penalty_amount, return_grad_u=True,
            comm_tag=_PresDiffTag)

        velocity = -(pressure_diffusivity/cv.mass)*grad_pressure

        boundary_flux = ablation_workshop_flux(dcoll, fluid_state, gas_model,
            velocity, bprime_class, quadrature_tag, dd_wall, time)

        energy_boundaries = {
            BoundaryDomainTag("prescribed"):
                PrescribedFluxDiffusionBoundary(boundary_flux),
            BoundaryDomainTag("neumann"):
                NeumannDiffusionBoundary(0.0)
        }

        energy_viscous_rhs = diffusion_operator(dcoll,
            kappa=tv.thermal_conductivity, boundaries=energy_boundaries,
            u=dv.temperature, penalty_amount=penalty_amount,
            comm_tag=_TempDiffTag)

        viscous_rhs = ConservedVars(
            mass=pressure_scaling_factor*pressure_viscous_rhs,
            momentum=cv.momentum*0.0,
            energy=energy_viscous_rhs,
            species_mass=cv.species_mass)  # this should be empty in this case

        # ~~~~~
        # inviscid RHS, energy equation only
        field = cv.mass*gas_model.eos.enthalpy(temperature)
        energy_inviscid_rhs = compute_div(actx, dcoll, quadrature_tag, field,
                                          velocity, velocity_boundaries, dd_wall)

        inviscid_rhs = ConservedVars(
            mass=zeros,
            momentum=cv.momentum*0.0,
            energy=-energy_inviscid_rhs,
            species_mass=cv.species_mass)  # this should be empty in this case

        # ~~~~~
        # decomposition for each component of the resin
        resin_pyrolysis = pyrolysis.get_source_terms(temperature=dv.temperature,
                                                     chi=wv.material_densities)

        # flip sign due to mass conservation
        gas_source_term = -pressure_scaling_factor*sum(resin_pyrolysis)

        # viscous dissipation due to friction inside the porous
        visc_diss_energy = tv.viscosity*wv.void_fraction**2*(
            (1.0/wv.permeability)*np.dot(velocity, velocity))

        source_terms = ConservedVars(
            mass=gas_source_term,
            momentum=cv.momentum*0.0,
            energy=visc_diss_energy,
            species_mass=cv.species_mass)  # this should be empty in this case

        return inviscid_rhs + viscous_rhs + source_terms, resin_pyrolysis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    visualizer = make_visualizer(dcoll)

    def my_write_viz(step, t, state):

        cv = state.cv
        dv = state.dv
        wv = state.wv

        viz_fields = [("CV_density", cv.mass),
                      ("CV_energy", cv.energy),
                      ("DV_T", dv.temperature),
                      ("DV_P", dv.pressure),
                      ("WV_phase_1", wv.material_densities[0]),
                      ("WV_phase_2", wv.material_densities[1]),
                      ("WV_phase_3", wv.material_densities[2]),
                      ("WV_tau", wv.tau),
                      ("WV_void_fraction", wv.void_fraction),
                      ("WV_permeability", wv.permeability),
                      ("WV_tortuosity", wv.tortuosity),
                      ("WV_density", wv.density),
                      ]

        # depending on the version, paraview may complain without this
        viz_fields.append(("x", nodes[0]))

        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
            step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)

    def my_write_restart(step, t, state):
        cv = state.cv
        dv = state.dv
        wv = state.wv
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "material_densities": wv.material_densities,
                "tseed": dv.temperature,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "nparts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_pre_step(step, t, dt, state):

        cv, material_densities, tseed = state
        fluid_state = compiled_make_state(cv, tseed, material_densities)
        dv = fluid_state.dv

        try:

            if logmgr:
                logmgr.tick_before()

            do_garbage = check_step(step=step, interval=ngarbage)
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_garbage and (gc_timer is not None):
                with gc_timer:
                    gc.collect()

            if do_health:
                if check_naninf_local(dcoll, "vol", dv.temperature):
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=fluid_state)

            if do_viz:
                my_write_viz(step=step, t=t, state=fluid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state)
            my_write_restart(step=step, t=t, state=fluid_state)
            raise

        return state, dt

    def my_rhs(t, state):

        cv, material_densities, tseed = state

        fluid_state = make_state(cv, temperature_seed, material_densities)

        boundaries = make_obj_array([pressure_boundaries, velocity_boundaries])

        cv_rhs, wall_rhs = phenolics_operator(
            dcoll=dcoll, fluid_state=fluid_state, boundaries=boundaries,
            gas_model=gas_model, pyrolysis=pyrolysis, quadrature_tag=quadrature_tag,
            time=t, bprime_class=bprime_class,
            pressure_scaling_factor=pressure_scaling_factor, penalty_amount=1.0)

        return make_obj_array([cv_rhs, wall_rhs, tseed*0.0])

    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer:
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections")
                gc.freeze()

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    from mirgecom.steppers import advance_state
    current_step, current_t, advanced_state = \
        advance_state(rhs=my_rhs, timestepper=ssprk43_step,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=make_obj_array([fluid_state.cv, material_densities,
                                            fluid_state.temperature]),
                      t=current_t, t_final=t_final, istep=istep,
                      force_eval=True)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_cv, current_material_densities, tseed = advanced_state
    current_state = make_state(current_cv, tseed, current_material_densities)
    my_write_viz(step=current_step, t=current_t, state=current_state)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":

    import argparse
    casename = "ablation"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    from warnings import warn
    warn("Automatically turning off DV logging. MIRGE-Com Issue(578)")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    main(actx_class=actx_class, use_logmgr=args.log, casename=casename,
         restart_file=restart_file)

# vim: foldmethod=marker
