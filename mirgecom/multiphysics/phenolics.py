""":mod:`mirgecom.multiphysics.phenolics` handles phenolics modeling."""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
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

from meshmode.dof_array import DOFArray
from pytools.obj_array import make_obj_array
from mirgecom.fluid import ConservedVars


def initializer(dcoll, gas_model, material_densities, temperature,
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
        raise ValueError("Temperature does not have the proper shape")

    tau = gas_model.wall.decomposition_progress(material_densities)

    # gas constant
    gas_const = 8314.46261815324/gas_model.eos.molar_mass(temperature)

    if gas_density is None:
        eps_gas = gas_model.wall.void_fraction(tau)
        eps_rho_gas = eps_gas*pressure/(gas_const*temperature)

    # internal energy (kinetic energy is neglected)
    eps_rho_solid = sum(material_densities)
    bulk_energy = (
        eps_rho_gas*(gas_model.eos.enthalpy(temperature) - gas_const*temperature)
        + eps_rho_solid*gas_model.wall.enthalpy(temperature, tau)
    )

    momentum = make_obj_array([tau*0.0 for i in range(dcoll.dim)])

    return ConservedVars(mass=eps_rho_gas, energy=bulk_energy, momentum=momentum)
