================================
Mathematical Model of Fluid Flow
================================

.. important::

   |mirgecom| main repo branch currently provides fluid simulation operators and
   utilities for inviscid flow (i.e. the :ref:`Euler equations<euler-eqns>`).
   Viscous flow simulation capabilities are forthcoming.

.. raw:: latex

    \let\b=\mathbf

.. raw:: html

    \(
    \let\b=\mathbf
    \)

.. _ns-eqns:

|mirgecom| provides capabilities for solving the compressible Navier-Stokes equations for
viscous flows and the :ref:`Euler equations<euler-eqns>` equations for inviscid flows of
reactive fluid mixtures. |mirgecom| supports reactive fluid mixtures with a number of
mixture species $N_s$ on unstructured meshes of 1, 2, and 3-dimensional domains $\Omega$,
and :ref:`discretizes the equations in a Discontinuous Galerkin setting<disc-strat>`.

The formulation presented here follows [Ihme_2014]_ and [Cook_2009]_. Unless otherwise
noted, Einstein summation convention is used throughout the following sections.  The
governing equations, written in conservative form, are summarized as follows:

$$
\partial_{t}{\rho} + \partial_{j}{(\rho v)_j} &= {S}_\rho \\
\partial_{t}(\rho{E}) + \partial_j\left(\left\{\rho E + p\right\}v_j + q_j -
\tau_{jk}v_k\right) &= {S}_{\rho E} \\
\partial_{t}({\rho}{v_i}) + \partial_j\left((\rho v)_i v_j + p\delta_{ij} -
\tau_{ij}\right) &= {S}_{\rho v_i} \\
\partial_{t}(\rho{Y})_{\alpha} + \partial_j\left((\rho{Y})_{\alpha}v_j +
(\b{J}_{\alpha})_j\right) &= {S}_{\alpha},
$$

with fluid density $\rho$, velocity components $v_i$, momentum density components
$((\rho v)_i$), total energy $(\rho E)$, and vector of species mass fractions
${Y}_{\alpha}$. The :ref:`thermodynamic pressure<eos-and-matprop>` of the fluid is $p$.
${\tau_{ij}}$ are the components of the
:ref:`viscous stress tensor<viscous-stress-tensor>`, $q_i$ are the components of the total
:ref:`heat flux<heat-flux>` vector, and the components of the species
:ref:`diffusive flux<diffusive-flux>` vector are $(\b{J}_{\alpha})_i$. Mixtures have
$N_s$ components with $1 \le \alpha \le N_s$.

The equations can be recast in a more compact form:

$$
\partial_t{\b{Q}} + \partial_j{\b{F}^{I}(\b{Q})_j} =
\partial_j{\b{F}^{V}(\b{Q}, \nabla\b{Q})_j} + \b{S},
$$

where $\b{Q}$ is the vector of conserved variables, $\b{F}^I$ is the vector of
inviscid fluxes, $\b{F}^V$ is the vector of viscous fluxes, and the vector of sources
for each scalar equation  is $\b{S}$. The components of each vector follow directly from
above:

$$
\b{Q} = \begin{bmatrix}\rho\\\rho{E}\\(\rho{v})_{i}\\(\rho{Y})_{\alpha}\end{bmatrix},
~\b{F}^{I}_{j} = \begin{bmatrix}(\rho{v})_{j}\\\left(\rho{E}+p\right){v}_{j}\\
\left((\rho{v})_{j}{v}_{i}+p\delta_{ij}\right)\\(\rho{Y})_{\alpha}{v}_{j}\end{bmatrix},
~\b{F}^V_{j} = \begin{bmatrix}0\\\left(\tau_{jk}{v}_{k}-{q}_{j}\right)\\
{\tau}_{ij}\\-(\b{J}_{\alpha})_{j}\end{bmatrix},
~\b{S} = \begin{bmatrix}0\\E^{\mathtt{chem}}\\0\\W^{\mathtt{chem}}_{\alpha}
\end{bmatrix}
$$

where ${E}^{\text{chem}}$, and $W^{\text{chem}}_{\alpha}$, are the chemical reaction
source terms in the energy and species conservation equations, respectively.  See
:ref:`Chemistry` for more details on chemical reaction source terms, and
:ref:`here<disc-strat>` for details on the discretization strategy for this system of
conservation equations.

.. _euler-eqns:

The Euler equations for inviscid flows are recovered from the Navier-Stokes system
above when the viscous fluxes vanish. That is, when $\b{F}^V=0$, we are left with a
system of nonlinear equations for a completely inviscid fluid. |mirgecom| provides an
Euler operator, with associated utilities functions, for solving flows of this type.
Inviscid fluxes and utilities are found in :mod:`mirgecom.inviscid`, and the Euler
operator for the RHS in :mod:`mirgecom.euler`.  Viscous fluxes and utilities for
calculating the components of the viscous fluxes are found in :mod:`mirgecom.viscous`.

.. _viscous-stress-tensor:

Viscous stress tensor
---------------------
The viscous stress tensor has components:

$$
\tau_{ij} = \mu \left(\partial_j{v_i} + \partial_i{v_j}\right) +(\mu_B - \frac{2}{3}\mu)
\partial_k{v_k}\delta_{ij}
$$

with fluid velocity components ${v}_{i}$, the first coefficient of fluid viscosity $\mu$,
and bulk viscosity $\mu_B$.  The viscous stress tensor is computed by |mirgecom| in the
:mod:`~mirgecom.viscous` module routine :func:`~mirgecom.viscous.viscous_stress_tensor`.

.. _diffusive-flux:

Diffusive flux
--------------
The species diffusive fluxes are given by:

$$
\b{J}_{\alpha} = -\rho{d}_{(\alpha)}\nabla{Y}_{\alpha},
$$

with gas density $\rho$, species diffusivities ${d}_{\alpha}$, and
species mass fractions ${Y}_{\alpha}$.  The parens $(\alpha)$ indicate no sum
over repeated indices is to be performed.


.. _heat-flux:

Heat flux
---------

The total heat flux $\b{q}$ is calculated as the sum of the
conductive and diffusive components, $\b{q}_{c}$ and $\b{q}_{d}$,
respectively:

$$
\b{q} = \b{q}_c + \b{q}_d
$$

Conductive heat flux
^^^^^^^^^^^^^^^^^^^^
The conductive heat flux vector is defined directly from Fourier's law of thermal
conduction:

$$
\b{q}_c = -\kappa\nabla{T},
$$

where $\kappa$ is the thermal conductivity, and ${T}$ is the gas
temperature.

Diffusive heat flux
^^^^^^^^^^^^^^^^^^^
The diffusive heat flux vector is defined as

$$
\b{q}_d = {h}_{\alpha}\b{J}_{\alpha},
$$

with the species specific enthalpy ${h}_{\alpha}$, and the species
diffusive flux vector $\b{J}_{\alpha}$.

.. _chemistry:

Chemistry
---------

Chemical reactions introduce source terms in the energy and species conservation equations.
The species source term is the amount of mass produced for each species:

$$
W^{\mathtt{chem}}_{\alpha} = w_{(\alpha)}\partial_t{\omega}_{\alpha},
$$

where ${w}_{\alpha}$ is the molecular weight of each species, and
$\partial_t{\omega}_{\alpha}$ is the net chemical production rate for each species. Here,
the parens $(\alpha)$ indicates no sum is to be performed over repeated indices.

The energy source term is the amount of thermal energy used to create each species:

$$
E^{\mathtt{chem}} = -h^f_{\alpha}W^{\mathtt{chem}}_{\alpha},
$$

where $h^f_{\alpha}$ is the enthalpy of formation for each species.

.. _eos-and-matprop:

Equations of State and Material properties
------------------------------------------

Equations of state (EOS) provide functions that relate the fluid state $\b{Q}$,
and the thermodynamic properties such as pressure $p$, temperature $T$, specific
enthalpies $h_{\alpha}$, and total energy $E$.  The EOS provided by |mirgecom| are
documented in :mod:`mirgecom.eos`.

Material properties including the first coefficient of viscosity, $\mu$, bulk viscosity
$\mu_B$, thermal conductivity $\kappa$, and species diffusivities ${d}_{\alpha}$ depend on
the state of the fluid $\b{Q}$, in general, and are provided by transport models.
Transport models provided by |mirgecom| are documented :mod:`mirgecom.transport`.

.. note::
   
    The EOS and transport models provide closure for the fluid model in that the fluid
    thermal state variables such as pressure $p$, temperature $T$, and material
    properties such as viscosity $\mu$, and thermal conductivity $\kappa$ are functions of
    the current fluid state $\b{Q}$. The EOS and transport models provide constructs
    that manage the relationships between these quantities, and provide methods for
    calculating them from minimal working sets of input data.

Code correspondence
-------------------

The following summarizes the code components and constructs that implement the various
pieces of the conservation systems presented above.

- Inviscid flow (:mod:`mirgecom.inviscid`)

  - inviscid flux :func:`~mirgecom.inviscid.inviscid_flux`

  - Euler RHS: :func:`~mirgecom.euler.euler_operator`

- Viscous flow (soon)

- Equations of State (:mod:`mirgecom.eos`)

  - single ideal gas: :class:`~mirgecom.eos.IdealSingleGas`

  - gas mixture: :class:`~mirgecom.eos.PyrometheusMixture`

- Transport models (soon)

- Chemistry (:mod:`pyrometheus`)
