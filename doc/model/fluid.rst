=====
Fluid
=====

.. note::

   This model document has been updated to be *MIRGE-Com* specific, and to include multi-component mixtures
   with chemical reactions.

.. important::

   *MIRGE-Com* main repo branch currently provides fluid simulation operators and utilities for inviscid
   flow (i.e. the :ref:`Euler equations<Euler-eqns>`).  Viscous flow simulation capabilities are forthcoming.

.. raw:: latex

    \def\RE{\operatorname{RE}}
    \def\PR{\operatorname{PR}}

.. raw:: html

    \(
    \def\RE{\operatorname{RE}}
    \def\PR{\operatorname{PR}}
    \)

.. _NS-eqns:

*MIRGE-Com* provides capabilities for solving the compressible Navier-Stokes equations for viscous flows and
the :ref:`Euler equations<Euler-eqns>` equations for inviscid flows of reactive fluid mixtures. *MIRGE-Com*
supports reactive fluid mixtures with a number of mixture species = $N_s$ on unstructured meshes and
:ref:`discretizes the equations in a Discontinuous Galerkin setting<disc-strat>`.

The formulation presented here follows from [Ihme_2014]_ and [Cook_2009]_. The governing equations, written in conservative form, are summarized as follows:

.. math::
    \partial_{t}{\rho} + \partial_{j}{\rho v_j} &= S_\rho \\
    \partial_{t}(\rho{E}) + \partial_j\left(\left\{\rho E + p\right\}v_j + q_j - \tau_{jk}v_k\right) &= S_{\rho E} \\
    \partial_{t}({\rho}{v_i}) + \partial_j\left(\rho v_i v_j + p\delta_{ij} - \tau_{ij}\right) &= S_{\rho v_i} \\
    \partial_{t}(\rho{Y}_{\alpha}) + \partial_j\left(\rho{Y}_{\alpha}v_j + (\mathbf{J}_{\alpha})_j\right) &= S_{\alpha},

with fluid density $\rho$, velocity components $v_i$, momentum density components $\rho v_i$, total energy $\rho E$,
and vector of species mass fractions ${Y}_{\alpha}$. The :ref:`thermodynamic pressure<eos-and-matprop>` of the fluid
is $p$.  ${\tau_{ij}}$ are the components of the :ref:`viscous stress tensor<viscous-stress-tensor>`, $q_i$ are the
components of the total :ref:`heat flux<heat-flux>` vector, and the components of the
species :ref:`diffusive flux<diffusive-flux>` vector are $(\mathbf{J}_{\alpha})_i$. Mixtures have $N_s$ components
with $1 \le \alpha \le N_s$. Unless otherwise noted, repeated indices imply summation.

The equations can be recast in a more compact form:

.. math::

    \partial_t{\mathbf{Q}} + \partial_j{\mathbf{F}^{I}_j} = \partial_j{\mathbf{F}^{V}_j} + \mathbf{S},

where $\mathbf{Q}$ is the vector of conserved variables, $\mathbf{F}^I$ is the vector of inviscid fluxes,
$\mathbf{F}^V$ is the vector of viscous fluxes, and the vector of sources for each scalar equation  is $\mathbf{S}$.
The components of each vector follow directly from above:

.. math::

   \mathbf{Q} = \begin{bmatrix}\rho\\\rho{E}\\\rho{v}_{i}\\\rho{Y}_{\alpha}\end{bmatrix},
   ~\mathbf{F}^{I}_{j} = \begin{bmatrix}\rho{v}_{j}\\\left(\rho{E}+p\right){v}_{j}\\
   \left(\rho{v}_{j}{v}_{i}+p\delta_{ij}\right)\\\rho{Y}_{\alpha}{v}_{j}\end{bmatrix},
   ~\mathbf{F}^V_{j} = \begin{bmatrix}0\\\left(\tau_{jk}{v}_{k}-{q}_{j}\right)\\{\tau}_{ij}\\
   -(\mathbf{J}_{\alpha})_{j}\end{bmatrix},
   ~\mathbf{S} = \begin{bmatrix}0\\E^{\mathtt{chem}}\\0\\W^{\mathtt{chem}}_{\alpha}\end{bmatrix}

where ${E}^{\mathtt{chem}}$, and $W^{\mathtt{chem}}_{\alpha}$, are the chemical reaction source terms
in the energy and species conservation equations, respectively.  See :ref:`Chemistry` for more details
on chemical reaction source terms, and :ref:`here<disc-strat>` for details on the discretization
strategy for this system of conservation equations.

.. _Euler-eqns:

The Euler equations for inviscid flows are recovered from the Navier-Stokes system above when the
viscous fluxes vanish. That is, when $\mathbf{F}^V=0$, we are left with a system of nonlinear
equations for a completely inviscid fluid. *MIRGE-Com* provides an Euler operator, with associated
utilities functions, for solving flows of this type.  Inviscid fluxes and utilities are found in
:mod:`mirgecom.inviscid`, and the Euler operator for the RHS in :mod:`mirgecom.euler`.

.. _viscous-stress-tensor:

Viscous stress tensor
---------------------
The viscous stress tensor has components:

.. math::
    \tau_{ij} = \mu \left(\partial_j{v_i} + \partial_i{v_j}\right)
    +(\mu_B - \frac{2}{3}\mu)\partial_k{v_k}\delta_{ij}

with fluid velocity components ${v}_{i}$, the first coefficient of fluid
viscosity $\mu$, and bulk viscosity $\mu_B$.


.. _diffusive-flux:

Diffusive flux
--------------
The species diffusive fluxes are given by:

.. math::
   \mathbf{J}_{\alpha} = -\rho{d}_{(\alpha)}\nabla{Y}_{\alpha},

with gas density $\rho$, species diffusivities ${d}_{\alpha}$, and
species mass fractions ${Y}_{\alpha}$.  The parens $(\alpha)$ indicate no sum
over repeated indices is to be performed.


.. _heat-flux:

Heat flux
---------

The total heat flux $\mathbf{q}$ is calculated as the sum of the
conductive and diffusive components, $\mathbf{q}_{c}$ and $\mathbf{q}_{d}$,
respectively:

.. math::
   \mathbf{q} = \mathbf{q}_c + \mathbf{q}_d


Conductive heat flux
^^^^^^^^^^^^^^^^^^^^
The conductive heat flux vector is defined directly from Fourier's law of thermal conduction:

.. math::
    \mathbf{q}_c = -\kappa\nabla{T},

where $\kappa$ is the thermal conductivity, and ${T}$ is the gas
temperature.

Diffusive heat flux
^^^^^^^^^^^^^^^^^^^
The diffusive heat flux vector is defined as

.. math::
   \mathbf{q}_d = {h}_{\alpha}\mathbf{J}_{\alpha},

with the species specific enthalpy ${h}_{\alpha}$, and the species
diffusive flux vector $\mathbf{J}_{\alpha}$.

.. _Chemistry:

Chemistry
---------

Chemical reactions introduce source terms in the energy and species conservation equations.
The species source term is the amount of mass produced for each species:

.. math::
   W^{\mathtt{chem}}_{\alpha} = w_{(\alpha)}\dot{\omega}_{\alpha},

where ${w}_{\alpha}$ is the molecular weight of each species, and $\dot{\omega}_{\alpha}$ is the net
chemical production rate for each species. Here, the parens $(\alpha)$ indicates no sum is to be performed
over repeated indices. 

The energy source term is the amount of thermal energy used to create each species:

.. math::
   E^{\mathtt{chem}} = -h^f_{\alpha}W^{\mathtt{chem}}_{\alpha},

where $h^f_{\alpha}$ is the enthalpy of formation for each species.

.. _eos-and-matprop:

Equations of State and Material properties
------------------------------------------

Equations of state (EOS) provide functions that relate the fluid state $Q$, and the
thermodynamic properties such as pressure $p$, temperature $T$, specific enthalpies $h_{\alpha}$,
and total energy $E$.  The EOS provided *MIRGE-Com* are documented in :mod:`mirgecom.eos`.

Material properties including the first coefficient of viscosity, $\mu$, bulk viscosity $\mu_B$,
thermal conductivity $\kappa$, and species diffusivities ${d}_{\alpha}$ depend on the state of
the fluid $\mathbf{Q}$, in general, and are provided by transport models.  Transport models provided
by *MIRGE-Com* ~~are~~ (will be) documented in forthcoming the transport module.
