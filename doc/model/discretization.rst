=====================
Navier-Stokes with DG
=====================

.. _disc-strat:

Discretization Strategy
-----------------------

How to discretize the conservation equations with DG, including how to handle the required fluxes,
particularly in the viscous setting, is a current topic of research and internal discussion.  The
following references are useful:

* "The DG Book:" Nodal Discontinuous Galerkin Methods, [Hesthaven_2008]_
* The BR1 algorithm for discretization of Navier-Stokes, [Bassi_1997]_
* NS with reactions, [Ihme_2014]_, and [Cook_2009]_
* The BR2 algorithm, [Bassi_2000]_
* [Ayuso_2009]_

*MIRGE-Com* currently employs a strategy akin to the BR1 algorithm outlined in [Bassi_1997]_, but
with thermal terms and chemical reaction sources as outlined in [Ihme_2014]_ and [Cook_2009]_.

2nd order terms on the viscous RHS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The viscous fluxes $\mathbf{F}^{V}$ are proportional to gradients of the fluid state variables,
introducing 2nd order terms on the RHS of the conservation equations. These 2nd order terms with their
relevant rhs component are summarized below.

Momentum equation
"""""""""""""""""
The 2nd order terms in the viscous RHS for the moementum equation are:

.. math::
   \partial_j \tau_{ij} = \left[\partial_j\left(\mu\partial_j{v}_i\right) +
   \partial_j\left(\mu\partial_i{v}_j\right) + \partial_j\left(\mu_{B} -
   \frac{2}{3}\mu\right)\partial_k{v}_k\delta_{ij}\right]


Energy equation
"""""""""""""""
The 2nd order terms in the energy equation RHS have convective, conductive, and
diffusive terms as follows:

- Convective part

.. math::
   \partial_j \tau_{jk} {v}_k = \left[\partial_j\left(\mu\partial_k{v}_j{v}_k\right) +
   \partial_j\left(\mu\partial_j{v}^2_k\right) + \partial_j\left(\mu_{B} -
   \frac{2}{3}\mu\right)\partial_m{v}_m\delta_{jk}{v}_k\right]
   

- Conductive part

The conductive heat part of the RHS is:

.. math::
   \partial_j{(q_{c})_j} = \partial_j\kappa\partial_j{T},

where $T$ is the fluid temperature.

- Diffusive part

The diffusive heat part of the RHS is:

.. math::
   \partial_j{(q_{d})_j} = \partial_j\left(\rho{h}_{\alpha}{d}_{(\alpha)}\partial_j{Y}_{\alpha}\right)
   
with fluid density $\rho$, species diffusivity ${d}_{(\alpha)}$, and species mass fractions
${Y}_{\alpha}$. 

Species equation
""""""""""""""""
The species diffusive transport RHS is:

.. math::
   \partial_j{(J_{\alpha})_j} = \partial_j\left(\rho{d}_{(\alpha)}\partial_j{Y}_{\alpha}\right),

with fluid density $\rho$, species diffusivity ${d}_{(\alpha)}$, and species mass fractions
${Y}_{\alpha}$. 
