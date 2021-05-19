=======================
 Navier-Stokes with DG
=======================

.. _disc-strat:

Discretization Strategy
=======================

How to discretize the conservation equations with DG, including how to handle the required
fluxes, particularly in the viscous setting, is a current topic of research and internal
discussion.  The following references are useful:

* "The DG Book:" Nodal Discontinuous Galerkin Methods, [Hesthaven_2008]_
* The BR1 method for discretization of Navier-Stokes, [Bassi_1997]_
* NS with reactions, [Ihme_2014]_, and [Cook_2009]_
* The BR2 method, [Bassi_2000]_
* [Ayuso_2009]_

*MIRGE-Com* currently employs a strategy akin to the BR1 algorithm outlined in
[Bassi_1997]_, but with thermal terms and chemical reaction sources as outlined in
[Ihme_2014]_ and [Cook_2009]_.

Navier-Stokes BR1
=================

The main system of equations we solve in the BR1 approach is summarized as follows:

The compressible NS equations are rewritten as the following coupled system for two unknowns,
$\mathbf{Q}$ and $\mathbf{\Sigma}$:

$$
\mathbf{\Sigma} - \nabla{\mathbf{Q}} &= \mathbf{0}\quad \text{aux eqn}\\
\frac{\partial \mathbf{Q}}{\partial t} + \underbrace{\nabla\cdot\mathbf{F}^I(\mathbf{Q}) -
\nabla\cdot\mathbf{F}^V(\mathbf{Q},\mathbf{\Sigma})}_{= \nabla\cdot\mathbf{F}(\mathbf{Q},
\mathbf{\Sigma})} &= \mathbf{S} \quad \text{primary eqn}
$$

Let $\Omega_h$ denote a collection of disjoint elements $E$. Then the DG formulation is
obtained by multiplying by ''test functions'' $\mathbf{v}_h$, $\mathbf{w}_h$ (one for each
equation repsectively) and integrating over each element:

$$
\sum_{E\in\Omega_h} \left\lbrack \int_E \mathbf{v}_h\cdot\frac{\partial \mathbf{Q}_h}
{\partial t} d\Omega + \oint_{\partial E} \mathbf{v}_h\mathbf{h} d\sigma - \int_E \nabla
\mathbf{v}_h\cdot\mathbf{F}(\mathbf{Q}_h, \mathbf{\Sigma}_h)d\Omega\right\rbrack &=
\sum_{E\in\Omega_h} \int_E \mathbf{v}_h\cdot\mathbf{S}_h d\Omega, &\quad \forall
\mathbf{v}_h \\
\sum_{E\in\Omega_h}\left\lbrack \int_E \mathbf{w}_h\cdot \mathbf{\Sigma}_h d\Omega -
\oint_{\partial E} \mathbf{w}_h\cdot\mathbf{H}_s d\sigma + \int_E \nabla\mathbf{w}_h\cdot
\mathbf{Q}_h d\Omega\right\rbrack &= 0, &\quad \forall \mathbf{w}_h,
$$

where $\mathbf{h} = \mathbf{h}_e - \mathbf{h}_v$ is a *numerical flux* consisting of a
*invicid* and *viscous* component respectively, and $\mathbf{H}_s$ is the
*gradient numerical flux* for the auxiliary equation. Here, we use the subscript "h" to
denote discretized quantities.

Since $\mathbf{F}^I(\mathbf{Q}_h)\cdot\mathbf{n}$ is discontinuous, the quantities are
allowed to vary on either side of a shared element boundary. That is, $\mathbf{F}^I
(\mathbf{Q}^+_h)\cdot\mathbf{n}^+ \neq \mathbf{F}^I(\mathbf{Q}^-_h)\cdot\mathbf{n}^-$,
where $\mathbf{n}^\pm$ denotes the ''cell-local'' (or interior) normal with respect to the
element $E^\pm$ which share a face: $\partial E^+ \cap \partial E^- \neq \emptyset$.
Similarly for $\mathbf{F}^V(\mathbf{Q}_h, \mathbf{\Sigma}_h)\cdot\mathbf{n}$ and
$\mathbf{Q}_h\mathbf{n}$.

Expanding out the trial and test functions in terms of the local element basis in each element:
$$
\mathbf{Q}(\mathbf{x}, t)_h|_E = \sum_{i=1}^n \mathbf{Q}_i(t)\phi_i^k(\mathbf{x}), \quad
\mathbf{\Sigma}(\mathbf{x})_h|_E = \sum_{i=1}^n \mathbf{\Sigma}_i\phi_i^k(\mathbf{x}), \\
\mathbf{v}(\mathbf{x})_h|_E = \sum_{i=1}^n \mathbf{v}_i\phi_i^k(\mathbf{x}), \quad
\mathbf{w}(\mathbf{x})_h|_E = \sum_{i=1}^n \mathbf{w}_i\phi_i^k(\mathbf{x}),
$$
allows us to obtain a set of algebraic equations for the prognostic state $\mathbf{Q}_h$ and
the auxiliary gradient variable $\mathbf{\Sigma}_h$. That is, for each
$j = 1, \cdots, \dim P^k$, we have:
$$
\begin{align}
\frac{d}{dt} \sum_{E\in\Omega_h}\int_E \phi_j^k\mathbf{Q}_h d\Omega &= \sum_{E\in\Omega_h}
\left\lbrack\int_E \nabla\phi_j^k\cdot\left(\mathbf{F}^V(\mathbf{Q}_h, \mathbf{\Sigma}_h) -
\mathbf{F}^I(\mathbf{Q}_h)\right)d\Omega\right\rbrack \\
&- \sum_{E\in\Omega_h}\left\lbrack\oint_{\partial E}\phi_j^k \mathbf{h}_e(\mathbf{Q}_h^+,
\mathbf{Q}^-_h; \mathbf{n}) d\sigma + \oint_{\partial E} \phi_j^k \mathbf{h}_v(\mathbf{Q}_h^+,
\mathbf{\Sigma}_h^+, \mathbf{Q}_h^-, \mathbf{\Sigma}_h^-; \mathbf{n}) d\sigma\right\rbrack \\
&+ \sum_{E\in\Omega_h} \int_E \phi_j^k\mathbf{S}_h d\Omega, \\
\sum_{E\in\Omega_h}\int_E\phi_j^k \mathbf{\Sigma}_h d\Omega &= \sum_{E\in\Omega_h}\left\lbrack
\oint_{\partial{E}}\phi_j^k \mathbf{H}_s(\mathbf{Q}^+_h, \mathbf{Q}_h^-; \mathbf{n}) d\sigma -
\int_E\nabla\phi^k_j\cdot\mathbf{Q}_h d\Omega\right\rbrack.
\end{align}
$$

Numerical fluxes
----------------

To account for the discontinuities at element faces, DG employs numerical fluxes, which are
enforced to be singled-valued, but are functions of both $\pm$ states:

$$
\mathbf{h}_e(\mathbf{Q}_h^+, \mathbf{Q}^-_h; \mathbf{n}) &\approx \mathbf{F}^I(\mathbf{Q}_h)
\cdot\mathbf{n}, \\
\mathbf{h}_v(\mathbf{Q}_h^+, \mathbf{\Sigma}_h^+, \mathbf{Q}_h^-, \mathbf{\Sigma}_h^-;
\mathbf{n}) &\approx \mathbf{F}^V(\mathbf{Q}_h, \mathbf{\Sigma}_h)\cdot\mathbf{n}\\
\mathbf{H}_s(\mathbf{Q}^+_h, \mathbf{Q}_h^-; \mathbf{n}) &\approx \mathbf{Q}_h\mathbf{n}.
$$

Choices of numerical fluxes corresponding to BR1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Take $\mathbf{h}_e(\mathbf{Q}_h^+, \mathbf{Q}^-_h; \mathbf{n})$ to be one of:
  
  * Local Lax-Friedrichs (LLF), Roe, Engquist-Osher, HLLC (there are many more!). If you're
    feeling especially brave, you can even use the average of the inviscid flux (centered
    flux).

  * Mirgecom uses LLF at the moment:
    $$
    \mathbf{h}_{e}(\mathbf{Q}_h^+, \mathbf{Q}^-_h; \mathbf{n}) = \frac{1}{2}\left(
    \mathbf{F}^{I}(\mathbf{Q}_h^+)+\mathbf{F}^{I}(\mathbf{Q}_h^-)\right) - \frac{\lambda}
    {2}\left(\mathbf{Q}_h^+ - \mathbf{Q}_h^-\right)\mathbf{n},
    $$
    where $\lambda$ is the characteristic max wave-speed of the fluid. Numerical fluxes
    which penalize the ''jump'' of the state $\left(\mathbf{Q}_h^+ - \mathbf{Q}_h^-\right)
    \mathbf{n}$ act as an additional source of dissipation, which has a stabilizing effect
    on the numerics.

* Take $\mathbf{h}_v(\mathbf{Q}_h^+, \mathbf{\Sigma}_h^+, \mathbf{Q}_h^-,
  \mathbf{\Sigma}_h^-; \mathbf{n})$ to be defined in a ''centered'' (average) way:
  $$
  \mathbf{h}_v(\mathbf{Q}_h^+, \mathbf{\Sigma}_h^+, \mathbf{Q}_h^-, \mathbf{\Sigma}_h^-;
  \mathbf{n}) = \frac{1}{2}\left(\mathbf{F}^V(\mathbf{Q}_h^+, \mathbf{\Sigma}_h^+) +
  \mathbf{F}^V(\mathbf{Q}_h^-, \mathbf{\Sigma}_h^-)\right)\cdot\mathbf{n}
  $$

* And similarly for the gradient flux:
  $$
  \mathbf{H}_s(\mathbf{Q}_h^+, \mathbf{Q}_h^- ; \mathbf{n}) = \frac{1}{2}\left(
  \mathbf{Q}_h^+ + \mathbf{Q}_h^-\right)\mathbf{n}
  $$


Domain boundary considerations
------------------------------

What happens when $\partial E \cap \partial\Omega \neq \emptyset$?

In DG, numerical fluxes are not only responsible for handling the flow of information
between adjacent cells, but they also enforce information at the boundaries.

We denote the **boundary fluxes** as $\mathbf{h}^*_e(\mathbf{Q}_{bc})$,
$\mathbf{h}^*_v(\mathbf{Q}_{bc}$, $\mathbf{\Sigma}_{bc})$, and
$\mathbf{H}^*_s(\mathbf{Q}_{bc})$, where $\mathbf{Q}_{bc}$, $\mathbf{\Sigma}_{bc}$ denote
boundary conditions imposed on the state, and the gradient of the state respectively.

For all $\partial E \cap \partial\Omega$ there is no $+$ side to consider; just the
interior state ($-$ side) and the prescribed boundary conditions $\mathbf{Q}_{bc},
\mathbf{\Sigma}_{bc}$.

* At solid walls:

  $\mathbf{h}^*_e$ is equal to the (interior; - side) pressure contribution of
  $\mathbf{F}^I(\mathbf{Q}_{bc})\cdot\mathbf{n}$ (since $\mathbf{V}\cdot\mathbf{n} = 0$).
    
  * The viscous boundary flux is computed as:
    $$
    \mathbf{h}^*_v(\mathbf{Q}_{bc}, \mathbf{\Sigma}_{bc}) = \mathbf{F}_V(\mathbf{Q}_{bc},
    \mathbf{\Sigma}_{bc})\cdot\mathbf{n},
    $$
    where $\mathbf{Q}_{bc}$ are the same values used to prescribe $\mathbf{h}^*_e$.

  * If there are no conditions of $\nabla\mathbf{Q}\cdot\mathbf{n}$, then
    $$
    \mathbf{\Sigma}_{bc} = \mathbf{\Sigma}_h^-.
    $$

  Otherwise, $\mathbf{\Sigma}_{bc}$ will need to be modified accordingly.

* At inflow/outflow boundaries:

  $$
  \mathbf{h}^*_e(\mathbf{Q}_{bc}) = \mathbf{h}_e(\mathbf{Q}_{bc}, \mathbf{Q}^-_{h};
  \mathbf{n}).
  $$

  * $\mathbf{Q}_{bc}$ is also used to define the gradient boundary flux:
    $$
    \mathbf{H}^*_s(\mathbf{Q}_{bc}) = \mathbf{Q}_{bc}\mathbf{n}.
    $$

  * The viscous boundary flux is evaluated as:
    $$
    \mathbf{h}^*_v = \mathbf{h}_v(\mathbf{Q}_{bc}, \mathbf{\Sigma}_h^-, \mathbf{Q}_h^-,
    \mathbf{\Sigma}_h^-; \mathbf{n}),
    $$
    where $\mathbf{Q}_{bc}$ are the same values used for $\mathbf{h}^*_e$.



2nd order terms on the viscous RHS
----------------------------------

This section breaks out explicit component versions of the 2nd order terms on the RHS to
help guide a discussion about alternate approaches to discretization in which a generic
diffusion operator could potentially be responsible for integrating some of these terms.

The viscous fluxes $\mathbf{F}^{V}$ are proportional to gradients of the fluid state
variables, introducing 2nd order terms on the RHS of the conservation equations. These 2nd
order terms with their relevant RHS component are summarized below.

Momentum equation
^^^^^^^^^^^^^^^^^

The 2nd order terms in the viscous RHS for the moementum equation are:

$$
\partial_j \tau_{ij} = \left[\partial_j\left(\mu\partial_j{v}_i\right) + \partial_j
\left(\mu\partial_i{v}_j\right) + \partial_j\left(\mu_{B} - \frac{2}{3}\mu\right)
\partial_k{v}_k\delta_{ij}\right]
$$

Energy equation
^^^^^^^^^^^^^^^

The 2nd order terms in the energy equation RHS have convective, conductive, and
diffusive terms as follows:

- Convective part

$$
\partial_j \tau_{jk} {v}_k = \left[\partial_j\left(\mu\partial_k{v}_j{v}_k\right) +
\partial_j\left(\mu\partial_j{v}^2_k\right) + \partial_j\left(\mu_{B} - \frac{2}{3}\mu
\right)\partial_m{v}_m\delta_{jk}{v}_k\right]
$$   

- Conductive part

The conductive heat part of the RHS is:

$$
\partial_j{(q_{c})_j} = \partial_j\kappa\partial_j{T},
$$

where $T$ is the fluid temperature.

- Diffusive part

The diffusive heat part of the RHS is:

$$
\partial_j{(q_{d})_j} = \partial_j\left(\rho{h}_{\alpha}{d}_{(\alpha)}\partial_j
{Y}_{\alpha}\right)
$$

with fluid density $\rho$, species diffusivity ${d}_{(\alpha)}$, and species mass
fractions ${Y}_{\alpha}$. 

Species equation
^^^^^^^^^^^^^^^^

The species diffusive transport RHS is:

$$
\partial_j{(J_{\alpha})_j} = \partial_j\left(\rho{d}_{(\alpha)}\partial_j{Y}_{\alpha}
\right),
$$

with fluid density $\rho$, species diffusivity ${d}_{(\alpha)}$, and species mass
fractions ${Y}_{\alpha}$. 
