=======================
Discretization Strategy
=======================

.. _disc-strat:

Literature on DG for Fluid Flow
===============================

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

Nodal DG for Navier-Stokes
==========================

The main system of equations we solve in the BR1 approach is summarized as follows:

The compressible NS equations are rewritten as the following coupled system for two
unknowns, $\mathbf{Q}$ and $\mathbf{\Sigma}$:

$$
\let\b=\mathbf
\b{\Sigma} - \nabla{\b{Q}} &= \b{0}\quad{\text{auxiliary eqn}}\\
\frac{\partial \b{Q}}{\partial t} + \underbrace{\nabla\cdot\b{F}^I(\b{Q}) -
\nabla\cdot\b{F}^V(\b{Q},\b{\Sigma})}_{= \nabla\cdot\b{F}(\b{Q},
\b{\Sigma})} &= \b{S} \quad{\text{primary eqn}}
$$

Let $\Omega_h$ denote a collection of disjoint elements $E$. The DG method constructs
approximations $\mathbf{Q}_h$ and $\mathbf{\Sigma}_h$ to $\mathbf{Q}$ and $\mathbf{\Sigma}$,
respectively, in discontinuous finite element spaces. For any integer $k$, we define
the following discontinuous spaces of piecewise (vector-valued) polynomial functions:
$$
\let\b=\mathbf
\b{V}^k_h &= \left\lbrace \b{v} \in L^2(\Omega_h)^N:
\b{v}|_E \in \lbrack P^k(E) \rbrack^N, \text{ for all } E \in \Omega_h
\right\rbrace, \\
\b{W}^k_h &= \left\lbrace \b{w} \in L^2(\Omega_h)^{N\times d}:
\b{w}|_E \in \lbrack P^k(E) \rbrack^{N\times d}, \text{ for all } E \in \Omega_h
\right\rbrace,
$$
where $N = d + 2 + N_s$, $d$ is the spatial dimension, and $N_s$ is the total number of
mixture species. Here, $P^k(E)$ denotes a polynomial space on $E$ consisting of functions
of degree $\leq k$. The DG formulation is obtained by multiplying by ''test functions''
$\mathbf{v}_h \in \mathbf{V}^k_h$, $\mathbf{w}_h \in \mathbf{W}^k_h$
(one for each equation repsectively) and integrating over each element.
The resulting DG problem reads as follows. Find $(\mathbf{Q}_h,
\mathbf{\Sigma}_h) \in \mathbf{V}^k_h \times \mathbf{W}^k_h$ such that, for all
$(\mathbf{v}_h, \mathbf{w}_h) \in \mathbf{V}^k_h \times \mathbf{W}^k_h$, we have:

$$
\let\b=\mathbf
\sum_{E\in\Omega_h} \left\lbrack \int_E \b{v}_h\cdot\frac{\partial \b{Q}_h}
{\partial t} d\Omega + \oint_{\partial E} \b{v}_h\b{h} d\sigma - \int_E \nabla
\b{v}_h\cdot\b{F}(\b{Q}_h, \b{\Sigma}_h)d\Omega\right\rbrack &=
\sum_{E\in\Omega_h} \int_E \b{v}_h\cdot\b{S}_h d\Omega, \\
\sum_{E\in\Omega_h}\left\lbrack \int_E \b{w}_h\cdot \b{\Sigma}_h d\Omega -
\oint_{\partial E} \b{w}_h\cdot\b{H}_s d\sigma + \int_E \nabla\b{w}_h\cdot
\b{Q}_h d\Omega\right\rbrack &= 0,
$$

where $\mathbf{h} = \mathbf{h}_e - \mathbf{h}_v$ is a *numerical flux* consisting of a
*inviscid* and *viscous* component respectively, and $\mathbf{H}_s$ is the
*gradient numerical flux* for the auxiliary equation. Here, we use the subscript "$h$" to
denote discretized quantities. See below for more on these functions.

Since $\mathbf{F}^I(\mathbf{Q}_h)\cdot\mathbf{n}$ is discontinuous, the quantities are
allowed to vary on either side of a shared element boundary. That is, $\mathbf{F}^I
(\mathbf{Q}^+_h)\cdot\mathbf{n}^+ \neq \mathbf{F}^I(\mathbf{Q}^-_h)\cdot\mathbf{n}^-$.
Here, $\mathbf{n}^+$ and $\mathbf{n}^-$ denote the ''cell-local'' (outward pointing) normals
for elements $E^+$ and $E^-$ (respectively) which share a face:
$\partial E^+ \cap \partial E^- \neq \emptyset$.
Similarly for $\mathbf{F}^V(\mathbf{Q}_h, \mathbf{\Sigma}_h)\cdot\mathbf{n}$ and
$\mathbf{Q}_h\mathbf{n}$.

Expanding out the trial and test functions (component-wise) in terms of the local
element basis in each element:
$$
\let\b=\mathbf
\b{Q}(\b{x}, t)_h|_E = \sum_{i=1}^n \b{Q}_i(t)\phi_i^k(\b{x}), \quad
\b{\Sigma}(\b{x})_h|_E = \sum_{i=1}^n \b{\Sigma}_i\phi_i^k(\b{x}), \\
\b{v}(\b{x})_h|_E = \sum_{i=1}^n \b{v}_i\phi_i^k(\b{x}), \quad
\b{w}(\b{x})_h|_E = \sum_{i=1}^n \b{w}_i\phi_i^k(\b{x}),
$$
allows us to obtain a set of algebraic equations for the conserved state $\mathbf{Q}_h$ and
the auxiliary gradient variable $\mathbf{\Sigma}_h$. That is, for each
$j = 1, \dots, \dim P^k$, we have:
$$
\let\b=\mathbf
\frac{d}{dt} \sum_{E\in\Omega_h}\int_E \phi_j^k\b{Q}_h d\Omega &= \sum_{E\in\Omega_h}
\left\lbrack\int_E \nabla\phi_j^k\cdot\left(\b{F}^V(\b{Q}_h, \b{\Sigma}_h) -
\b{F}^I(\b{Q}_h)\right)d\Omega\right\rbrack \\
&- \sum_{E\in\Omega_h}\left\lbrack\oint_{\partial E}\phi_j^k \b{h}_e(\b{Q}_h^+,
\b{Q}^-_h; \b{n}) d\sigma + \oint_{\partial E} \phi_j^k \b{h}_v(\b{Q}_h^+,
\b{\Sigma}_h^+, \b{Q}_h^-, \b{\Sigma}_h^-; \b{n}) d\sigma\right\rbrack \\
&+ \sum_{E\in\Omega_h} \int_E \phi_j^k\b{S}_h d\Omega, \\
\sum_{E\in\Omega_h}\int_E\phi_j^k \b{\Sigma}_h d\Omega &= \sum_{E\in\Omega_h}\left\lbrack
\oint_{\partial{E}}\phi_j^k \b{H}_s(\b{Q}^+_h, \b{Q}_h^-; \b{n}) d\sigma -
\int_E\nabla\phi^k_j\cdot\b{Q}_h d\Omega\right\rbrack.
$$

Numerical fluxes
================

To account for the discontinuities at element faces, DG employs numerical fluxes, which are
enforced to be singled-valued, but are functions of both $\pm$ states:

$$
\let\b=\mathbf
\b{h}_e(\b{Q}_h^+, \b{Q}^-_h; \b{n}) &\approx \b{F}^I(\b{Q}_h)
\cdot\b{n}, \\
\b{h}_v(\b{Q}_h^+, \b{\Sigma}_h^+, \b{Q}_h^-, \b{\Sigma}_h^-;
\b{n}) &\approx \b{F}^V(\b{Q}_h, \b{\Sigma}_h)\cdot\b{n}\\
\b{H}_s(\b{Q}^+_h, \b{Q}_h^-; \b{n}) &\approx \b{Q}_h\b{n}.
$$

Choices of numerical fluxes corresponding to BR1
------------------------------------------------

Numerical inviscid flux
^^^^^^^^^^^^^^^^^^^^^^^

Typical choices for $\mathbf{h}_e(\mathbf{Q}_h^+, \mathbf{Q}^-_h; \mathbf{n})$ include,
but are not limited to:

* Local Lax-Friedrichs (LLF)
* Roe
* Engquist-Osher
* HLLC

|mirgecom| currently uses LLF, which is implemented as follows:
$$
\let\b=\mathbf
\b{h}_{e}(\b{Q}_h^+, \b{Q}^-_h; \b{n}) = \frac{1}{2}\left(
\b{F}^{I}(\b{Q}_h^+)+\b{F}^{I}(\b{Q}_h^-)\right) - \frac{\lambda}
{2}\left(\b{Q}_h^+ - \b{Q}_h^-\right)\b{n},
$$
where $\lambda$ is the characteristic max wave-speed of the fluid. Numerical fluxes
which penalize the ''jump'' of the state $\left(\mathbf{Q}_h^+ - \mathbf{Q}_h^-\right)
\mathbf{n}$ act as an additional source of dissipation, which has a stabilizing effect
on the numerics.

Numerical viscous flux
^^^^^^^^^^^^^^^^^^^^^^
Take $\mathbf{h}_v(\mathbf{Q}_h^+, \mathbf{\Sigma}_h^+, \mathbf{Q}_h^-, \mathbf{\Sigma}_h^-; \mathbf{n})$ to be
defined in a ''centered'' (average) way:
$$
\let\b=\mathbf
\b{h}_v(\b{Q}_h^+, \b{\Sigma}_h^+, \b{Q}_h^-, \b{\Sigma}_h^-;
\b{n}) = \frac{1}{2}\left(\b{F}^V(\b{Q}_h^+, \b{\Sigma}_h^+) +
\b{F}^V(\b{Q}_h^-, \b{\Sigma}_h^-)\right)\cdot\b{n}
$$

Numerical gradient flux
^^^^^^^^^^^^^^^^^^^^^^^
And similarly for the gradient flux:
$$
\let\b=\mathbf
\b{H}_s(\b{Q}_h^+, \b{Q}_h^- ; \b{n}) = \frac{1}{2}\left(
\b{Q}_h^+ + \b{Q}_h^-\right)\b{n}
$$

Domain boundary considerations
------------------------------

What happens when $\partial E \cap \partial\Omega \neq \emptyset$?

In DG, numerical fluxes are not only responsible for handling the flow of information
between adjacent cells, but they also enforce information flow at the boundaries.

We denote the *boundary fluxes* as $\mathbf{h}^*_e(\mathbf{Q}_{bc})$,
$\mathbf{h}^*_v(\mathbf{Q}_{bc}$, $\mathbf{\Sigma}_{bc})$, and
$\mathbf{H}^*_s(\mathbf{Q}_{bc})$, where $\mathbf{Q}_{bc}$, $\mathbf{\Sigma}_{bc}$ denote
boundary conditions imposed on the state, and the gradient of the state respectively.

For all $\partial E \cap \partial\Omega$ there is no $+$ side to consider; just the
interior state ($-$ side) and the prescribed boundary conditions $\mathbf{Q}_{bc},
\mathbf{\Sigma}_{bc}$.

Solid walls
^^^^^^^^^^^

Inviscid boundary flux
""""""""""""""""""""""
$\mathbf{h}^*_e$ is equal to the (interior; - side) pressure contribution of
$\mathbf{F}^I(\mathbf{Q}_{bc})\cdot\mathbf{n}$
(since $\mathbf{V}\cdot\mathbf{n} = 0$).

Viscous boundary flux
"""""""""""""""""""""
$$
\let\b=\mathbf
\b{h}^*_v(\b{Q}_{bc}, \b{\Sigma}_{bc}) = \b{F}_V(\b{Q}_{bc},
\b{\Sigma}_{bc})\cdot\b{n},
$$
where $\mathbf{Q}_{bc}$ are the same values used to prescribe $\mathbf{h}^*_e$.


Gradient boundary flux
""""""""""""""""""""""
If there are no conditions on $\nabla\mathbf{Q}\cdot\mathbf{n}$, then:
$$
\mathbf{\Sigma}_{bc} = \mathbf{\Sigma}_h^-.
$$
Otherwise, $\mathbf{\Sigma}_{bc}$ will need to be modified accordingly.

Inflow/outflow boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^
Inviscid boundary flux
""""""""""""""""""""""
$$
\let\b=\mathbf
\b{h}^*_e(\b{Q}_{bc}) = \b{h}_e(\b{Q}_{bc}, \b{Q}^-_{h};
\b{n}).
$$

Viscous boundary flux
"""""""""""""""""""""
$$
\let\b=\mathbf
\b{h}^*_v = \b{h}_v(\b{Q}_{bc}, \b{\Sigma}_h^-, \b{Q}_h^-,
\b{\Sigma}_h^-; \b{n}),
$$
where $\mathbf{Q}_{bc}$ are the same values used for $\mathbf{h}^*_e$.


Gradient boundary flux
""""""""""""""""""""""
$\mathbf{Q}_{bc}$ is also used to define the gradient boundary flux:
$$
\mathbf{H}^*_s(\mathbf{Q}_{bc}) = \mathbf{Q}_{bc}\mathbf{n}.
$$

Second-order terms on the viscous RHS
=====================================

This section breaks out explicit component versions of the 2nd order terms on the RHS to
help guide a discussion about alternate approaches to discretization in which a generic
diffusion operator could potentially be responsible for integrating some of these terms.

The viscous fluxes $\mathbf{F}^{V}$ are proportional to gradients of the fluid state
variables, introducing 2nd order terms on the RHS of the conservation equations. These 2nd
order terms with their relevant RHS component are summarized below.

Momentum equation
-----------------

The 2nd order terms in the viscous RHS for the momentum equation are:

$$
\partial_j \tau_{ij} = \left[\partial_j\left(\mu\partial_j{v}_i\right) + \partial_j
\left(\mu\partial_i{v}_j\right) + \partial_j\left(\mu_{B} - \frac{2}{3}\mu\right)
\partial_k{v}_k\delta_{ij}\right]
$$

Energy equation
---------------

The 2nd order terms in the energy equation RHS have convective, conductive, and
diffusive terms as follows:

Convective part
^^^^^^^^^^^^^^^
$$
\partial_j \tau_{jk} {v}_k = \left[\partial_j\left(\mu\partial_k{v}_j{v}_k\right) +
\partial_j\left(\mu\partial_j{v}^2_k\right) + \partial_j\left(\mu_{B} - \frac{2}{3}\mu
\right)\partial_m{v}_m\delta_{jk}{v}_k\right]
$$

Conductive part
^^^^^^^^^^^^^^^
The conductive heat part of the RHS is:

$$
\partial_j{(q_{c})_j} = \partial_j\kappa\partial_j{T},
$$

where $T$ is the fluid temperature.

Diffusive part
^^^^^^^^^^^^^^
The diffusive heat part of the RHS is:

$$
\partial_j{(q_{d})_j} = \partial_j\left(\rho{h}_{\alpha}{d}_{(\alpha)}\partial_j
{Y}_{\alpha}\right)
$$

with fluid density $\rho$, species diffusivity ${d}_{(\alpha)}$, and species mass
fractions ${Y}_{\alpha}$.

Species equation
----------------

The species diffusive transport RHS is:

$$
\partial_j{(J_{\alpha})_j} = \partial_j\left(\rho{d}_{(\alpha)}\partial_j{Y}_{\alpha}
\right),
$$

with fluid density $\rho$, species diffusivity ${d}_{(\alpha)}$, and species mass
fractions ${Y}_{\alpha}$.
