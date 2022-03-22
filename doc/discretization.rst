=======================
Discretization Strategy
=======================

.. _disc-strat:

.. raw:: latex

    \let\b=\mathbf

.. raw:: html

    \(
    \let\b=\mathbf
    \)

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
unknowns, $\b{Q}$ and $\b{\Sigma}$:

$$
\b{\Sigma} - \nabla{\b{Q}} &= \b{0}\quad{\text{auxiliary eqn}}\\
\frac{\partial \b{Q}}{\partial t} + \underbrace{\nabla\cdot\b{F}^I(\b{Q}) -
\nabla\cdot\b{F}^V(\b{Q},\b{\Sigma})}_{= \nabla\cdot\b{F}(\b{Q},
\b{\Sigma})} &= \b{S} \quad{\text{primary eqn}}
$$

Let $\Omega_h$ denote a collection of disjoint elements $E$. The DG method constructs
approximations $\b{Q}_h$ and $\b{\Sigma}_h$ to $\b{Q}$ and $\b{\Sigma}$,
respectively, in discontinuous finite element spaces. For any integer $k$, we define
the following discontinuous spaces of piecewise (vector-valued) polynomial functions:
$$
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
$\b{v}_h \in \b{V}^k_h$, $\b{w}_h \in \b{W}^k_h$
(one for each equation repsectively) and integrating over each element.
The resulting DG problem reads as follows. Find $(\b{Q}_h,
\b{\Sigma}_h) \in \b{V}^k_h \times \b{W}^k_h$ such that, for all
$(\b{v}_h, \b{w}_h) \in \b{V}^k_h \times \b{W}^k_h$, we have:

$$
\sum_{E\in\Omega_h} \left\lbrack \int_E \b{v}_h\cdot\frac{\partial \b{Q}_h}
{\partial t} d\Omega + \oint_{\partial E} \b{v}_h\b{h} d\sigma - \int_E \nabla
\b{v}_h\cdot\b{F}(\b{Q}_h, \b{\Sigma}_h)d\Omega\right\rbrack &=
\sum_{E\in\Omega_h} \int_E \b{v}_h\cdot\b{S}_h d\Omega, \\
\sum_{E\in\Omega_h}\left\lbrack \int_E \b{w}_h\cdot \b{\Sigma}_h d\Omega -
\oint_{\partial E} \b{w}_h\cdot\b{H}_s d\sigma + \int_E \nabla\b{w}_h\cdot
\b{Q}_h d\Omega\right\rbrack &= 0,
$$

where $\b{h} = \b{h}_e - \b{h}_v$ is a *numerical flux* respectively representing
the inviscid and viscous contributions to the physical flux
$\left(\b{F}^I(\b{Q}_h) - F^V(\b{Q}_h, \nabla\b{Q}_h)\right)\cdot\b{n}$, with
$\b{n}$ being the outward facing unit normal with respect to the local element.
$\b{H}_s$ is the *gradient numerical flux* ($\b{Q}_h\b{n}$) for the auxiliary equation.
Here, we use the subscript "$h$" to denote discretized quantities. See below for more
on these functions.

Since $\b{F}^I(\b{Q}_h)\cdot\b{n}$ is discontinuous, the quantities are
allowed to vary on either side of a shared element boundary. That is, $\b{F}^I
(\b{Q}^+_h)\cdot\b{n}^+ \neq \b{F}^I(\b{Q}^-_h)\cdot\b{n}^-$.
Here, $\b{n}^+$ and $\b{n}^-$ denote the ''cell-local'' (outward pointing) unit normals
for elements $E^+$ and $E^-$ (respectively) which share a face:
$\partial E^+ \cap \partial E^- \neq \emptyset$.
Similarly for $\b{F}^V(\b{Q}_h, \b{\Sigma}_h)\cdot\b{n}$ and
$\b{Q}_h\b{n}$.

Expanding out the trial and test functions (component-wise) in terms of the local
element basis in each element:
$$
\b{Q}(\b{x}, t)_h|_E = \sum_{i=1}^n \b{Q}_i(t)\phi_i^k(\b{x}), \quad
\b{\Sigma}(\b{x})_h|_E = \sum_{i=1}^n \b{\Sigma}_i\phi_i^k(\b{x}), \\
\b{v}(\b{x})_h|_E = \sum_{i=1}^n \b{v}_i\phi_i^k(\b{x}), \quad
\b{w}(\b{x})_h|_E = \sum_{i=1}^n \b{w}_i\phi_i^k(\b{x}),
$$
allows us to obtain a set of algebraic equations for the conserved state $\b{Q}_h$ and
the auxiliary gradient variable $\b{\Sigma}_h$. That is, for each
$j = 1, \dots, \dim P^k$, we have:
$$
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
\b{h}_e(\b{Q}_h^+, \b{Q}^-_h; \b{n}) &\approx \b{F}^I(\b{Q}_h)
\cdot\b{n}, \\
\b{h}_v(\b{Q}_h^+, \b{\Sigma}_h^+, \b{Q}_h^-, \b{\Sigma}_h^-;
\b{n}) &\approx \b{F}^V(\b{Q}_h, \b{\Sigma}_h)\cdot\b{n}\\
\b{H}_s(\b{Q}^+_h, \b{Q}_h^-; \b{n}) &\approx \b{Q}_h\b{n}.
$$

Choices of numerical fluxes corresponding to BR1
------------------------------------------------

Inviscid numerical flux
^^^^^^^^^^^^^^^^^^^^^^^

Typical choices for $\b{h}_e(\b{Q}_h^+, \b{Q}^-_h; \b{n})$ include,
but are not limited to:

* Local Lax-Friedrichs (LLF)
* Roe
* Engquist-Osher
* HLLC

|mirgecom| currently uses LLF, which is implemented as follows:
$$
\b{h}_{e}(\b{Q}_h^+, \b{Q}^-_h; \b{n}) = \frac{1}{2}\left(
\b{F}^{I}(\b{Q}_h^+)+\b{F}^{I}(\b{Q}_h^-)\right) - \frac{\lambda}
{2}\left(\b{Q}_h^+ - \b{Q}_h^-\right)\b{n},
$$
where $\lambda$ is the characteristic max wave-speed of the fluid. Numerical fluxes
which penalize the ''jump'' of the state $\left(\b{Q}_h^+ - \b{Q}_h^-\right)
\b{n}$ act as an additional source of dissipation, which has a stabilizing effect
on the numerics.

Viscous numerical flux
^^^^^^^^^^^^^^^^^^^^^^
Take $\b{h}_v(\b{Q}_h^+, \b{\Sigma}_h^+, \b{Q}_h^-, \b{\Sigma}_h^-; \b{n})$ to be
defined in a ''centered'' (average) way:
$$
\b{h}_v(\b{Q}_h^+, \b{\Sigma}_h^+, \b{Q}_h^-, \b{\Sigma}_h^-;
\b{n}) = \frac{1}{2}\left(\b{F}^V(\b{Q}_h^+, \b{\Sigma}_h^+) +
\b{F}^V(\b{Q}_h^-, \b{\Sigma}_h^-)\right)\cdot\b{n}
$$

Gradient numerical flux
^^^^^^^^^^^^^^^^^^^^^^^
And similarly for the gradient flux:
$$
\b{H}_s(\b{Q}_h^+, \b{Q}_h^- ; \b{n}) = \frac{1}{2}\left(
\b{Q}_h^+ + \b{Q}_h^-\right)\b{n}
$$

It is worth noting here that when $\b{Q}_h^+ = \b{Q}_h^-$, then there is no change in
field value across the boundary, resulting in a $\nabla{\b{Q}}$ which vanishes in the
normal direction corresponding to that boundary or face.

That is, if $\b{Q}_h^+ = \b{Q}_h^-$, then $\nabla{\b{Q}} \cdot \hat{\b{n}} = 0$.

Domain boundary treatments
==========================

What happens when $\partial E \cap \partial\Omega \neq \emptyset$?

In DG, numerical fluxes are not only responsible for handling the flow of information
between adjacent cells, but they also enforce information flow at the boundaries.

The relevant quantities for the boundary treatments are as follows:

.. math::
   
  \b{Q}^- &\equiv \text{solution on the interior of the boundary face} \\
  \b{\Sigma}^- &\equiv \text{gradient of solution on interior of boundary face} \\
  \b{Q}_{bc} &\equiv \text{solution on the exterior of the boundary face (boundary soln)} \\
  \b{\Sigma}_{bc} &\equiv \text{grad of soln on exterior of boundary face} \\
  \b{h}^*_e(\b{Q}_{bc}) &\equiv \text{boundary flux for the divergence of inviscid flux} \\
  \b{h}^*_v(\b{Q}_{bc}, \b{\Sigma}_{bc}) &\equiv \text{bndry flux for divergence of viscous flux} \\
  \b{H}^*(\b{Q}_{bc}) &\equiv \text{boundary flux for the gradient of the solution} \\
  \hat{\b{n}} &\equiv \text{outward pointing normal for the boundary face}

For all $\partial E \cap \partial\Omega$ the $+$ side is on the domain boundary. 
Boundary conditions are set by prescribing one or more components of the solution
or its gradient, $\b{Q}^+ = \b{Q}_{bc}$, and $\b{\Sigma}^+ = \b{\Sigma}_{bc}$, respectively,
or by prescribing one or more components of the boundary fluxes $\b{h}^*_e$, $\b{h}^*_v$,
and $\b{H}^*_s$.  Descriptions of particular boundary treatments follow in the next few
sections.


Solid walls
-----------

There are a few versions of solid wall treatments implemented in mirgecom:

1. Adiabatic slip wall
2. Adiabatic noslip wall
3. Isothermal noslip wall

Common to all implemented wall boundary treatments, we start by calculating or prescribing a
boundary solution, $\b{Q}_{bc}$, for the exterior of the boundary face.  The following
sections will describe how each of the wall treatments compute the boundary solution,
and then the remaining relevant quantities described above.

Adiabtic slip wall
^^^^^^^^^^^^^^^^^^

The adiabatic slip wall is an inviscid-only boundary condition.  The boundary solution
is prescribed as follows:

.. math::

   \b{Q}_{bc} = \b{Q}^- - 2*\left(\rho\b{v}^-\cdot\hat{\b{n}}\right)\hat{\b{n}},

where $\b{v}^-$ is the fluid velocity corresponding to $\b{Q}^-$.

The flux for the divergence of the inviscid flux is then calculated with the same numerical
flux function as used in the volume: $\b{h}^*_e = \b{h}_{e}(\b{Q}^-, \b{Q}_{bc})$.


No-slip walls
^^^^^^^^^^^^^

Boundary solution
"""""""""""""""""

For walls enforcing a no-slip condition, we choose the "no-slip boundary solution" as:

.. math::

   \b{Q}_{bc} = \b{Q}^- - 2\rho\b{v}^-,

where $\b{v}^-$ is the fluid velocity corresponding to $\b{Q}^-$.

Gradient boundary flux
""""""""""""""""""""""

The boundary flux for $\nabla{\b{Q}}$ at the boundary is computed with a central
flux as follows:

.. math::

   \b{H}^*(\b{Q}_{bc}) = \b{H}_s(\b{Q}^-, \b{Q}_{bc}) = \frac{1}{2}\left(\b{Q}^- + \b{Q}_{bc}\right)\b{n},

using the no-slip boundary solution, $\b{Q}_{bc}$, as defined above.

Since:

.. math::

   \rho^+ &= \rho^- \\
   (\rho{E})^+ &= (\rho{E})^- \\
   (\rho{Y})^+ &= (\rho{Y})^-,

we expect:

.. math::

   \nabla(\rho) \cdot \hat{\b{n}} &= 0 \\
   \nabla(\rho{E}) \cdot \hat{\b{n}} &= 0 \\
   \nabla(\rho{Y}) \cdot \hat{\b{n}} &= 0

We compute $\nabla{Y}$ and $\nabla{E}$ from the product rule:

.. math::

   \nabla{Y} &= \frac{1}{\rho}\left(\nabla{(\rho{Y})} - Y\nabla{\rho}\right) \\  
   \nabla{E} &= \frac{1}{\rho}\left(\nabla{(\rho{E})} - E\nabla{\rho}\right)

So we likewise expect:

.. math::

   \nabla{Y} \cdot \hat{\b{n}} &= 0 \\
   \nabla{E} \cdot \hat{\b{n}} &= 0

Inviscid boundary flux
""""""""""""""""""""""

The inviscid boundary flux is calculated from the numerical flux function
used for inviscid interfacial fluxes in the volume:

.. math::

   \b{h}^*_e = \b{h}_e(\b{Q}^-, \b{Q}_{bc})

Intuitively, we expect $\b{h}^*_e$ is equal to the (interior; - side) pressure contribution of
$\b{F}^I(\b{Q}_{bc})\cdot\b{n}$ (since $\b{V}\cdot\b{n} = 0$).

Viscous boundary flux
"""""""""""""""""""""

*MIRGE-Com* has a departure from BR1 for the computation of viscous fluxes.  This section
will describe both the viscous flux calculation prescribed by BR1, and also what
*MIRGE-Com* is currently doing.

---------

BR1 prescribes the following boundary treatment:

The viscous boundary flux at solid walls is computed as:

.. math::

   \b{h}^*_v(\b{Q}_{bc}, \b{\Sigma}_{bc}) = \b{F}_V(\b{Q}_{bc},\b{\Sigma}_{bc})\cdot\b{n},

where $\b{Q}_{bc}$ are the same values used to prescribe $\b{h}^*_e$.

If there are no conditions on $\nabla\b{Q}\cdot\b{n}$, then:
$$
\b{\Sigma}_{bc} = \b{\Sigma}_h^-.
$$
Otherwise, $\b{\Sigma}_{bc}$ will need to be modified accordingly.

--------

MIRGE-Com currently does the following:

.. math::

   \b{h}^*_v(\b{Q}_{bc}, \b{\Sigma}_{bc}) = \b{h}_v\left(\b{Q}^-,\b{\Sigma}^-,\b{Q}_{bc},\b{\Sigma}_{bc}\right),

where $\b{Q}_{bc}$ are the same values used to prescribe $\b{h}^*_e$.



Inflow/outflow boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^
Inviscid boundary flux
""""""""""""""""""""""
$$
\b{h}^*_e(\b{Q}_{bc}) = \b{h}_e(\b{Q}_{bc}, \b{Q}^-_{h};
\b{n}).
$$

Viscous boundary flux
"""""""""""""""""""""
$$
\b{h}^*_v = \b{h}_v(\b{Q}_{bc}, \b{\Sigma}_h^-, \b{Q}_h^-,
\b{\Sigma}_h^-; \b{n}),
$$
where $\b{Q}_{bc}$ are the same values used for $\b{h}^*_e$.


Gradient boundary flux
""""""""""""""""""""""
$\b{Q}_{bc}$ is also used to define the gradient boundary flux:
$$
\b{H}^*_s(\b{Q}_{bc}) = \b{Q}_{bc}\b{n}.
$$
