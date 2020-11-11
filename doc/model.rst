Model
=====

.. note::

    This is mildly converted from PlasCom2 for now, sourced from `here
    <https://bitbucket.org/xpacc-dev/plascom2/src/GoldenCopyCandidate/doc/Theory.dox>`__.
    Do not consider this information authoritative while this notice is still
    present.

.. raw:: latex

    \def\RE{\operatorname{RE}}
    \def\PR{\operatorname{PR}}

.. raw:: html

    \(
    \def\RE{\operatorname{RE}}
    \def\PR{\operatorname{PR}}
    \)

In fluid domains, the code solves the compressible Navier-Stokes equations
in curvilinear coordinates.  The basic equations, in a Cartesian coordinate space, for the conserved mass
density $\rho$, momentum density $\rho u_i$, and total energy
density $\rho E$ are, in index form with summation convention are given as

.. math::
    \frac{\partial \rho}{\partial t} + \frac{\partial }{\partial x_j} \rho u_j &= S_\rho \\
    \frac{\partial \rho u_i}{\partial t} + \frac{\partial}{\partial x_j}\left(\rho u_i u_j + p\delta_{ij} - \tau_{ij}\right) &= S_{\rho u_i} \\
    \frac{\partial \rho E}{\partial t} + \frac{\partial}{\partial x_j}\left(\left\{\rho E + p\right\}u_j + q_j - u_i \tau_{ij}\right) &= S_{\rho E},

where $p$ is the thermodynamic pressure, $\tau_{ij}$ is the
viscous stress tensor, and $q_i$ is the heat flux in the $i$th
direction. $S_\rho$, $S_{\rho u_i}$, and $S_{\rho E}$ are are mass, momentum, and energy density source terms.  These equations can be written in the compact form

.. math::

    \frac{\partial Q}{\partial t} + \frac{\partial \vec{F}_j}{\partial x_j} = S,

where $Q = [\rho\,\rho \vec{u}\,\rho E]^T$ is the vector of conserved
variables, $\vec{F} = \vec{F}^I - \vec{F}^V$ is the flux vector account
for both visicd and inviscid terms, and $S$ is the source term vector.

Viscous stress constitutive relation
------------------------------------

The viscous stress tensor is defined as

.. math::
    \tau_{ij} = \mu \left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right) + \lambda \frac{\partial u_k}{\partial x_k}\delta_{ij}

where $\mu$ and $\lambda$ are the first and second coefficients
of viscosity, respectively; both may be a function of temperature.  Note
that Stokes' hypothesis $(\lambda = -\frac{2}{3}\mu)$ is not
automatically enforced and that $\lambda$ is related to bulk
viscosity $\mu_B$ as $\lambda = \mu_B - (2/3)\mu$.

.. _heat-flux-constitutive:

Heat flux constitutive relation
-------------------------------

The heat flux vector is defined as

.. math::
    q_i = - \kappa \frac{\partial T}{\partial x_i}

where $\kappa$ is the thermal conductivity.

Transport Coefficient Models
----------------------------

The first viscosity coefficient $\mu$, bulk viscosity coefficient,
$\mu_B$, and the thermal conductivity $k$ depend on the thermodynamic
state of the fluid.

Power Law
---------

The power law model gives the dynamic viscosity, $\mu$ as

.. math::
    \mu = \beta T^n

where $\beta$ and $n$ are user specified parameters,
typically $n = 0.666$ and $\beta = 4.093 x 10^{-7}$ for air.

The bulk viscosity is defined as

.. math::
    \mu_B = \alpha \mu

where $\alpha$ is a user specified parameter, typically $\alpha = 0.6$ for air.

Thus the second coefficient of viscosity can be calculated as

.. math::
    \lambda = \left(\alpha - 2/3\right) \mu

The power law model calculates the (TODO)

Equations of State
------------------

The equations of state provides closure by relating the intensive state variables,
pressure and temperature, to the extensive state variables, specific internal energy and volume.

Calorically perfect ideal gas
-----------------------------

The equation of state currently available is that of an ideal gas,
assuming constant specific heats.  The equations of state are

.. math::
    P = \rho R T

where $R$ is the specific gas constant, defined as $R = R_u / W$ with
$R_u$ the universal gas constant, and $W$ the molecular weight.

The specific heat capacity at constant volume and pressure are defined as

.. math::
    C_v &= \left(\frac{\partial E}{\partial T}\right)_v  \\
    C_p &= \left(\frac{\partial H}{\partial T}\right)_p

Then, by substitution into the equation of state we get the following relation

.. math::
    R = C_p - C_v

By defining the specific heat ratio, $\gamma = \frac{C_p}{C_v}$, the
following expressions give the relationship between specific energy, pressure,
and temperature.

.. math::
    P &= (\gamma -1) \rho e \\
    T &= \frac{\gamma-1}{R} e

Non-dimensionalization
----------------------

\PC2 can run in either a dimensional or non-dimensional mode.
The code uses the following variables to define the non-dimensional scaling:

$\rho^*_\infty$, $P^*_\infty$,
$T^*_\infty$, and $L^*$,
a length scale.  Where $*$ denotes a dimensional value and $\infty$ denotes
the reference state. There are two optional non-dimensional spaces available to the user, as shown in the table below.

====================================================================== =============================================================================
Standard (``nonDimensional=1``)                                        Legacy PlasComCM (``nonDimensional=2``)
====================================================================== =============================================================================
$u^*_\infty = \sqrt \frac{P^*_\infty}{\rho^*_\infty}$                  $u^*_\infty = \sqrt \frac{\gamma P^*_\infty}{\rho^*_\infty}$
$e^*_\infty = (u^*_\infty)^2 = \frac{P^*_\infty}{\rho^*_\infty}$       $e^*_\infty = (u^*_\infty)^2 = \frac{\gamma P^*_\infty}{\rho^*_\infty}$
$\rho = \rho^* /\rho^*_\infty$                                         $\rho = \rho^* /\rho^*_\infty$
$P = P^* /P^*_\infty$                                                  $P = P^* /(\rho^*_\infty (u^*_\infty)^2)$
$T = T^* /T^*_\infty$                                                  $T = T^* /((\gamma-1)T^*_\infty)$
$u_i = u^*_i /u^*_\infty$                                              $u_i = u^*_i /u^*_\infty$
$e = e^* /e^*_\infty$                                                  $e = e^* /e^*_\infty$
$t = t^* /(L^* / u^*_\infty)$                                          $t = t^* /(L^* / u^*_\infty)$
$x_i = x_i^* /L^*$                                                     $x_i = x_i^* /L^*$
====================================================================== =============================================================================

Substitution into the dimensional form of the Navier-Stokes equations yields
the non-dimensional equivalent

.. math::
    \frac{\partial \rho}{\partial t} + \frac{\partial }{\partial x_j} \rho u_j &=
       S_\rho \\
    \frac{\partial \rho u_i}{\partial t} + \frac{\partial}{\partial x_j}\left(\rho u_i u_j
       + p\delta_{ij} - \tau_{ij}\right) &= S_{\rho u_i} \\
    \frac{\partial \rho E}{\partial t} +
      \frac{\partial}{\partial x_j}\left(\left\{\rho E + p\right\}u_j +
      q_j - u_i \tau_{ij}\right) &= S_{\rho E}

with the following non-dimensionalization for the source terms

.. math::
    S_\rho        &= \frac{S^*_\rho L^*}{\rho^*_\infty U^*_\infty} \\
    S_{\rho u_i}  &= \frac{S^*_{\rho u_i } L^*}{\rho^*_\infty (U^*_\infty)^2 } \\
    S_{\rho E}    &= \frac{S^*_{\rho E} L^*}{\rho^*_\infty (U^*_\infty)^3}

by choosing the following non-dimensionalizations for the transport coefficients

.. math::
    \mu       &= \mu^* /\mu^*_\infty \\
    \lambda   &= \lambda^* /\lambda^*_\infty \\
    \kappa   &= \kappa^* /\kappa^*_\infty \\

the non-dimensional viscous stress tensor and heat flux vector can be written as

.. math::
    \tau_{ij} &= \frac{\mu}{\RE} \left(\frac{\partial u_i}{\partial x_j} +
      \frac{\partial u_j}{\partial x_i}\right) +
      \frac{\lambda}{\RE} \frac{\partial u_k}{\partial x_k}\delta_{ij} \\
    q_i &= - \frac{\mu}{\RE \Pr} \frac{\partial T}{\partial x_i}

where $\RE$ is defined as the code Reynolds number,
$\RE = \frac{\rho^*_\infty U^*_\infty L^*}{\mu^*_\infty}$
and \PR is defined as the Prandtl number,
$\PR = \frac{(C^*_p)_\infty\mu^*_\infty}{k^*_\infty} = \frac{C_p\mu}{k}$
which define the dimensional reference values $\mu^*_\infty$ and $\kappa^*_\infty$ respectively.

Non-dimensional equation of state
---------------------------------

There are no special modifications to the calorically perfect gas equation of
state, with the exception of the specific gas constant. The reference gas
constant is calculated and non-dimensionalized as follows

.. math::
    R^*_\infty     &= \frac{P^*_\infty}{\rho^*_\infty T^*_\infty} \\
    R       &= R^* /R^*_\infty \\

For the standard non-dimensionalization, $R$ is exactly 1.0. For the legacy
non-dimensionalization, $R = \frac{\gamma-1}{\gamma}$.
