Non-dimensionalization
======================

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
