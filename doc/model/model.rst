=====
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



.. toctree::

   viscous
   heat-flux
   transport
   power-law
   eos
   ideal-gas
   non-dimen
