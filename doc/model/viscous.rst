Viscous stress constitutive relation
====================================

The viscous stress tensor is defined as

.. math::
    \tau_{ij} = \mu \left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right) + \lambda \frac{\partial u_k}{\partial x_k}\delta_{ij}

where $\mu$ and $\lambda$ are the first and second coefficients
of viscosity, respectively; both may be a function of temperature.  Note
that Stokes' hypothesis $(\lambda = -\frac{2}{3}\mu)$ is not
automatically enforced and that $\lambda$ is related to bulk
viscosity $\mu_B$ as $\lambda = \mu_B - (2/3)\mu$.
