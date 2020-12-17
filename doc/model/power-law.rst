Power Law
=========

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
