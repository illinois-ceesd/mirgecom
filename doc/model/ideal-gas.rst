Calorically perfect ideal gas
=============================

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
