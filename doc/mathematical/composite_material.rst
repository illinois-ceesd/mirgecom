Wall Degradation
================

Carbon Fiber Oxidation 
----------------------

    This section covers the response of carbon fiber when exposed to oxygen.
    The carbon fibers are characterized as a highly porous material,
    with void fraction $\approx 90\%$. As the material is exposed to the flow,
    oxygen can diffuse inside the wall and result in oxidation, not only
    limited to the surface but also as a volumetric process. For now, convection
    inside the wall will be neglected and the species are only allowed to diffuse.

    The temporal evolution of mass density is solved in
    order to predict the material degradation. As the
    :class:`~mirgecom.materials.carbon_fiber.Oxidation` progresses,
    the temporal evolution of the fibers mass is given by

    .. math ::
        \frac{\partial \rho_s}{\partial t} = \dot{\omega_s} \mbox{ .}

    This process creates gaseous species following

    .. math ::
        \frac{\partial \rho_g}{\partial t} = - \dot{\omega}_s \mbox{ ,}

    where the
    :attr:`~mirgecom.fluid.ConservedVars.mass`
    is $\rho_g$. The source term indicates that the fibers become gas in order
    to satisfy mass conservation.

    The
    :attr:`~mirgecom.fluid.ConservedVars.energy`
    of the bulk material is given by

    .. math::
        \frac{\partial \rho_b e_b}{\partial t}
        = \nabla \cdot \left( \bar{\boldsymbol{\kappa}} \nabla T 
        + h_\alpha \mathbf{J}_{\alpha} \right)
        \mbox{ .}

    The first term on the RHS is modeled as an effective diffusive transfer
    using Fourier's law, where the thermal conductivity is given by
    $\bar{\boldsymbol{\kappa}}$.
    The second term on the RHS is due to the species diffusion, where the
    species specific enthalpy ${h}_{\alpha}$, and the species
    diffusive flux vector $\mathbf{J}_{\alpha}$. The sub-index $b$ is the bulk
    material consisting of both solid and gas phases, where

    .. math::
        \rho_b e_b = \epsilon_{g} \rho_g e_g + \epsilon_s \rho_s h_s \mbox{ .}

    The internal energy of the gas is given by $e_g(T)$ and the solid 
    energy/enthalpy is $h_s(T)$.

    From the conserved variables, it is possible to compute the oxidation
    progress, denoted by
    :attr:`~mirgecom.wall_model.PorousWallVars.tau`.
    As a consequence, the instantaneous material properties will change due to
    the mass loss.

    The
    :attr:`~mirgecom.eos.GasDependentVars.temperature`
    is evaluated using Newton iteration based on both
    :attr:`~mirgecom.eos.PyrometheusMixture.get_internal_energy` and
    :attr:`~mirgecom.wall_model.PorousWallEOS.enthalpy`,
    as well as their respective derivatives, namely
    :attr:`~mirgecom.eos.PyrometheusMixture.heat_capacity_cv` and
    :attr:`~mirgecom.wall_model.PorousWallEOS.heat_capacity`.
    Note that :mod:`pyrometheus` is used to handle the species properties.

Composite Materials
-------------------

    This section covers the response of composite materials made of phenolic
    resin and carbon fibers.
    Phenolic resins are added to rigidize the fibers by permeating the
    material and filling partially the gaps between fibers, reducing the porousity
    to $\approx 80\%$. As the material is heated up by the flow, the resin
    pyrolyses, i.e., it degrades and produces gaseous species.

    The temporal evolution of wall density is solved in order to predict the
    material degradation. As the
    :class:`~mirgecom.materials.tacot.Pyrolysis` progresses, the mass of each 
    $i$ constituents of the resin, denoted by
    :attr:`~mirgecom.wall_model.PorousWallVars.material_densities`,
    is calculated as

    .. math ::
        \frac{\partial \rho_i}{\partial t} = \dot{\omega}_i \mbox{ .}

    This process creates gaseous species following

    .. math ::
        \frac{\partial \rho_g}{\partial t} + \nabla \cdot \rho_g \mathbf{u} = 
            - \sum_i \dot{\omega}_i \mbox{ ,}

    where the
    :attr:`~mirgecom.fluid.ConservedVars.mass`
    is $\rho_g$. The source term indicates that all solid resin must become
    gas in order to satisfy mass conservation. Lastly, the gas velocity
    $\mathbf{u}$ is obtained by Darcy's law, given by

    .. math::
        \mathbf{u} = \frac{\mathbf{K}}{\mu \epsilon} \cdot \nabla P \mbox{ .}

    In this equation, $\mathbf{K}$ is the second-order permeability tensor,
    $\mu$ is the gas viscosity, $\epsilon$ is the void fraction and $P$ is
    the gas pressure.

    The
    :attr:`~mirgecom.fluid.ConservedVars.energy`
    of the bulk material is given by

    .. math::
        \frac{\partial \rho_b e_b}{\partial t}
        + \nabla \cdot (\epsilon_{g} \rho_g h_g \mathbf{u})
        = \nabla \cdot \left( \bar{\boldsymbol{\kappa}} \nabla T \right)
        + \mu \epsilon_{g}^2 (\bar{\mathbf{K}}^{-1} \cdot \vec{v} ) \cdot \vec{v}
        \mbox{ .}

    The first term on the RHS is modeled as an effective diffusive transfer
    using Fourier's law, where the thermal conductivity is given by
    $\bar{\boldsymbol{\kappa}}$. The second term on the RHS account for the
    viscous dissipation in the Darcy's flow. The sub-index $b$ is the bulk
    material consisting of both solid and gas phases, where

    .. math::
        \rho_b e_b = \epsilon_{g} \rho_g e_g + \epsilon_s \rho_s e_s \mbox{ .}

    The energy of the gas is given by $e_g(T) = h_g(T) - \frac{P}{\rho_g}$,
    where $h_g$ is its enthalpy.

    From the conserved variables, it is possible to compute the decomposition
    status, denoted by
    :attr:`~mirgecom.wall_model.PorousWallVars.tau`.
    This yields the proportion of virgin (unpyrolyzed material) to char (fully
    pyrolyzed) and, consequently, the different thermophysicochemical
    properties of the solid phase. Thus, the instantaneous material properties
    depend on the current state of the material, as well as the 
    :attr:`~mirgecom.eos.GasDependentVars.temperature`.
    It is evaluated using Newton iteration based on
    :attr:`~mirgecom.eos.PyrometheusMixture.get_internal_energy` and
    :attr:`~mirgecom.wall_model.PorousWallEOS.enthalpy`,
    as well as their respective derivatives
    :attr:`~mirgecom.eos.PyrometheusMixture.heat_capacity_cv` and
    :attr:`~mirgecom.wall_model.PorousWallEOS.heat_capacity`.

    In *MIRGE-Com*, the solid properties are obtained by fitting polynomials
    to tabulated data for easy evaluation of the properties based on the
    temperature. The complete list of properties can be find, for instance, in
    :mod:`~mirgecom.materials.tacot`.
    Different materials can be incorporated as separated files.

.. important ::

    The current implementation follows the description of [Lachaud_2014]_ 
    for type 2 code. Additional details, extensive formulation and references
    are provided in https://github.com/illinois-ceesd/phenolics-notes
