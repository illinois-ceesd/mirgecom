Wall Degradation Modeling
=========================

Conserved Quantities
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: mirgecom.multiphysics.wall_model.WallConservedVars

Equations of State
^^^^^^^^^^^^^^^^^^
.. autoclass:: mirgecom.multiphysics.wall_model.WallEOS
.. autoclass:: mirgecom.multiphysics.wall_model.WallDependentVars
.. autoclass:: mirgecom.multiphysics.oxidation.OxidationWallModel
.. autoclass:: mirgecom.multiphysics.phenolics.PhenolicsWallModel

Model-specific properties
^^^^^^^^^^^^^^^^^^^^^^^^^
    The properties of the materials can be defined in specific files:


Carbon fiber
------------
.. automodule:: mirgecom.materials.carbon_fiber

TACOT
-----
.. automodule:: mirgecom.materials.tacot

HARLEM
------
.. automodule:: mirgecom.materials.harlem

Carbon Fiber Oxidation 
^^^^^^^^^^^^^^^^^^^^^^

    This section covers the response of carbon fiber when exposed to oxygen.
    The specific files used are:

    .. automodule:: mirgecom.multiphysics.oxidation

Composite Materials
^^^^^^^^^^^^^^^^^^^

    This section covers the response of composite materials made of phenolic
    resin and carbon fibers. The specific files used are:

    .. automodule:: mirgecom.multiphysics.phenolics

    The carbon fibers are characterized as a highly porous material,
    with void fraction $\approx 90\%$. Phenolic resins are added to rigidize
    the fibers by permeating the material and filling partially the gaps between
    fibers. As the material is heated up by the flow, the resin pyrolysis, i.e.,
    it degrades and produces gaseous species.

    The temporal evolution of
    :class:`~mirgecom.multiphysics.wall_model.WallConservedVars` is solved in
    order to predict the material degradation. As the
    :class:`~mirgecom.materials.tacot.Pyrolysis` progresses, the mass of each 
    $i$ constituents of the resin, denoted by
    :attr:`~mirgecom.multiphysics.wall_model.WallConservedVars.mass`,
    is calculated as

    .. math ::
        \frac{\partial \rho_i}{\partial t} = \dot{\omega}_i \mbox{ .}

    This process creates gaseous species following

    .. math ::
        \frac{\partial \rho_g}{\partial t} + \nabla \cdot \rho_g \mathbf{u} = 
            - \sum_i \dot{\omega}_i \mbox{ ,}

    where the
    :attr:`~mirgecom.fluid.ConservedVars.mass`
    is $\rho_g$. The source term indicates that all solid resin must become gas
    in order to stisfy mass conservation. Lastly, the gas velocity $\mathbf{u}$
    is obtained by Darcy's law, given by

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
    :attr:`~mirgecom.multiphysics.wall_model.WallDependentVars.tau`.
    This yields the proportion of virgin (unpyrolyzed material) to char (fully
    pyrolyzed) and, consequently, the different thermophysicochemical
    properties of the solid phase. Thus, the instantaneous material properties
    depend on the current state of the material, as well as the 
    :attr:`~mirgecom.eos.GasDependentVars.temperature`.
    It is evaluated using Newton iteration based on both
    :attr:`~mirgecom.materials.tacot.GasProperties.gas_enthalpy` (tabulated
    data) or 
    :attr:`~mirgecom.eos.PyrometheusMixture.get_internal_energy` (Pyrometheus)
    and
    :attr:`~mirgecom.multiphysics.phenolics.PhenolicsWallModel.solid_enthalpy`,
    as well as their respective derivatives, namely
    :attr:`~mirgecom.materials.tacot.GasProperties.gas_heat_capacity` and
    :attr:`~mirgecom.multiphysics.phenolics.PhenolicsWallModel.solid_heat_capacity`.

    In *MIRGE-Com*, the solid properties are obtained by fitting polynomials
    to tabulated data for easy evaluation of the properties based on the
    temperature. The complete list of properties can be find, for instance, in
    :mod:`~mirgecom.materials.tacot`.
    Different materials can be incorporated as separated files.
    
    The
    :class:`~mirgecom.materials.tacot.GasProperties` are obtained based on
    tabulated data, assuming chemical equilibrium, and evaluated with splines
    for interpolation of the entries. However, the data is currently obtained
    by PICA.

.. important ::

    The current implementation follows the description of [Lachaud_2014]_ 
    for type 2 code. Additional details, extensive formulation and references
    are provided in https://github.com/illinois-ceesd/phenolics-notes

Helper Functions
----------------
.. autofunction:: mirgecom.multiphysics.phenolics.initializer
