Composite Materials
===================

    The time evolution of
    :class:`~mirgecom.phenolics.phenolics.PhenolicsConservedVars` is solved in
    order to predict the material degradation when exposed to hot gases. As
    the material heats up, the :class:`~mirgecom.phenolics.tacot.Pyrolysis` of
    a solid phenolic resin creates gaseous species that, in turn, flow outwards.
    Thus, the instantaneous material properties depend on both pyrolysis
    progress as well as the temperature. 

    From the conserved variables, it is possible to extract the spatial
    advancement of the degradation by evaluating
    :attr:`~mirgecom.phenolics.phenolics.PhenolicsDependentVars.tau`.
    This yields the proportion of virgin (unpyrolyzed material) to char (fully
    pyrolyzed) and, consequently, the different thermophysicochemical properties
    of the solid phase.

    Then, :attr:`~mirgecom.phenolics.phenolics.PhenolicsDependentVars.temperature`
    can be evaluated using Newton iterations based on both
    :attr:`~mirgecom.phenolics.phenolics.PhenolicsEOS.gas_enthalpy` and
    :attr:`~mirgecom.phenolics.phenolics.PhenolicsEOS.solid_enthalpy`,
    as well as their derivative
    :attr:`~mirgecom.phenolics.phenolics.PhenolicsEOS.gas_heat_capacity_cp` and
    :attr:`~mirgecom.phenolics.phenolics.PhenolicsEOS.solid_heat_capacity_cp`.

    Solid properties are obtained by fitting polynomials to tabulated data for
    easy evaluation of the properties based on the temperature. The complete 
    list of properties can be find, for instance, in
    :mod:`~mirgecom.phenolics.tacot`.
    Different materials can be incorporated as separated files.
    
    :class:`~mirgecom.phenolics.gas.GasProperties` are obtained based on
    tabulated data, assuming chemical equilibrium, and evaluated with splines
    for interpolation of the entries.

.. important ::

    Additional details and formulation are provided in
    https://github.com/illinois-ceesd/phenolics-notes

.. automodule:: mirgecom.phenolics
.. automodule:: mirgecom.phenolics.phenolics
.. automodule:: mirgecom.phenolics.tacot
.. automodule:: mirgecom.phenolics.gas
