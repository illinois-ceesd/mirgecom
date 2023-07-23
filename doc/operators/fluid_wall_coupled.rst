Coupled Fluid-Wall Operators
============================

1) :class:`~mirgecom.multiphysics.thermally_coupled_fluid_wall`.

2) :class:`~mirgecom.multiphysics.multiphysics_coupled_fluid_wall`.

3) :class:`~mirgecom.materials.initializer`.

Heat conduction coupling
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: mirgecom.multiphysics.thermally_coupled_fluid_wall

Wall variables
--------------
.. autoclass:: mirgecom.wall_model.SolidWallConservedVars
.. autoclass:: mirgecom.wall_model.SolidWallDependentVars
.. autoclass:: mirgecom.wall_model.SolidWallState

EOS
---
.. autoclass:: mirgecom.wall_model.SolidWallModel

Model-specific properties
-------------------------
    The properties are defined exclusively at the driver.

Porous flow coupling
^^^^^^^^^^^^^^^^^^^^
FIXME: improve documentation

.. automodule:: mirgecom.multiphysics.multiphysics_coupled_fluid_wall

Wall variables
--------------
.. autoclass:: mirgecom.wall_model.PorousWallVars

Porous Media EOS
----------------
.. autoclass:: mirgecom.wall_model.PorousFlowModel

Model-specific properties
-------------------------
    The properties of the materials are defined in specific files and used by
    :class:`~mirgecom.wall_model.PorousWallProperties`.

.. autoclass:: mirgecom.wall_model.PorousWallProperties

Carbon fiber
""""""""""""
.. automodule:: mirgecom.materials.carbon_fiber

TACOT
"""""
.. automodule:: mirgecom.materials.tacot

Helper Functions
^^^^^^^^^^^^^^^^
.. automodule:: mirgecom.materials.initializer
