from abc import ABC, abstractmethod
from typing import ClassVar, Hashable, Optional, Sequence

import numpy as np

import meshmode.mesh.generation as mgen
from meshmode.mesh import Mesh
from meshmode.mesh.io import read_gmsh
from meshmode.mesh import TensorProductElementGroup


class MeshBuilder(ABC):
    resolutions: ClassVar[Sequence[Hashable]]
    ambient_dim: ClassVar[int]

    @abstractmethod
    def get_mesh(self, resolution: Hashable,
                 mesh_order: Optional[int] = None) -> Mesh:
        ...


class _GmshMeshBuilder(MeshBuilder):
    resolutions: ClassVar[Sequence[Hashable]] = [None]

    def __init__(self, filename: str) -> None:
        self._mesh_fn = filename

    def get_mesh(self, resolution, mesh_order=None) -> Mesh:
        assert resolution is None
        assert mesh_order is None
        return read_gmsh(self._mesh_fn, force_ambient_dim=self.ambient_dim)


class GmshMeshBuilder2D(_GmshMeshBuilder):
    ambient_dim = 2


class GmshMeshBuilder3D(_GmshMeshBuilder):
    ambient_dim = 3


class Curve2DMeshBuilder(MeshBuilder):
    ambient_dim = 2
    resolutions: ClassVar[Sequence[Hashable]] = [16, 32, 64, 128]

    def get_mesh(self, resolution, mesh_order=None):
        if mesh_order is None:
            mesh_order = 4
        return mgen.make_curve_mesh(
            self.curve_fn,      # pylint: disable=no-member
            np.linspace(0.0, 1.0, resolution + 1),
            mesh_order)


class EllipseMeshBuilder(Curve2DMeshBuilder):
    def __init__(self, radius=3.1, aspect_ratio=2):
        self.radius = radius
        self.aspect_ratio = aspect_ratio

    @property
    def curve_fn(self):
        return lambda t: self.radius * mgen.ellipse(self.aspect_ratio, t)


class StarfishMeshBuilder(Curve2DMeshBuilder):
    narms = 5
    amplitude = 0.25

    @property
    def curve_fn(self):
        return mgen.NArmedStarfish(self.narms, self.amplitude)


class SphereMeshBuilder(MeshBuilder):
    ambient_dim = 3

    resolutions: ClassVar[Sequence[Hashable]] = [0, 1, 2, 3]

    radius: float

    def __init__(self, radius=1):
        self.radius = radius

    def get_mesh(self, resolution, mesh_order=4):
        from meshmode.mesh.generation import generate_sphere
        return generate_sphere(self.radius, order=mesh_order,
                uniform_refinement_rounds=resolution)


class SpheroidMeshBuilder(MeshBuilder):
    ambient_dim = 3

    resolutions: ClassVar[Sequence[Hashable]] = [0, 1, 2, 3]

    radius: float
    aspect_ratio: float

    def __init__(self, radius=1, aspect_ratio=2):
        self.radius = radius
        self.aspect_ratio = aspect_ratio

    def get_mesh(self, resolution, mesh_order=4):
        from meshmode.mesh.generation import generate_sphere
        mesh = generate_sphere(self.radius, order=mesh_order,
                uniform_refinement_rounds=resolution)

        from meshmode.mesh.processing import affine_map
        return affine_map(mesh, A=np.diag([1.0, 1.0, self.aspect_ratio]))


class _BoxMeshBuilderBase(MeshBuilder):
    resolutions: ClassVar[Sequence[Hashable]] = [3, 5, 9, 12]
    mesh_order = 1
    group_cls = None
    a = (-0.5, -0.5, -0.5)
    b = (+0.5, +0.5, +0.5)
    tpe: bool = False

    def __init__(self, tpe=False, a=(-0.5, -0.5, -0.5),
                 b=(0.5, 0.5, 0.5)):
        self.tpe = tpe
        self.a = a
        self.b = b

    def get_mesh(self, resolution, mesh_order=None):
        if mesh_order is None:
            mesh_order = self.mesh_order
        if not isinstance(resolution, (list, tuple)):
            resolution = (resolution,) * self.ambient_dim
        group_cls = TensorProductElementGroup if self.tpe else None
        return mgen.generate_regular_rect_mesh(
            a=self.a, b=self.b,
            nelements_per_axis=resolution,
            group_cls=group_cls, order=mesh_order)


class BoxMeshBuilder1D(_BoxMeshBuilderBase):
    ambient_dim = 1


class BoxMeshBuilder2D(_BoxMeshBuilderBase):
    ambient_dim = 2


class BoxMeshBuilder3D(_BoxMeshBuilderBase):
    ambient_dim = 3


class WarpedRectMeshBuilder(MeshBuilder):
    resolutions: ClassVar[Sequence[Hashable]] = [4, 6, 8]

    def __init__(self, dim):
        self.dim = dim

    def get_mesh(self, resolution, mesh_order=4):
        return mgen.generate_warped_rect_mesh(
            dim=self.dim, order=mesh_order, nelements_side=resolution)
