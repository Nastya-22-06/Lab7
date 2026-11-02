"""Точки, грани, многогранники и генерация пяти Платоновых тел."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
import math

import numpy as np

from .transforms import identity, transform_points


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class VertexRef:
    """Индексы v/vt/vn одной вершины полигона в формате OBJ (с нуля)."""
    position: int
    uv: int | None = None
    normal: int | None = None


@dataclass(frozen=True)
class Face:
    indices: tuple[int, ...]
    corners: tuple[VertexRef, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.indices) < 3 or len(set(self.indices)) < 3:
            raise ValueError("Грань должна содержать не менее трёх разных вершин")
        if self.corners is None:
            object.__setattr__(self, "corners", tuple(VertexRef(i) for i in self.indices))
        elif len(self.corners) != len(self.indices) or any(
            corner.position != i for corner, i in zip(self.corners, self.indices)
        ):
            raise ValueError("Индексы v в углах должны совпадать с вершинами грани")

    def normal(self, vertices: np.ndarray) -> np.ndarray:
        # По первым трём НЕколлинеарным вершинам (полигон может быть n-угольником).
        p = vertices[self.indices[0]]
        for i in range(1, len(self.indices) - 1):
            q, r = vertices[[self.indices[i], self.indices[i + 1]]]
            normal = np.cross(q - p, r - p)
            length = np.linalg.norm(normal)
            if length > 1e-12:
                return normal / length
        return np.zeros(3)


@dataclass
class Polyhedron:
    name: str
    vertices: list[Point3D]
    faces: list[Face]
    model_matrix: np.ndarray = field(default_factory=identity)
    texcoords: list[tuple[float, float]] = field(default_factory=list)
    normals: list[Point3D] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.vertices:
            raise ValueError("Модель не может быть пустой")
        if not np.all(np.isfinite(self.local_vertices)):
            raise ValueError("Координаты модели должны быть конечными")
        for face in self.faces:
            for corner in face.corners:
                if not 0 <= corner.position < len(self.vertices):
                    raise ValueError("Индекс вершины грани находится вне диапазона")
                if corner.uv is not None and not 0 <= corner.uv < len(self.texcoords):
                    raise ValueError("Индекс UV находится вне диапазона")
                if corner.normal is not None and not 0 <= corner.normal < len(self.normals):
                    raise ValueError("Индекс нормали находится вне диапазона")

    @property
    def local_vertices(self) -> np.ndarray:
        return np.array([v.as_array() for v in self.vertices], dtype=float)

    @property
    def world_vertices(self) -> np.ndarray:
        return transform_points(self.local_vertices, self.model_matrix)

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.world_vertices, axis=0)

    @property
    def edges(self) -> list[tuple[int, int]]:
        result: set[tuple[int, int]] = set()
        for face in self.faces:
            ring = face.indices
            for a, b in zip(ring, ring[1:] + ring[:1]):
                result.add(tuple(sorted((a, b))))
        return sorted(result)

    def compute_vertex_normals(self, assign_missing: bool = False) -> np.ndarray:
        """Среднее нормалей смежных граней; импортированные vn не теряем."""
        positions = self.local_vertices
        sums = np.zeros((len(positions), 3), dtype=float)
        for face in self.faces:
            normal = face.normal(positions)
            for index in set(face.indices):
                sums[index] += normal
        lengths = np.linalg.norm(sums, axis=1)
        good = lengths > 1e-12
        sums[good] /= lengths[good, None]
        if assign_missing and any(corner.normal is None for face in self.faces for corner in face.corners):
            start = len(self.normals)
            self.normals.extend(Point3D(*n) for n in sums)
            self.faces = [Face(face.indices, tuple(
                corner if corner.normal is not None
                else VertexRef(corner.position, corner.uv, start + corner.position)
                for corner in face.corners
            )) for face in self.faces]
        return sums

    def apply(self, matrix: np.ndarray) -> None:
        """Матрицы-столбцы: M_new = T @ M_old."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Преобразование должно быть конечной матрицей 4×4")
        self.model_matrix = matrix @ self.model_matrix

    def reset(self) -> None:
        self.model_matrix = identity()


def _mesh(name: str, raw_vertices: list[tuple[float, float, float]] | np.ndarray,
          raw_faces: list[tuple[int, ...]] | list[list[int]]) -> Polyhedron:
    points = np.asarray(raw_vertices, dtype=float)
    points -= points.mean(axis=0)
    radius = np.linalg.norm(points, axis=1).max()
    points *= 1.65 / radius
    faces = []
    for face in raw_faces:
        indices = tuple(face)
        p, q, r = points[list(indices[:3])]
        normal = np.cross(q - p, r - p)
        face_center = np.mean(points[list(indices)], axis=0)
        if np.dot(normal, face_center) < 0:
            indices = tuple(reversed(indices))
        faces.append(Face(indices))
    return Polyhedron(name, [Point3D(*map(float, xyz)) for xyz in points], faces)


def tetrahedron() -> Polyhedron:
    vertices = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return _mesh("Тетраэдр", vertices, faces)


def hexahedron() -> Polyhedron:
    vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    return _mesh("Гексаэдр (куб)", vertices, faces)


def octahedron() -> Polyhedron:
    vertices = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return _mesh("Октаэдр", vertices, faces)


def _icosahedron_raw() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    phi = (1.0 + math.sqrt(5.0)) / 2
    vertices = np.array(
        [(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)] +
        [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)] +
        [(a * phi, 0, b) for a in (-1, 1) for b in (-1, 1)], dtype=float)
    distances = [np.linalg.norm(vertices[i] - vertices[j]) for i, j in combinations(range(12), 2)]
    side = min(distances)
    near = np.isclose
    faces = [triangle for triangle in combinations(range(12), 3)
             if all(near(np.linalg.norm(vertices[i] - vertices[j]), side, atol=1e-8)
                    for i, j in combinations(triangle, 2))]
    return vertices, faces


def icosahedron() -> Polyhedron:
    vertices, faces = _icosahedron_raw()
    return _mesh("Икосаэдр ★", vertices, faces)


def dodecahedron() -> Polyhedron:
    # Двойственный икосаэдру: 20 центров треугольных граней → 20 вершин.
    source_vertices, triangles = _icosahedron_raw()
    dual_vertices = np.array([source_vertices[list(triangle)].mean(axis=0) for triangle in triangles])
    faces = []
    for vertex_index, axis in enumerate(source_vertices):
        neighbor_faces = [i for i, triangle in enumerate(triangles) if vertex_index in triangle]
        direction = axis / np.linalg.norm(axis)
        auxiliary = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(auxiliary, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        # Сортируем центры соседних треугольников циклически вокруг вершины.
        ordered = sorted(neighbor_faces, key=lambda i: math.atan2(np.dot(dual_vertices[i], up), np.dot(dual_vertices[i], right)))
        faces.append(tuple(ordered))
    return _mesh("Додекаэдр ★", dual_vertices, faces)


SOLIDS = {
    "Тетраэдр": tetrahedron,
    "Гексаэдр (куб)": hexahedron,
    "Октаэдр": octahedron,
    "Икосаэдр ★": icosahedron,
    "Додекаэдр ★": dodecahedron,
}


# Экспорт/импорт OBJ находится в obj_io.py. Сохраняем прежний путь импорта.
def save_obj(polyhedron: Polyhedron, path: str | Path) -> None:
    from .obj_io import save_obj as _save_obj
    _save_obj(polyhedron, path)
