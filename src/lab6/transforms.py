"""Однородные матрицы 4×4. Используются столбцы [x, y, z, 1]."""
from __future__ import annotations

import math
import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def translation(dx: float, dy: float, dz: float) -> np.ndarray:
    matrix = identity()
    matrix[:3, 3] = (dx, dy, dz)
    return matrix


def scaling(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.diag([sx, sy, sz, 1.0]).astype(float)


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    matrix = identity()
    matrix[1:3, 1:3] = [[c, -s], [s, c]]
    return matrix


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    matrix = identity()
    matrix[0, 0], matrix[0, 2] = c, s
    matrix[2, 0], matrix[2, 2] = -s, c
    return matrix


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    matrix = identity()
    matrix[:2, :2] = [[c, -s], [s, c]]
    return matrix


ROTATIONS = {"X": rotation_x, "Y": rotation_y, "Z": rotation_z}


def rotation_axis(axis: str, angle: float) -> np.ndarray:
    if axis.upper() not in ROTATIONS:
        raise ValueError("Ось должна быть X, Y или Z")
    return ROTATIONS[axis.upper()](angle)


def reflection(plane: str) -> np.ndarray:
    """Плоскости XY, XZ, YZ проходят через начало координат."""
    signs = {"XY": (1, 1, -1), "XZ": (1, -1, 1), "YZ": (-1, 1, 1)}
    if plane.upper() not in signs:
        raise ValueError("Плоскость должна быть XY, XZ или YZ")
    return scaling(*signs[plane.upper()])


def around_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """T(C) @ M @ T(-C)."""
    x, y, z = np.asarray(point, dtype=float)
    return translation(x, y, z) @ matrix @ translation(-x, -y, -z)


def rotate_around_line(a: np.ndarray, b: np.ndarray, angle: float) -> np.ndarray:
    """Поворот вокруг AB: перенос, совмещение с OZ, поворот, обратные шаги."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("A и B должны быть конечными трёхмерными точками")
    direction = b - a
    length = np.linalg.norm(direction)
    if length < 1e-10:
        raise ValueError("Точки A и B должны отличаться")
    l, m, n = direction / length
    # После Rx(psi), Ry(-theta) единичный вектор AB совпадёт с OZ.
    psi = math.degrees(math.atan2(m, n))
    theta = math.degrees(math.atan2(l, math.hypot(m, n)))
    align = rotation_y(-theta) @ rotation_x(psi)
    return translation(*a) @ align.T @ rotation_z(angle) @ align @ translation(*(-a))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Преобразование массива N×3 столбцами однородных координат."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Ожидается массив N×3")
    if np.asarray(matrix).shape != (4, 4):
        raise ValueError("Ожидается матрица 4×4")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    result = (np.asarray(matrix) @ homogeneous.T).T
    w = result[:, 3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("Невозможно выполнить перспективное деление: w = 0")
    return result[:, :3] / w[:, None]
