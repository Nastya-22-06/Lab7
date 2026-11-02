"""Видовая/проекционная матрицы и переход NDC → пиксели."""
from __future__ import annotations

import math
import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    world_up = np.array([0., 1., 0.]) if up is None else np.asarray(up, float)
    forward = target - eye
    if np.linalg.norm(forward) < 1e-12:
        raise ValueError("Камера должна отличаться от цели")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-10:
        raise ValueError("Направление взгляда параллельно up")
    right /= np.linalg.norm(right)
    up_vector = np.cross(right, forward)
    result = np.eye(4)
    result[:3, :3] = [right, up_vector, -forward]
    result[:3, 3] = -result[:3, :3] @ eye
    return result


def perspective(fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    if not (0 < fov_deg < 180 and aspect > 0 and 0 < near < far):
        raise ValueError("Некорректные параметры перспективной проекции")
    f = 1 / math.tan(math.radians(fov_deg) / 2)
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0], matrix[1, 1] = f / aspect, f
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -(2 * far * near) / (far - near)
    matrix[3, 2] = -1
    return matrix


def orthographic(half_height: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    if half_height <= 0 or aspect <= 0 or not (0 < near < far):
        raise ValueError("Некорректные параметры аксонометрической проекции")
    half_width = half_height * aspect
    matrix = np.eye(4)
    matrix[0, 0] = 1 / half_width
    matrix[1, 1] = 1 / half_height
    matrix[2, 2] = -2 / (far - near)
    matrix[2, 3] = -(far + near) / (far - near)
    return matrix


def project(points: np.ndarray, view: np.ndarray, projection: np.ndarray,
            width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает экранные N×2 и признак допустимого перспективного деления."""
    pts = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    clip = (projection @ view @ homogeneous.T).T
    valid = clip[:, 3] > 1e-8
    ndc = np.full((len(pts), 3), np.nan)
    ndc[valid] = clip[valid, :3] / clip[valid, 3, None]
    screen = np.column_stack(((ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2))
    return screen, valid & np.all(np.isfinite(ndc), axis=1)
