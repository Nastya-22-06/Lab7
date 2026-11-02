"""Программная растеризация треугольников: RGBA + Z-буфер NumPy.

Это основа для лабораторных №8–9: сюда можно добавить интерполяцию
вершинных цветов (Гуро), UV и нормалей для пиксельного освещения.
"""
from __future__ import annotations

import numpy as np


def clip_polygon(vertices: np.ndarray) -> np.ndarray:
    """Отсечение в clip space по шести плоскостям однородного куба."""
    polygon = np.asarray(vertices, dtype=float)
    planes = (
        np.array([1., 0., 0., 1.]), np.array([-1., 0., 0., 1.]),
        np.array([0., 1., 0., 1.]), np.array([0., -1., 0., 1.]),
        np.array([0., 0., 1., 1.]), np.array([0., 0., -1., 1.]),
    )
    for plane in planes:
        if len(polygon) == 0:
            break
        output = []
        previous = polygon[-1]
        prev_value = float(previous @ plane)
        for current in polygon:
            value = float(current @ plane)
            if (value >= 0) != (prev_value >= 0):
                t = prev_value / (prev_value - value)
                output.append(previous + t * (current - previous))
            if value >= 0:
                output.append(current)
            previous, prev_value = current, value
        polygon = np.array(output, dtype=float).reshape(-1, 4)
    return polygon


def clip_segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return clip_polygon(np.array([a, b]))


def to_screen(clipped: np.ndarray, width: int, height: int) -> np.ndarray:
    ndc = clipped[:, :3] / clipped[:, 3, None]
    return np.column_stack(((ndc[:, 0] + 1.) * width / 2.,
                            (1. - ndc[:, 1]) * height / 2., ndc[:, 2]))


def draw_triangle(p: np.ndarray, color: tuple[int, int, int],
                  pixels: np.ndarray, zbuffer: np.ndarray) -> None:
    """Барицентрическая растеризация + сравнение глубин в каждом пикселе."""
    height, width = zbuffer.shape
    left = max(0, int(np.floor(np.min(p[:, 0]))))
    right = min(width - 1, int(np.ceil(np.max(p[:, 0]))))
    top = max(0, int(np.floor(np.min(p[:, 1]))))
    bottom = min(height - 1, int(np.ceil(np.max(p[:, 1]))))
    if left > right or top > bottom:
        return
    x0, y0 = p[0, :2]
    x1, y1 = p[1, :2]
    x2, y2 = p[2, :2]
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if abs(area) < 1e-9:
        return
    ys, xs = np.ogrid[top:bottom + 1, left:right + 1]
    xs, ys = xs + 0.5, ys + 0.5
    b0 = ((x1 - xs) * (y2 - ys) - (y1 - ys) * (x2 - xs)) / area
    b1 = ((x2 - xs) * (y0 - ys) - (y2 - ys) * (x0 - xs)) / area
    b2 = 1. - b0 - b1
    inside = (b0 >= -1e-7) & (b1 >= -1e-7) & (b2 >= -1e-7)
    if not np.any(inside):
        return
    depth = b0 * p[0, 2] + b1 * p[1, 2] + b2 * p[2, 2]
    region_depth = zbuffer[top:bottom + 1, left:right + 1]
    drawn = inside & (depth < region_depth)
    if np.any(drawn):
        region_depth[drawn] = depth[drawn]
        region_pixels = pixels[top:bottom + 1, left:right + 1]
        region_pixels[drawn] = (*color, 255)


def draw_depth_line(p0: np.ndarray, p1: np.ndarray,
                    pixels: np.ndarray, zbuffer: np.ndarray,
                    color: tuple[int, int, int] = (204, 225, 250),
                    test_depth: bool = True) -> None:
    height, width = zbuffer.shape
    length = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    if length > 10000:
        return
    t = np.linspace(0., 1., max(2, length))
    points = p0[None, :] * (1. - t[:, None]) + p1[None, :] * t[:, None]
    x = np.rint(points[:, 0]).astype(int)
    y = np.rint(points[:, 1]).astype(int)
    visible = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(visible):
        return
    x, y, depth = x[visible], y[visible], points[visible, 2]
    mask = (depth <= zbuffer[y, x] + 0.008) if test_depth else np.ones(len(x), dtype=bool)
    x, y = x[mask], y[mask]
    pixels[y, x] = (*color, 255)
