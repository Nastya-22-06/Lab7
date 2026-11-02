"""Фигура вращения и дискретная поверхность z = f(x, y)."""
from __future__ import annotations

import ast
import math

import numpy as np

from .geometry import Face, Point3D, Polyhedron, VertexRef
from .transforms import rotation_axis, transform_points


PROFILE_DEFAULT = """0.0 -1.4 0.0
0.65 -1.4 0.0
0.80 -1.1 0.0
0.55 -0.3 0.0
0.35 0.15 0.0
0.55 0.9 0.0
0.55 1.25 0.0
0.0 1.25 0.0"""

FUNCTIONS = {
    "Волны: sin(x)*cos(y)": "sin(x)*cos(y)",
    "Радиальная волна": "sin(sqrt(x*x+y*y))",
    "Седло": "0.15*(x*x-y*y)",
    "Параболоид": "0.12*(x*x+y*y)",
    "Своя функция": "sin(x)*cos(y)",
}


def read_profile(text: str) -> list[Point3D]:
    points = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            values = [float(v) for v in line.replace(";", " ").replace(",", " ").split()]
        except ValueError as exc:
            raise ValueError(f"Образующая: строка {number}, ожидаются три числа X Y Z") from exc
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Образующая: строка {number}, ожидаются три конечных числа X Y Z")
        points.append(Point3D(*values))
    if not 2 <= len(points) <= 250:
        raise ValueError("Образующая должна содержать от 2 до 250 точек")
    if all(np.linalg.norm(np.subtract(p.as_array(), points[0].as_array())) < 1e-10 for p in points):
        raise ValueError("Все точки образующей совпадают")
    return points


def revolution(profile: list[Point3D], axis: str, steps: int) -> Polyhedron:
    """Поворачиваем образующую на 360°; точки на оси не дублируем."""
    axis = axis.upper()
    if axis not in ("X", "Y", "Z"):
        raise ValueError("Ось должна быть X, Y или Z")
    if not 3 <= steps <= 256:
        raise ValueError("Количество разбиений должно быть от 3 до 256")
    if len(profile) < 2:
        raise ValueError("Нужно хотя бы две точки образующей")
    base = np.array([p.as_array() for p in profile])
    axis_id = "XYZ".index(axis)
    other = [i for i in range(3) if i != axis_id]
    rings: list[list[int]] = []
    points: list[Point3D] = []
    for step in range(steps):
        angle = 360.0 * step / steps
        rotated = transform_points(base, rotation_axis(axis, angle))
        ring: list[int] = []
        for k, pos in enumerate(rotated):
            if np.linalg.norm(base[k, other]) < 1e-9 and step > 0:
                ring.append(rings[0][k])
            else:
                ring.append(len(points))
                points.append(Point3D(*pos))
        rings.append(ring)

    # UV на шве повторяет координату u=1, не дублируя 3D-вершину.
    uv: list[tuple[float, float]] = []
    for j in range(steps + 1):
        for k in range(len(profile)):
            uv.append((j / steps, k / (len(profile) - 1)))

    faces: list[Face] = []
    for j in range(steps):
        nj = (j + 1) % steps
        for k in range(len(profile) - 1):
            p00, p10 = rings[j][k], rings[nj][k]
            p11, p01 = rings[nj][k + 1], rings[j][k + 1]
            raw = [(p00, j * len(profile) + k),
                   (p10, (j + 1) * len(profile) + k),
                   (p11, (j + 1) * len(profile) + k + 1),
                   (p01, j * len(profile) + k + 1)]
            # На полюсах получаются треугольники, а не вырожденные четырёхугольники.
            raw = [entry for i, entry in enumerate(raw) if i == 0 or entry[0] != raw[i - 1][0]]
            if len(raw) > 2 and raw[-1][0] == raw[0][0]:
                raw.pop()
            if len(set(i for i, _ in raw)) < 3:
                continue
            indices = tuple(i for i, _ in raw)
            coords = np.array([points[i].as_array() for i in indices])
            normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
            radial = coords.mean(axis=0)
            radial[axis_id] = 0.0
            if np.dot(normal, radial) < 0:
                raw.reverse()
            refs = tuple(VertexRef(i, tex) for i, tex in raw)
            faces.append(Face(tuple(c.position for c in refs), refs))
    if not faces:
        raise ValueError("Образующая лежит на оси: невозможно получить поверхность")
    return Polyhedron(f"Фигура вращения ({axis}, {steps})", points, faces, texcoords=uv)


# AST вместо произвольного eval кода пользователя.
_ALLOWED_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan,
                  "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
                  "log": np.log}
_ALLOWED_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)
_ALLOWED_NODE_TYPES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
                       ast.Name, ast.Load, ast.Constant) + _ALLOWED_OPS


def _formula(expr: str):
    if len(expr) > 180:
        raise ValueError("Формула слишком длинная")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Синтаксическая ошибка в формуле") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError("В формуле разрешены числа, x, y и математические функции")
        if isinstance(node, ast.Name) and node.id not in {"x", "y", "pi", *(_ALLOWED_FUNCS.keys())}:
            raise ValueError(f"Неизвестное имя в формуле: {node.id}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                                           or node.func.id not in _ALLOWED_FUNCS
                                           or node.keywords):
            raise ValueError("Допустимы только вызовы sin, cos, tan, sqrt, abs, exp, log")
        if isinstance(node, ast.Constant) and (type(node.value) not in (float, int)
                                                or abs(node.value) > 1e7):
            raise ValueError("Неверная числовая константа")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            # Не даём вычислять вложенные степени с гигантскими показателями.
            exponent = node.right
            if not isinstance(exponent, ast.Constant) or type(exponent.value) not in (int, float) or abs(exponent.value) > 10:
                raise ValueError("Показатель степени должен быть числом не больше 10 по модулю")
    code = compile(tree, "<surface>", "eval")

    def evaluate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        scope = {**_ALLOWED_FUNCS, "x": x, "y": y, "pi": math.pi}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = eval(code, {"__builtins__": {}}, scope)
        z = np.broadcast_to(np.asarray(result, dtype=float), x.shape)
        if not np.all(np.isfinite(z)):
            raise ValueError("Функция не определена на выбранном диапазоне")
        if np.max(np.abs(z)) > 1e5:
            raise ValueError("Значения функции слишком велики")
        return z
    return evaluate


def function_surface(expr: str, x0: float, x1: float, y0: float, y1: float,
                     nx: int, ny: int) -> Polyhedron:
    if not (math.isfinite(x0) and math.isfinite(x1) and math.isfinite(y0) and math.isfinite(y1)):
        raise ValueError("Диапазоны должны быть конечными")
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Должно выполняться x0 < x1 и y0 < y1")
    if not (2 <= nx <= 180 and 2 <= ny <= 180):
        raise ValueError("Разбиения по каждой оси: 2..180")
    x, y = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))
    z = _formula(expr)(x, y)
    vertices = [Point3D(float(a), float(b), float(c)) for a, b, c in zip(x.flat, y.flat, z.flat)]
    texcoords = [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    faces: list[Face] = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            ids = (a, a + 1, a + nx + 2, a + nx + 1)  # CCW -> нормаль +Z
            refs = tuple(VertexRef(index, index) for index in ids)
            faces.append(Face(ids, refs))
    return Polyhedron(f"z = {expr}", vertices, faces, texcoords=texcoords)
