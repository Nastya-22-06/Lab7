"""Wavefront OBJ: позиции v, UV vt, нормали vn и индексные углы граней."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .geometry import Face, Point3D, Polyhedron, VertexRef


def _index(text: str, size: int, line: int) -> int:
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"Строка {line}: неверный индекс {text!r}") from exc
    # В OBJ положительные индексы идут с 1, отрицательные — от конца списка.
    index = number - 1 if number > 0 else size + number
    if not 0 <= index < size:
        raise ValueError(f"Строка {line}: индекс {number} вне диапазона 1..{size}")
    return index


def load_obj(path: str | Path) -> Polyhedron:
    path = Path(path)
    vertices: list[Point3D] = []
    texcoords: list[tuple[float, float]] = []
    normals: list[Point3D] = []
    faces: list[Face] = []
    object_name = path.stem
    raw_lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    # OBJ допускает перенос строки через обратную косую черту.
    lines: list[tuple[int, str]] = []
    pending = ""
    first_line = 1
    for line_number, line in enumerate(raw_lines, 1):
        segment = line.split("#", 1)[0].strip()
        if not pending:
            first_line = line_number
        if segment.endswith("\\"):
            pending += segment[:-1] + " "
            continue
        lines.append((first_line, pending + segment))
        pending = ""
    if pending.strip():
        raise ValueError("Файл OBJ заканчивается незавершённой строкой")

    for line_number, line in lines:
        if not line:
            continue
        parts = line.split()
        kind, values = parts[0], parts[1:]
        try:
            if kind == "v":
                if len(values) < 3:
                    raise ValueError("v требует не менее трёх координат")
                xyz = np.array([float(v) for v in values[:3]], dtype=float)
                if len(values) > 3:
                    w = float(values[3])
                    if abs(w) < 1e-12:
                        raise ValueError("однородная координата w не должна быть нулевой")
                    xyz /= w
                if not np.all(np.isfinite(xyz)):
                    raise ValueError("координаты должны быть конечными")
                vertices.append(Point3D(*xyz))
            elif kind == "vt":
                if not values:
                    raise ValueError("vt требует координату u")
                u = float(values[0])
                v = float(values[1]) if len(values) > 1 else 0.0
                if not np.isfinite(u) or not np.isfinite(v):
                    raise ValueError("неверные координаты текстуры")
                texcoords.append((u, v))
            elif kind == "vn":
                if len(values) < 3:
                    raise ValueError("vn требует три координаты")
                xyz = np.array([float(v) for v in values[:3]], dtype=float)
                if not np.all(np.isfinite(xyz)):
                    raise ValueError("неверные координаты нормали")
                normals.append(Point3D(*xyz))
            elif kind == "f":
                if len(values) < 3:
                    raise ValueError("f должна иметь хотя бы три угла")
                corners: list[VertexRef] = []
                for token in values:
                    fields = token.split("/")
                    if len(fields) > 3 or not fields[0]:
                        raise ValueError(f"неверный формат угла грани: {token}")
                    v = _index(fields[0], len(vertices), line_number)
                    vt = _index(fields[1], len(texcoords), line_number) if len(fields) > 1 and fields[1] else None
                    vn = _index(fields[2], len(normals), line_number) if len(fields) > 2 and fields[2] else None
                    corners.append(VertexRef(v, vt, vn))
                faces.append(Face(tuple(c.position for c in corners), tuple(corners)))
            elif kind == "o" and values and not faces:
                object_name = " ".join(values)[:100]
            # mtllib/usemtl, s, g, l, vp и прочее не влияют на геометрию ЛР №7.
        except ValueError as exc:
            if str(exc).startswith("Строка "):
                raise
            raise ValueError(f"Строка {line_number}: {exc}") from exc

    if not vertices or not faces:
        raise ValueError("В OBJ должны быть координаты v и грани f")
    return Polyhedron(object_name, vertices, faces,
                      texcoords=texcoords, normals=normals)


def save_obj(polyhedron: Polyhedron, path: str | Path) -> None:
    """Сохраняет мировые координаты и корректные f v/vt/vn (включая швы)."""
    path = Path(path)
    vertices = polyhedron.world_vertices
    flip = np.linalg.det(polyhedron.model_matrix[:3, :3]) < 0
    normal_matrix = np.linalg.pinv(polyhedron.model_matrix[:3, :3]).T
    world_normals = []
    for normal in polyhedron.normals:
        vector = normal_matrix @ normal.as_array()
        length = np.linalg.norm(vector)
        world_normals.append(vector / length if length > 1e-12 else vector)
    # Если в OBJ не были заданы все нормали, достраиваем их из смежных граней.
    missing = any(corner.normal is None for face in polyhedron.faces for corner in face.corners)
    generated_start = len(world_normals)
    if missing:
        for xyz in polyhedron.compute_vertex_normals():
            vector = normal_matrix @ xyz
            length = np.linalg.norm(vector)
            world_normals.append(vector / length if length > 1e-12 else vector)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# Generated by Lab 7 — geometry / Wavefront OBJ\n")
        output.write(f"o {polyhedron.name.replace(chr(10), ' ')}\n")
        for xyz in vertices:
            output.write("v " + " ".join(f"{v:.10g}" for v in xyz) + "\n")
        for u, v in polyhedron.texcoords:
            output.write(f"vt {u:.10g} {v:.10g}\n")
        for xyz in world_normals:
            output.write("vn " + " ".join(f"{v:.10g}" for v in xyz) + "\n")
        for face in polyhedron.faces:
            corners = reversed(face.corners) if flip else face.corners
            tokens = []
            for corner in corners:
                pos = str(corner.position + 1)
                vn = corner.normal if corner.normal is not None else generated_start + corner.position
                uv = str(corner.uv + 1) if corner.uv is not None else ""
                tokens.append(f"{pos}/{uv}/{vn + 1}")
            output.write("f " + " ".join(tokens) + "\n")
