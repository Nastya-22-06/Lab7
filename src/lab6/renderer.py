"""3D окно: модель -> вид -> проекция -> пиксели через собственный Z-буфер."""
from __future__ import annotations

import math
import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .geometry import Polyhedron
from .projection import look_at, orthographic, perspective, project
from .rasterizer import clip_polygon, clip_segment, to_screen, draw_triangle, draw_depth_line


class Viewport(QWidget):
    camera_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.polyhedron: Polyhedron | None = None
        self.mode = "perspective"
        self.show_faces = True
        self.show_edges = True
        self.show_axes = True
        self.yaw = 45.0
        self.pitch = 25.0
        self.distance = 7.5
        self.target = np.zeros(3)
        self.radius = 1.65
        self.last_mouse_pos = None
        self.setMinimumSize(490, 440)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_polyhedron(self, model: Polyhedron) -> None:
        self.polyhedron = model
        self.fit_camera()

    def fit_camera(self) -> None:
        if self.polyhedron is not None:
            vertices = self.polyhedron.world_vertices
            self.target = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            self.radius = max(0.1, float(np.linalg.norm(vertices - self.target, axis=1).max()))
            self.distance = max(2.8, self.radius * 3.5)
        self.yaw, self.pitch = 45., 25.
        self.update()
        self.camera_changed.emit()

    def reset_camera(self) -> None:
        self.fit_camera()

    def _camera(self):
        a, b = math.radians(self.yaw), math.radians(self.pitch)
        position = np.array([math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)])
        eye = self.target + self.distance * position
        view = look_at(eye, self.target)
        aspect = max(1., self.width()) / max(1., self.height())
        far = max(100., self.distance + 4. * self.radius + 10.)
        projection = (perspective(46., aspect, near=0.05, far=far) if self.mode == "perspective"
                      else orthographic(self.distance * .43, aspect, near=0.05, far=far))
        return view, projection

    def _line3d(self, painter, a, b, view, projection, color, width=1.):
        pts, valid = project(np.array([a, b], dtype=float), view, projection, self.width(), self.height())
        if not np.all(valid):
            return None
        first, last = QPointF(*pts[0]), QPointF(*pts[1])
        pen = QPen(QColor(color))
        pen.setWidthF(width)
        painter.setPen(pen)
        painter.drawLine(first, last)
        return last

    def _draw_background(self, painter, view, projection):
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0., QColor("#101b30"))
        bg.setColorAt(1., QColor("#080f1d"))
        painter.fillRect(self.rect(), QBrush(bg))
        if self.show_axes:
            # Координатные плоскости/оси оставляем в QPainter: не растеризация модели.
            extent = max(4., min(20., self.radius * 2.5))
            step = extent / 8
            base = -self.radius * .9
            for k in range(-8, 9):
                coordinate = k * step
                self._line3d(painter, [coordinate, base, -extent],
                             [coordinate, base, extent], view, projection, "#243651", .7)
                self._line3d(painter, [-extent, base, coordinate],
                             [extent, base, coordinate], view, projection, "#243651", .7)
            for tip, color, label in [([2.5, 0, 0], "#ee879a", "X"),
                                       ([0, 2.5, 0], "#65deb4", "Y"),
                                       ([0, 0, 2.5], "#86bfff", "Z")]:
                origin = np.zeros(3)
                endpoint = np.array(tip, dtype=float) * max(1., self.radius / 1.65)
                place = self._line3d(painter, origin, endpoint, view, projection, color, 1.9)
                if place:
                    painter.setPen(QColor(color))
                    painter.drawText(place + QPointF(4., -4.), label)

    def _render_mesh(self, view, projection, width: int, height: int) -> QImage | None:
        model = self.polyhedron
        if model is None:
            return None
        world = model.world_vertices
        transform = projection @ view
        homogeneous = np.column_stack((world, np.ones(len(world))))
        clipped_points = (transform @ homogeneous.T).T
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        zbuffer = np.full((height, width), np.inf, dtype=np.float32)
        light = np.array([-0.35, 0.85, 0.46])
        light /= np.linalg.norm(light)

        if self.show_faces:
            for index, face in enumerate(model.faces):
                face_clip = clip_polygon(clipped_points[list(face.indices)])
                if len(face_clip) < 3:
                    continue
                polygon = to_screen(face_clip, width, height)
                normal = face.normal(world)
                brightness = .40 + .60 * abs(float(np.dot(normal, light)))
                # Не Ламберт/Гуро: один оттенок на всю грань; освещение — в ЛР №9.
                base = np.array([63, 148, 222]) if index % 3 else np.array([82, 172, 224])
                rgb = tuple(np.clip(base * brightness + 15, 0, 255).astype(np.uint8))
                for i in range(1, len(polygon) - 1):
                    triangle = polygon[[0, i, i + 1]]
                    draw_triangle(triangle, rgb, rgba, zbuffer)

        if self.show_edges:
            for a, b in model.edges:
                segment = clip_segment(clipped_points[a], clipped_points[b])
                if len(segment) < 2:
                    continue
                screen = to_screen(segment[[0, -1]], width, height)
                draw_depth_line(screen[0], screen[1], rgba, zbuffer,
                                test_depth=self.show_faces)

        # QImage.copy() отделяет буфер Qt от временного numpy-массива.
        image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888)
        return image.copy()

    def paintEvent(self, event):  # Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        view, projection = self._camera()
        self._draw_background(painter, view, projection)
        if self.polyhedron is not None:
            # Ограничиваем число программно обрабатываемых пикселей.
            ratio = min(1., 780. / max(1, self.width()), 570. / max(1, self.height()))
            w = max(1, int(self.width() * ratio))
            h = max(1, int(self.height() * ratio))
            image = self._render_mesh(view, projection, w, h)
            if image is not None:
                painter.drawImage(self.rect(), image)
        painter.setPen(QColor("#bfcee6"))
        painter.setFont(QFont("Arial", 10))
        name = "ПЕРСПЕКТИВА" if self.mode == "perspective" else "АКСОНОМЕТРИЯ"
        painter.drawText(16, 27, name + " · программная растеризация")
        painter.setPen(QColor("#899cba"))
        painter.drawText(16, self.height() - 15, "ЛКМ — вращение камеры · колесо — масштаб · кнопка «Вписать»")
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self.last_mouse_pos
            self.yaw += delta.x() * .45
            self.pitch = max(-86., min(86., self.pitch + delta.y() * .45))
            self.last_mouse_pos = event.position()
            self.update()
            self.camera_changed.emit()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event):
        self.distance = np.clip(self.distance * (.86 if event.angleDelta().y() > 0 else 1.16),
                                max(.3, self.radius * .2), max(25., self.radius * 50.))
        self.update()
        self.camera_changed.emit()
