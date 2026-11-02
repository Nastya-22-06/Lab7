"""Интерфейс лабораторных №6–7: OBJ, поверхности и матричные действия."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QPlainTextEdit, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .geometry import SOLIDS, Polyhedron
from .obj_io import load_obj, save_obj
from .generators import PROFILE_DEFAULT, FUNCTIONS, read_profile, revolution, function_surface
from .renderer import Viewport
from .transforms import (
    around_point, reflection, rotate_around_line, rotation_axis, scaling, translation,
)


STYLE = """
QMainWindow, QWidget { background: #0c1525; color: #e8f0ff; font-family: 'Segoe UI', Arial; font-size: 12px; }
QGroupBox { border: 1px solid #263b57; border-radius: 10px; margin-top: 15px; padding: 13px 10px 10px; font-weight: 600; background: #111e31; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #c7d9fa; }
QDoubleSpinBox, QSpinBox, QLineEdit, QPlainTextEdit, QComboBox { border: 1px solid #324a6b; border-radius: 6px; background: #172941; padding: 5px; min-height: 24px; selection-background-color: #317be2; }
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border-color: #6aa9ff; }
QComboBox QAbstractItemView { background: #16263c; selection-background-color: #355f94; }
QPushButton { background: #235baf; border: 1px solid #3e7acb; border-radius: 7px; color: white; min-height: 29px; padding: 4px 10px; font-weight: 600; }
QPushButton:hover { background: #3374ce; }
QPushButton:pressed { background: #1c4585; }
QPushButton[secondary="true"] { background: #203047; border: 1px solid #38516f; }
QPushButton[secondary="true"]:hover { background: #2a4161; }
QCheckBox { spacing: 7px; }
QScrollArea { border: none; background: transparent; }
QLabel[muted="true"] { color: #8da3c3; }
QMenuBar { background: #101c30; color: #e8f0ff; }
QMenuBar::item:selected { background: #264976; }
QMenu { background: #142237; color: #e8f0ff; }
QMenu::item:selected { background: #315985; }
QStatusBar { color: #97acc9; background: #101c30; }
"""


def spin(value: float = 0.0, low: float = -1000.0, high: float = 1000.0,
         step: float = 0.25) -> QDoubleSpinBox:
    field = QDoubleSpinBox()
    field.setRange(low, high)
    field.setDecimals(3)
    field.setSingleStep(step)
    field.setValue(value)
    field.setKeyboardTracking(False)
    return field


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Лабораторная №7 — Построение 3D-моделей")
        self.resize(1360, 840)
        self.setMinimumSize(1040, 660)
        self.setStyleSheet(STYLE)
        self.model: Polyhedron = SOLIDS["Гексаэдр (куб)"]()
        self.history: list[np.ndarray] = []

        container = QWidget()
        self.setCentralWidget(container)
        outer = QHBoxLayout(container)
        outer.setContentsMargins(12, 12, 12, 10)
        outer.setSpacing(12)

        self.viewport = Viewport()
        self.viewport.set_polyhedron(self.model)
        outer.addWidget(self.viewport, stretch=3)

        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setMinimumWidth(350)
        scroller.setMaximumWidth(440)
        controls = QWidget()
        self.panel = QVBoxLayout(controls)
        self.panel.setContentsMargins(8, 4, 8, 8)
        self.panel.setSpacing(10)
        scroller.setWidget(controls)
        outer.addWidget(scroller, stretch=1)

        self._build_panel()
        self._build_menu()
        self._refresh()
        self.statusBar().showMessage("ЛР 7: OBJ, фигуры вращения, поверхности · ЛР 6: матрицы 4×4")

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Файл")
        for name, callback, shortcut in [
            ("Загрузить OBJ...", self._import_obj, "Ctrl+O"),
            ("Экспорт модели в OBJ...", self._export_obj, "Ctrl+E"),
            ("Сохранить изображение PNG...", self._save_picture, "Ctrl+S"),
            ("Выход", self.close, "Ctrl+Q"),
        ]:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            menu.addAction(action)
        edit = self.menuBar().addMenu("Правка")
        undo = QAction("Отменить преобразование", self)
        undo.setShortcut(QKeySequence("Ctrl+Z"))
        undo.triggered.connect(self._undo)
        edit.addAction(undo)
        help_menu = self.menuBar().addMenu("Справка")
        help_action = QAction("Как пользоваться", self)
        help_action.triggered.connect(self._help)
        help_menu.addAction(help_action)

    def _section(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(7)
        self.panel.addWidget(box)
        return box, layout

    @staticmethod
    def _row(label: str, values: list[QWidget]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        if label:
            text = QLabel(label)
            text.setMinimumWidth(36)
            layout.addWidget(text)
        for element in values:
            layout.addWidget(element, 1)
        return widget

    @staticmethod
    def _button(text: str, callback, secondary=False) -> QPushButton:
        button = QPushButton(text)
        if secondary:
            button.setProperty("secondary", True)
        button.clicked.connect(callback)
        return button

    def _build_panel(self) -> None:
        title = QLabel("3D · Лабораторная №7")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.panel.addWidget(title)
        subtitle = QLabel("OBJ · Фигуры вращения · Поверхности · Матрицы")
        subtitle.setProperty("muted", True)
        self.panel.addWidget(subtitle)

        _, layout = self._section("01 · Многогранник и проекция")
        self.solid_combo = QComboBox()
        self.solid_combo.addItems(SOLIDS.keys())
        self.solid_combo.addItem("— OBJ / созданная модель —")
        self.solid_combo.setCurrentText("Гексаэдр (куб)")
        self.solid_combo.currentTextChanged.connect(self._switch_solid)
        layout.addWidget(self.solid_combo)
        self.projection_combo = QComboBox()
        self.projection_combo.addItem("Перспективная", "perspective")
        self.projection_combo.addItem("Аксонометрическая (параллельная)", "orthographic")
        self.projection_combo.currentIndexChanged.connect(self._switch_projection)
        layout.addWidget(self.projection_combo)
        checks = self._row("", [QCheckBox("Грани"), QCheckBox("Рёбра"), QCheckBox("Оси")])
        layout.addWidget(checks)
        self.faces_check, self.edges_check, self.axes_check = checks.findChildren(QCheckBox)
        for checkbox, name in [(self.faces_check, "show_faces"),
                               (self.edges_check, "show_edges"),
                               (self.axes_check, "show_axes")]:
            checkbox.setChecked(True)
            checkbox.toggled.connect(partial(self._toggle_view, name))

        _, layout = self._section("02 · Загрузка / экспорт модели OBJ")
        layout.addWidget(self._button("Открыть OBJ (Ctrl+O)", self._import_obj))
        layout.addWidget(self._button("Сохранить текущую модель OBJ (Ctrl+E)", self._export_obj, secondary=True))
        self.model_name = QLabel()
        self.model_name.setWordWrap(True)
        self.model_name.setProperty("muted", True)
        layout.addWidget(self.model_name)
        layout.addWidget(self._button("Вписать модель в окно", self.viewport.fit_camera, secondary=True))

        _, layout = self._section("03 · Фигура вращения")
        layout.addWidget(QLabel("Образующая: строки X Y Z (по одной точке на строку)"))
        self.profile_edit = QPlainTextEdit(PROFILE_DEFAULT)
        self.profile_edit.setMinimumHeight(135)
        layout.addWidget(self.profile_edit)
        self.revolution_axis = QComboBox()
        self.revolution_axis.addItems(["Y", "X", "Z"])
        self.revolution_steps = QSpinBox()
        self.revolution_steps.setRange(3, 256)
        self.revolution_steps.setValue(24)
        layout.addWidget(self._row("Ось / N", [self.revolution_axis, self.revolution_steps]))
        layout.addWidget(self._button("Подставить образующую для оси", self._profile_example, secondary=True))
        layout.addWidget(self._button("Построить фигуру вращения", self._generate_revolution))

        _, layout = self._section("04 · Поверхность z = f(x, y)")
        self.function_combo = QComboBox()
        self.function_combo.addItems(FUNCTIONS.keys())
        layout.addWidget(self.function_combo)
        self.formula_input = QLineEdit(FUNCTIONS[next(iter(FUNCTIONS))])
        self.formula_input.setPlaceholderText("Например: sin(x)*cos(y)")
        layout.addWidget(self._row("f(x,y)", [self.formula_input]))
        self.function_combo.currentTextChanged.connect(self._choose_function)
        self.xrange_fields = [spin(-3.), spin(3.)]
        self.yrange_fields = [spin(-3.), spin(3.)]
        layout.addWidget(self._row("X min/max", self.xrange_fields))
        layout.addWidget(self._row("Y min/max", self.yrange_fields))
        self.surface_nx, self.surface_ny = QSpinBox(), QSpinBox()
        for count in (self.surface_nx, self.surface_ny):
            count.setRange(2, 100)
            count.setValue(24)
        layout.addWidget(self._row("Nx / Ny", [self.surface_nx, self.surface_ny]))
        layout.addWidget(self._button("Построить поверхность", self._generate_surface))
        layout.addWidget(QLabel("Поддерживаются x, y, pi, sin, cos, tan, sqrt, "
                                "abs, exp, log, + - * / **."))

        _, layout = self._section("05 · Перенос в пространстве")
        self.move_fields = [spin(0) for _ in range(3)]
        layout.addWidget(self._row("Δ XYZ", self.move_fields))
        layout.addWidget(self._button("Переместить", self._move))

        _, layout = self._section("06 · Вращение")
        self.rotation_axis_combo = QComboBox()
        self.rotation_axis_combo.addItems(["X", "Y", "Z"])
        self.rotation_degrees = spin(30, -3600, 3600, 5)
        layout.addWidget(self._row("Ось / °", [self.rotation_axis_combo, self.rotation_degrees]))
        layout.addWidget(self._button("Вокруг оси координат", self._rotate_origin))
        layout.addWidget(self._button("Вокруг оси через центр фигуры", self._rotate_center, secondary=True))

        _, layout = self._section("07 · Масштабирование")
        self.scale_fields = [spin(1, -10, 10, 0.1) for _ in range(3)]
        layout.addWidget(self._row("XYZ", self.scale_fields))
        layout.addWidget(self._button("Относительно начала координат", self._scale_origin))
        layout.addWidget(self._button("Относительно центра фигуры", self._scale_center, secondary=True))
        layout.addWidget(QLabel("0 по оси сводит модель к плоскости — для отмены Ctrl+Z."))

        _, layout = self._section("08 · Отражение")
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "XZ", "YZ"])
        layout.addWidget(self._row("Плоск.", [self.plane_combo]))
        layout.addWidget(self._button("Отразить относительно плоскости", self._reflect))

        _, layout = self._section("09 · Поворот вокруг произвольной прямой")
        layout.addWidget(QLabel("Прямая задаётся точками A и B"))
        self.line_a = [spin(x) for x in (0., 0., 0.)]
        self.line_b = [spin(x) for x in (1., 1., 1.)]
        self.line_angle = spin(45, -3600, 3600, 5)
        layout.addWidget(self._row("A", self.line_a))
        layout.addWidget(self._row("B", self.line_b))
        layout.addWidget(self._row("Угол °", [self.line_angle]))
        layout.addWidget(self._button("Повернуть вокруг AB", self._rotate_line))

        _, layout = self._section("10 · Состояние и управление")
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.matrix_label = QLabel()
        self.matrix_label.setFont(QFont("Consolas", 9))
        self.matrix_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.matrix_label)
        actions = self._row("", [
            self._button("↶ Отменить", self._undo, secondary=True),
            self._button("Сбросить объект", self._reset_model, secondary=True),
        ])
        layout.addWidget(actions)
        layout.addWidget(self._button("Сбросить камеру", self.viewport.reset_camera, secondary=True))
        self.panel.addStretch()

    def _refresh(self) -> None:
        self.viewport.update()
        if hasattr(self, "model_name"):
            self.model_name.setText(f"Текущая модель: {self.model.name}")
        center = self.model.center
        self.info.setText(
            f"Вершин: {len(self.model.vertices)} · Рёбер: {len(self.model.edges)} · "
            f"Граней: {len(self.model.faces)}\n"
            f"Центр: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n"
            "Модельная матрица 4×4:")
        self.matrix_label.setText("\n".join(
            "  ".join(f"{number:7.3f}" for number in row) for row in self.model.model_matrix))

    def _apply(self, matrix: np.ndarray, description: str) -> None:
        try:
            self.history.append(self.model.model_matrix.copy())
            self.model.apply(matrix)
            self._refresh()
            self.statusBar().showMessage(description, 6500)
        except (ValueError, FloatingPointError) as exc:
            if self.history:
                self.history.pop()
            QMessageBox.warning(self, "Невозможно выполнить преобразование", str(exc))

    def _values(self, fields) -> list[float]:
        return [field.value() for field in fields]

    def _move(self) -> None:
        values = self._values(self.move_fields)
        self._apply(translation(*values), f"Перенос: {values}")

    def _rotate_origin(self) -> None:
        axis, angle = self.rotation_axis_combo.currentText(), self.rotation_degrees.value()
        self._apply(rotation_axis(axis, angle), f"Поворот {axis} вокруг начала координат на {angle}°")

    def _rotate_center(self) -> None:
        axis, angle = self.rotation_axis_combo.currentText(), self.rotation_degrees.value()
        self._apply(around_point(rotation_axis(axis, angle), self.model.center),
                    f"Поворот вокруг оси {axis} через центр на {angle}°")

    def _scale_origin(self) -> None:
        values = self._values(self.scale_fields)
        self._apply(scaling(*values), f"Масштабирование относительно O: {values}")

    def _scale_center(self) -> None:
        values = self._values(self.scale_fields)
        self._apply(around_point(scaling(*values), self.model.center),
                    f"Масштабирование относительно центра: {values}")

    def _reflect(self) -> None:
        plane = self.plane_combo.currentText()
        self._apply(reflection(plane), f"Отражение относительно {plane}")

    def _rotate_line(self) -> None:
        try:
            matrix = rotate_around_line(self._values(self.line_a), self._values(self.line_b), self.line_angle.value())
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))
            return
        self._apply(matrix, f"Поворот вокруг AB на {self.line_angle.value()}°")

    def _undo(self) -> None:
        if self.history:
            self.model.model_matrix = self.history.pop()
            self._refresh()
            self.statusBar().showMessage("Последнее преобразование отменено", 4000)
        else:
            self.statusBar().showMessage("История пуста", 3000)

    def _reset_model(self) -> None:
        self.history.append(self.model.model_matrix.copy())
        self.model.reset()
        self._refresh()

    def _replace_model(self, model: Polyhedron, message: str) -> None:
        self.model = model
        self.solid_combo.blockSignals(True)
        self.solid_combo.setCurrentIndex(self.solid_combo.count() - 1)
        self.solid_combo.blockSignals(False)
        self.history.clear()
        self.viewport.set_polyhedron(model)
        self._refresh()
        self.statusBar().showMessage(message, 7000)

    def _import_obj(self) -> None:
        start = Path(__file__).resolve().parents[2] / "examples" / "utah_teapot_lowpoly.obj"
        path, _ = QFileDialog.getOpenFileName(self, "Открыть модель OBJ", str(start), "Wavefront OBJ (*.obj);;Все файлы (*)")
        if not path:
            return
        try:
            model = load_obj(path)
            self._replace_model(model, f"OBJ загружен: {Path(path).name}")
        except (OSError, UnicodeError, ValueError) as exc:
            QMessageBox.warning(self, "Ошибка загрузки OBJ", str(exc))

    def _profile_example(self) -> None:
        axis = self.revolution_axis.currentText()
        coords = read_profile(PROFILE_DEFAULT)
        if axis == "Y":
            self.profile_edit.setPlainText(PROFILE_DEFAULT)
            return
        if axis == "X":
            result = [(p.y, p.x, p.z) for p in coords]
        else:
            result = [(p.x, p.z, p.y) for p in coords]
        self.profile_edit.setPlainText("\n".join(" ".join(str(n) for n in item) for item in result))

    def _generate_revolution(self) -> None:
        try:
            model = revolution(read_profile(self.profile_edit.toPlainText()),
                               self.revolution_axis.currentText(), self.revolution_steps.value())
            self._replace_model(model, "Фигура вращения построена (геометрия + грани + UV)")
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка образующей", str(exc))

    def _choose_function(self, text: str) -> None:
        if text != "Своя функция":
            self.formula_input.setText(FUNCTIONS[text])

    def _generate_surface(self) -> None:
        try:
            model = function_surface(
                self.formula_input.text(), self.xrange_fields[0].value(), self.xrange_fields[1].value(),
                self.yrange_fields[0].value(), self.yrange_fields[1].value(),
                self.surface_nx.value(), self.surface_ny.value())
            self._replace_model(model, "Поверхность z=f(x,y) построена (геометрия + грани + UV)")
        except (ValueError, OverflowError, TypeError) as exc:
            QMessageBox.warning(self, "Ошибка построения поверхности", str(exc))

    def _switch_solid(self, name: str) -> None:
        if name not in SOLIDS:
            return
        self.model = SOLIDS[name]()
        self.history.clear()
        self.viewport.set_polyhedron(self.model)
        self._refresh()

    def _switch_projection(self) -> None:
        self.viewport.mode = self.projection_combo.currentData()
        self.viewport.update()

    def _toggle_view(self, name: str, checked: bool) -> None:
        setattr(self.viewport, name, checked)
        self.viewport.update()

    def _export_obj(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт модели в OBJ", "model.obj", "Wavefront OBJ (*.obj)")
        if path:
            try:
                save_obj(self.model, Path(path))
                self.statusBar().showMessage(f"Сохранено: {path}", 6000)
            except OSError as exc:
                QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def _save_picture(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить снимок", "polyhedron.png", "PNG (*.png)")
        if path and not self.viewport.grab().save(path, "PNG"):
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить изображение")

    def _help(self) -> None:
        QMessageBox.information(self, "Управление", """
        1. Выберите многогранник (ЛР №6) либо загрузите OBJ (Ctrl+O).
        2. Фигура вращения: введите X Y Z по строкам, выберите ось и N.
        3. Поверхность: задайте f(x,y), интервалы и Nx, Ny.
        4. Для всех типов моделей доступны прежние матричные преобразования.
        5. ЛКМ вращает камеру, колесо — масштаб вида, «Вписать» — камера.
        6. Ctrl+E — сохранить OBJ, Ctrl+S — снимок PNG, Ctrl+Z — отмена.
        7. Рендеринг объекта программный, с массивом глубин (задел для ЛР №8–9).
        """)
