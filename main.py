"""Запуск: python main.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("Нет PySide6. Установите зависимости: pip install -r requirements.txt", file=sys.stderr)
        return 1
    from lab6.ui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Лабораторная №7 — 3D модели")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
