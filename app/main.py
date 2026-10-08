import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.core.config import init_environment
init_environment()

from app.ui.main_window import MainWindow


def main():
    # Mengizinkan fractional scaling (125%, 150%, 175%) agar pixel-perfect di berbagai resolusi laptop
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()