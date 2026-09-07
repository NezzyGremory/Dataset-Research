import sys

from PySide6.QtWidgets import QApplication

from app.core.config import APP_NAME
from app.core.logger import setup_logger
from app.ui.main_window import MainWindow


def main():
    logger = setup_logger()

    logger.info("Starting %s", APP_NAME)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()

    logger.info("Application started successfully")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()