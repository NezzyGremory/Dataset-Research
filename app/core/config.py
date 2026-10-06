import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_project_root() -> Path:
    """Mengembalikan direktori root proyek Dataset-Research."""
    # File ini berada di app/core/config.py -> parent x 3 = root
    return Path(__file__).resolve().parent.parent.parent


def init_environment() -> Optional[Path]:
    """
    Muat file .env secara andal dari berbagai kemungkinan lokasi:
    1. Project root (direktori utama proyek)
    2. Direktori executable (saat di-compile jadi .exe via PyInstaller)
    3. sys._MEIPASS (onefile mode PyInstaller)
    4. Current working directory
    5. Subdirektori data/.env

    Mengembalikan Path file .env yang berhasil dimuat, atau None.
    """
    candidates = []

    # 1. Project root
    root_dir = get_project_root()
    candidates.append(root_dir / ".env")

    # 2. PyInstaller / Frozen executable directory
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / ".env")
        if hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS) / ".env")

    # 3. Current working directory
    candidates.append(Path.cwd() / ".env")

    # 4. Folder data/.env
    candidates.append(root_dir / "data" / ".env")
    candidates.append(Path.cwd() / "data" / ".env")

    # Coba muat dari kandidat yang ditemukan
    for env_path in candidates:
        try:
            if env_path.is_file():
                load_dotenv(dotenv_path=env_path, override=True)
                return env_path
        except Exception:
            continue

    # Fallback pencarian standar python-dotenv
    load_dotenv(override=True)
    return None


# Muat environment saat modul pertama kali diimport
LOADED_ENV_PATH = init_environment()

APP_NAME = os.getenv("APP_NAME", "Dataset Research")
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", os.getenv("SERPAPI_KEY", ""))