import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_project_root() -> Path:
    """Mengembalikan direktori root proyek Dataset-Research."""
    # File ini berada di app/core/config.py -> parent x 3 = root
    return Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    """Return the writable runtime data directory for this application."""
    if not getattr(sys, "frozen", False):
        return get_project_root() / "data"

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        data_dir = Path(local_app_data) / "Dataset Research"
    else:
        data_dir = Path.home() / "AppData" / "Local" / "Dataset Research"

    data_dir.mkdir(parents=True, exist_ok=True)

    legacy_dir = Path(sys.executable).resolve().parent / "data"
    try:
        if legacy_dir.is_dir() and legacy_dir.resolve() != data_dir.resolve():
            for source in legacy_dir.rglob("*"):
                if not source.is_file():
                    continue
                destination = data_dir / source.relative_to(legacy_dir)
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    except OSError:
        # Keep the original data in place if migration is blocked; the app
        # can still start with its writable per-user data directory.
        pass

    return data_dir


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