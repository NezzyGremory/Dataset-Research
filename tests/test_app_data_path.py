from pathlib import Path

import pytest

from app.core import config


@pytest.fixture(autouse=True)
def clear_cached_data_directory():
    config.get_data_dir.cache_clear()
    yield
    config.get_data_dir.cache_clear()


def test_development_data_directory_stays_in_project(monkeypatch):
    monkeypatch.setattr(config.sys, "frozen", False, raising=False)

    assert config.get_data_dir() == config.get_project_root() / "data"


def test_packaged_data_directory_migrates_adjacent_legacy_data(tmp_path, monkeypatch):
    executable_dir = tmp_path / "portable"
    legacy_dir = executable_dir / "data"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.json").write_text('{"setting": true}', encoding="utf-8")
    (legacy_dir / "projects" / "1").mkdir(parents=True)
    (legacy_dir / "projects" / "1" / "version.csv").write_text("x\n1\n", encoding="utf-8")

    monkeypatch.setattr(config.sys, "frozen", True, raising=False)
    monkeypatch.setattr(config.sys, "executable", str(executable_dir / "Dataset Research.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    data_dir = config.get_data_dir()

    assert data_dir == tmp_path / "LocalAppData" / "Dataset Research"
    assert (data_dir / "config.json").read_text(encoding="utf-8") == '{"setting": true}'
    assert (data_dir / "projects" / "1" / "version.csv").is_file()
    assert (legacy_dir / "config.json").is_file()