import os
import sys
from unittest import mock
import pytest
import src.runtime_paths as rp
from src.runtime_paths import get_app_root, get_default_data_dir


def test_get_app_root_normal_run():
    """Verify that get_app_root returns the repository root parent of src/ when not frozen."""
    with mock.patch.object(sys, "frozen", False, create=True):
        app_root = get_app_root()
        # Verify it is a valid directory path and matches expected parent structure
        assert os.path.isdir(app_root)
        assert os.path.exists(os.path.join(app_root, "src"))


def test_get_app_root_frozen_with_meipass():
    """Verify that get_app_root returns the sys._MEIPASS directory when frozen by PyInstaller."""
    mock_meipass = os.path.abspath("mock_meipass_dir")
    with mock.patch.object(sys, "frozen", True, create=True), \
         mock.patch.object(sys, "_MEIPASS", mock_meipass, create=True):
        app_root = get_app_root()
        assert app_root == mock_meipass


def test_get_app_root_frozen_without_meipass():
    """Verify that get_app_root falls back to the sys.executable parent directory when frozen but _MEIPASS is absent."""
    mock_exe_path = os.path.join(os.path.abspath("mock_exe_dir"), "Odysseus.exe")
    with mock.patch.object(sys, "frozen", True, create=True), \
         mock.patch.object(sys, "executable", mock_exe_path, create=True):
        # Remove sys._MEIPASS if it exists in the test process environment
        if hasattr(sys, "_MEIPASS"):
            delattr(sys, "_MEIPASS")
        app_root = get_app_root()
        assert app_root == os.path.abspath("mock_exe_dir")


def test_get_default_data_dir_normal():
    """Verify that get_default_data_dir resolves to get_app_root() / 'data' when not frozen."""
    with mock.patch.object(sys, "frozen", False, create=True):
        res = get_default_data_dir()
        assert res == os.path.join(get_app_root(), "data")


def test_get_default_data_dir_frozen(tmp_path):
    """Verify that get_default_data_dir resolves to ~/.assist/data when frozen."""
    with mock.patch.object(sys, "frozen", True, create=True), \
         mock.patch.object(os.path, "expanduser", return_value=str(tmp_path)):
        res = get_default_data_dir()
        expected = str(tmp_path / ".assist" / "data")
        assert res == expected


def test_frozen_data_dir_migrates_legacy(tmp_path):
    legacy = tmp_path / ".odysseus" / "data"
    legacy.mkdir(parents=True)
    (legacy / "app.db").write_text("data")
    got = rp._frozen_data_dir(str(tmp_path))
    assert got == str(tmp_path / ".assist" / "data")
    assert (tmp_path / ".assist" / "data" / "app.db").read_text() == "data"
    assert not legacy.exists()  # moved, not copied


def test_frozen_data_dir_no_legacy_returns_assist(tmp_path):
    got = rp._frozen_data_dir(str(tmp_path))
    assert got == str(tmp_path / ".assist" / "data")


def test_frozen_data_dir_prefers_existing_assist(tmp_path):
    (tmp_path / ".assist" / "data").mkdir(parents=True)
    (tmp_path / ".odysseus" / "data").mkdir(parents=True)
    (tmp_path / ".odysseus" / "data" / "keep.txt").write_text("x")
    got = rp._frozen_data_dir(str(tmp_path))
    assert got == str(tmp_path / ".assist" / "data")
    assert (tmp_path / ".odysseus" / "data" / "keep.txt").exists()  # legacy untouched
