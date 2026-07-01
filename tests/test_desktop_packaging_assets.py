"""Guards that the packaging assets stay wired for the Phase 2 bundle.

These parse the committed build assets as text — they catch regressions
(missing heavy-dep collection, wrong app name, model cache not bundled)
without running a full PyInstaller/Inno build.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_spec_file_named_assist_exists():
    assert (ROOT / "Assist.spec").is_file()
    assert not (ROOT / "Odysseus.spec").exists()


def test_spec_builds_assist_named_app():
    spec = _read("Assist.spec")
    assert "name='Assist'" in spec or 'name="Assist"' in spec
    assert "launcher.py" in spec


def test_spec_collects_heavy_deps():
    spec = _read("Assist.spec")
    for pkg in ("chromadb", "onnxruntime", "fastembed"):
        assert pkg in spec, f"{pkg} not collected in Assist.spec"
    # pywebview backend is easy for PyInstaller to miss.
    assert "webview" in spec


def test_spec_bundles_embedding_model_cache():
    spec = _read("Assist.spec")
    assert "fastembed_cache" in spec


def test_build_script_uses_committed_spec_and_assist_name():
    ps = _read("build-windows-portable.ps1")
    assert "Assist.spec" in ps
    assert "fetch_embedding_model.py" in ps


def test_installer_script_exists_and_names_assist():
    iss = _read("installer/Assist.iss")
    assert 'MyAppName "Assist"' in iss
    assert "Assist-Setup" in iss  # OutputBaseFilename
    assert r"dist\Assist\*" in iss  # bundles the built app folder


def test_installer_does_not_store_user_data_under_app():
    # User data must live in ~/.odysseus/data, never under {app}, so uninstall
    # preserves it. Guard against accidentally adding a data dir to [Files].
    iss = _read("installer/Assist.iss")
    assert "{userappdata}" not in iss
    assert ".odysseus" not in iss


def test_build_installer_script_wires_iscc_and_version():
    ps = _read("build-installer.ps1")
    assert "ISCC" in ps
    assert "Assist.iss" in ps
    assert "APP_VERSION" in ps  # version pulled from the single source
