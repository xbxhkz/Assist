"""Guards that the llama-server bundling stays wired (no full PyInstaller run)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_spec_bundles_llama_dir():
    assert "build_assets/llama" in _read("Assist.spec")
    assert "'llama'" in _read("Assist.spec") or '"llama"' in _read("Assist.spec")


def test_build_script_fetches_llama_server():
    assert "fetch_llama_server.py" in _read("build-windows-portable.ps1")


def test_fetch_script_targets_build_assets_llama():
    src = _read("scripts/fetch_llama_server.py")
    assert "build_assets" in src and "llama" in src
