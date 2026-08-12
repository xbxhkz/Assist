"""Guards that the llama-server bundling stays wired (no full PyInstaller run)."""
import pathlib
import re
import shutil
import subprocess

import pytest

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


# --- background-removal model (U2Net ONNX) ---------------------------------
# Assist.spec's ('build_assets/bg_removal', 'bg_removal') datas entry is a HARD
# PyInstaller error when the source directory is missing, and build_assets/ is
# gitignored — so the portable build script MUST fetch it. sd-server broke
# clean-machine builds this exact way once already (see
# docs/superpowers/plans/2026-07-08-perf-improvements.md).

_PS1 = "build-windows-portable.ps1"


def test_spec_bundles_bg_removal_dir():
    spec = _read("Assist.spec")
    assert "build_assets/bg_removal" in spec
    assert "'bg_removal'" in spec or '"bg_removal"' in spec


def test_build_script_fetches_bg_removal_model():
    ps1 = _read(_PS1)
    assert "fetch_bg_removal_model.py" in ps1
    # A silently-failing fetch is as bad as no fetch: the spec entry still
    # errors out later. Mirror the sibling fetches' exit-code check.
    idx = ps1.index("fetch_bg_removal_model.py")
    assert "$LASTEXITCODE" in ps1[idx:idx + 200]


def _fast_guard_condition():
    """The `-Fast` skip guard's boolean expression, lifted from the script."""
    m = re.search(r'^if \(\$Fast -and (.+)\) \{', _read(_PS1), re.M)
    assert m, "could not find the -Fast build-mode guard in " + _PS1
    return "$Fast -and " + m.group(1)


def _eval_guard(cond, workdir):
    exe = shutil.which("pwsh") or shutil.which("powershell")
    out = subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-Command",
         f"$Fast = $true; if ({cond}) {{ 'SKIP' }} else {{ 'FETCH' }}"],
        cwd=str(workdir), capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.mark.skipif(
    not (shutil.which("pwsh") or shutil.which("powershell")),
    reason="PowerShell not available to evaluate the build script's guard",
)
def test_fast_guard_still_fetches_when_only_bg_removal_is_missing(tmp_path):
    """Behavioural check of the real guard expression: a machine that already
    has llama+sd but NOT bg_removal must still run the fetch step under -Fast,
    otherwise PyInstaller hard-errors on the missing datas source path."""
    cond = _fast_guard_condition()

    for name in ("llama", "sd"):
        (tmp_path / "build_assets" / name).mkdir(parents=True)
    assert _eval_guard(cond, tmp_path) == "FETCH"

    (tmp_path / "build_assets" / "bg_removal").mkdir()
    assert _eval_guard(cond, tmp_path) == "SKIP"


def test_fetch_script_targets_build_assets_bg_removal():
    src = _read("scripts/fetch_bg_removal_model.py")
    assert "build_assets" in src and "bg_removal" in src
