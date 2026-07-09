"""Text guards: the Folder access settings card is wired (standing extra
roots for the agent's file tools via tool_path_extra_roots)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_folder_access_card():
    html = _read("static/index.html")
    for el in ('id="set-folderAccessList"', 'id="set-folderAccessInput"',
               'id="set-folderAccessAdd"'):
        assert el in html, f"{el} missing from index.html"


def test_settings_js_wires_folder_access():
    js = _read("static/js/settings.js")
    assert "tool_path_extra_roots" in js
    assert "set-folderAccessAdd" in js
    assert "/api/workspace/vet" in js          # non-root paths get vetted
    assert "ENTIRE" in js                      # drive-root grants need confirm
