"""Text guards that the Help feature is wired (sidebar entry, modal, tabs,
manual content, tutorials launcher, about version)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_help_entry_and_modal():
    html = _read("static/index.html")
    for el in ('id="tool-help-btn"', 'id="help-modal"', 'id="help-tabs"',
               'id="help-tab-manual"', 'id="help-tab-tutorials"',
               'id="help-tab-about"', 'id="help-tutorials-list"',
               'id="help-about-version"'):
        assert el in html, f"{el} missing from index.html"
    assert 'src="/static/js/help.js"' in html


def test_manual_covers_the_core_features():
    html = _read("static/index.html")
    manual = html.split('id="help-tab-manual"', 1)[1].split('id="help-tab-tutorials"', 1)[0]
    for topic in ("Getting started", "Agent", "Local LLMs", "Image generation",
                  "Gallery", "Tasks", "Skills", "Keyboard shortcuts",
                  "Troubleshooting", "VS Code", "Folder access"):
        assert topic in manual, f"manual section missing: {topic}"


def test_help_js_wires_tutorials_and_about():
    js = _read("static/js/help.js")
    assert "/api/ready" in js                    # about: version source
    assert "tool-help-btn" in js                 # opener
    for cmd in ("/demo", "/tour-settings", "/tour-gallery", "/tour-notes"):
        assert cmd in js, f"tutorial entry missing: {cmd}"


def test_tutorial_commands_exist_in_slash_registry():
    """Every Start button must map to a registered slash command (or alias)."""
    js = _read("static/js/help.js")
    registry = _read("static/js/slashCommands.js")
    cmds = re.findall(r"cmd:\s*'/([\w-]+)'", js)
    assert cmds, "no tutorial commands found in help.js"
    for c in cmds:
        assert re.search(rf"(^\s*(?:'{c}'|{c}):|alias:.*'{c}')", registry, re.M), \
            f"/{c} is not a registered slash command"
