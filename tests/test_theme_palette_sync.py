"""Guard: the default theme's :root palette (style.css) and THEMES.dark
(theme.js) must stay identical, and be the approved Graphite values."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSS = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "js" / "theme.js").read_text(encoding="utf-8")

GRAPHITE = {"bg": "#13151A", "panel": "#1C1F27", "fg": "#E6E9EE",
            "border": "#2A2F3A", "red": "#45C4B0"}


def _root_var(name: str) -> str:
    # first :root definition wins (the default dark theme block)
    m = re.search(r"--%s:\s*(#[0-9a-fA-F]{6})" % name, CSS)
    assert m, f"--{name} not found in style.css"
    return m.group(1).upper()


def _themes_dark(key: str) -> str:
    block = re.search(r"dark:\s*\{([^}]*)\}", JS).group(1)
    m = re.search(r"%s:\s*'(#[0-9a-fA-F]{6})'" % key, block)
    assert m, f"{key} not found in THEMES.dark"
    return m.group(1).upper()


def test_root_matches_graphite():
    for css_name, key in [("bg", "bg"), ("panel", "panel"), ("fg", "fg"),
                          ("border", "border"), ("red", "red")]:
        assert _root_var(css_name) == GRAPHITE[key], css_name


def test_themes_dark_matches_root():
    for css_name, key in [("bg", "bg"), ("panel", "panel"), ("fg", "fg"),
                          ("border", "border"), ("red", "red")]:
        assert _themes_dark(key) == _root_var(css_name), key
