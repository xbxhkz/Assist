"""Guard: theme.js upgrades an UNTOUCHED saved 'dark' snapshot to the new
Graphite palette (so existing default users see the restyle), and leaves
hand-tweaked or other-named themes alone. Wired into getSaved() alongside the
existing preset-rename migrations. (Text-guard — repo has no JS test runner.)"""
import re
from pathlib import Path

JS = (Path(__file__).resolve().parent.parent / "static" / "js" / "theme.js").read_text(encoding="utf-8")


def test_migration_helper_present():
    assert "PREV_DARK" in JS
    # the previous dark palette it compares against, verbatim
    assert "#282c34" in JS and "#9cdef2" in JS and "#355a66" in JS
    assert re.search(r"function _upgradeDarkSnapshot", JS)


def test_migration_wired_into_getsaved():
    block = re.search(r"export function getSaved\(\) \{(.*?)\n\}", JS, re.S).group(1)
    assert "_upgradeDarkSnapshot" in block
