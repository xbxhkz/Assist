"""Guard: radius tokens exist and the shared window/control selectors use
them (so rounding is consistent and tunable, not scattered literals)."""
import re
from pathlib import Path

CSS = (Path(__file__).resolve().parent.parent / "static" / "style.css").read_text(encoding="utf-8")


def test_radius_tokens_defined():
    for tok, val in [("--radius-window", "16px"), ("--radius-card", "12px"),
                     ("--radius-control", "8px")]:
        assert re.search(rf"{tok}:\s*{val}", CSS), tok


def test_modal_content_uses_window_token():
    # line-anchored so we hit the BASE `.modal-content {` rule, not a
    # `.foo .modal-content {` scoped override.
    block = re.search(r"^\s*\.modal-content \{(.*?)\}", CSS, re.S | re.M).group(1)
    assert "border-radius:var(--radius-window)" in block.replace(" ", "")


def test_base_controls_use_control_token():
    block = re.search(r"^\s*input, textarea, button, select \{(.*?)\}",
                      CSS, re.S | re.M).group(1)
    assert "border-radius:var(--radius-control)" in block.replace(" ", "")


def test_card_token_is_applied():
    # cards/menus (.admin-card, .export-dropdown-menu, .model-picker-menu) route
    # through the card token, not scattered literals.
    assert CSS.count("var(--radius-card)") >= 3

