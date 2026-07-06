from pathlib import Path

CSS = (Path(__file__).resolve().parent.parent / "static" / "style.css").read_text(encoding="utf-8")


def test_sidebar_text_enlarged():
    assert "/* Larger sidebar text" in CSS
    assert ".sidebar .list-item .grow" in CSS
