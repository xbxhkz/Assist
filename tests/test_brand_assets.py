"""Brand assets exist and have the right shapes (generated from source PNGs)."""
import pathlib

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_app_icon_ico_exists():
    assert (ROOT / "static" / "icon.ico").is_file()


def test_pwa_icons_sizes():
    assert Image.open(ROOT / "static" / "icons" / "icon-192.png").size == (192, 192)
    assert Image.open(ROOT / "static" / "icons" / "icon-512.png").size == (512, 512)
    assert Image.open(ROOT / "static" / "icons" / "icon-maskable-512.png").size == (512, 512)


def test_wordmark_present():
    assert (ROOT / "docs" / "assist-wordmark.png").is_file()
