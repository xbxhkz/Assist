"""Generate Assist brand assets from the provided source PNGs (Phase 4).

Center-crops assistappicon.png to a square, resizes, and writes the Windows
app icon (.ico), the PWA icons, and a favicon; copies assistlogo.png into docs
as the wordmark. Run once at rebrand/build time; outputs are committed.
"""
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _square(img):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def main() -> int:
    src = os.path.join(ROOT, "assistappicon.png")
    img = _square(Image.open(src).convert("RGBA"))

    img.save(os.path.join(ROOT, "static", "icon.ico"),
             sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

    icons = os.path.join(ROOT, "static", "icons")
    img.resize((192, 192)).save(os.path.join(icons, "icon-192.png"))
    img.resize((512, 512)).save(os.path.join(icons, "icon-512.png"))
    img.resize((512, 512)).save(os.path.join(icons, "icon-maskable-512.png"))
    img.resize((32, 32)).save(os.path.join(ROOT, "static", "favicon.png"))

    logo = os.path.join(ROOT, "assistlogo.png")
    if os.path.isfile(logo):
        Image.open(logo).save(os.path.join(ROOT, "docs", "assist-wordmark.png"))

    print("brand assets generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
