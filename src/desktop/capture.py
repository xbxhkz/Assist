"""Screen capture → PNG bytes. Real capture uses mss; the grabber is injected
so tests never touch a display."""
import struct
import zlib


def _png_from_rgb(width, height, rgb):
    """Encode raw RGB bytes to PNG without Pillow (stdlib zlib only)."""
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)  # filter type 0
        raw.extend(rgb[y * stride:(y + 1) * stride])
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (sig + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


def _region_for(target, window_rect):
    if target == "full":
        return "full"
    if target.startswith("monitor:"):
        return {"monitor": int(target.split(":", 1)[1])}
    if target.startswith("window:"):
        wid = int(target.split(":", 1)[1])
        l, t, w, h = (window_rect or _default_window_rect)(wid)
        return {"left": l, "top": t, "width": w, "height": h}
    return "full"


def capture_png(target="full", *, grabber=None, window_rect=None):
    grabber = grabber or _default_grabber
    region = _region_for(target, window_rect)
    shot = grabber(region)
    w, h = shot.size
    return _png_from_rgb(w, h, shot.rgb)


def _default_grabber(region):  # pragma: no cover (needs a display)
    import mss
    with mss.mss() as sct:
        if region == "full":
            mon = sct.monitors[0]
        elif isinstance(region, dict) and "monitor" in region:
            mon = sct.monitors[region["monitor"]]
        else:
            mon = region
        img = sct.grab(mon)
        class Shot:
            size = (img.width, img.height)
            rgb = img.rgb
        return Shot()


def _default_window_rect(window_id):  # pragma: no cover (Windows-only)
    import ctypes
    import ctypes.wintypes
    rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(window_id, ctypes.byref(rect))
    return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
