import src.desktop.capture as cap


class FakeGrab:
    """Returns a tiny fake screenshot object mss-style."""
    def __init__(self):
        self.regions = []
    def __call__(self, region):
        self.regions.append(region)
        class Shot:
            size = (2, 2)
            rgb = b"\x00" * (2 * 2 * 3)
        return Shot()


def test_capture_full_returns_png_bytes():
    fg = FakeGrab()
    out = cap.capture_png("full", grabber=fg)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"     # PNG signature
    assert fg.regions == ["full"]


def test_capture_monitor_index_passed_through():
    fg = FakeGrab()
    cap.capture_png("monitor:2", grabber=fg)
    assert fg.regions == [{"monitor": 2}]


def test_capture_window_uses_rect():
    fg = FakeGrab()
    cap.capture_png("window:123", grabber=fg,
                    window_rect=lambda wid: (10, 20, 30, 40))
    assert fg.regions == [{"left": 10, "top": 20, "width": 30, "height": 40}]
