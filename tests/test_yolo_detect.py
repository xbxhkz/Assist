import cv2
import numpy as np
import src.vision.yolo as yolo


def _jpeg(w=90, h=60):
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), dtype=np.uint8))
    return buf.tobytes()


def test_position_grid():
    # 90x60 image: thirds at x=30/60, y=20/40
    assert yolo._position(10, 10, 90, 60) == "top-left"
    assert yolo._position(45, 30, 90, 60) == "center"
    assert yolo._position(80, 30, 90, 60) == "right"
    assert yolo._position(45, 50, 90, 60) == "bottom"


def test_format_detections_filters_and_positions():
    raw = [
        ("person", 0.92, 40.2, 10.0, 60.8, 55.0),  # center-ish, kept
        ("cup", 0.30, 0.0, 0.0, 10.0, 10.0),        # below conf, dropped
    ]
    dets = yolo.format_detections(raw, 90, 60, conf=0.4)
    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "person" and d["confidence"] == 0.92
    assert d["box"] == [40, 10, 61, 55]
    assert d["position"] in {"center", "middle", "right"}


def test_summarize_groups_and_pluralizes():
    dets = [
        {"label": "person", "confidence": 0.91, "box": [0, 0, 1, 1], "position": "left"},
        {"label": "person", "confidence": 0.88, "box": [0, 0, 1, 1], "position": "right"},
        {"label": "laptop", "confidence": 0.84, "box": [0, 0, 1, 1], "position": "center"},
    ]
    s = yolo.summarize(dets)
    assert "2 people" in s and "laptop" in s and "91%" in s
    assert yolo.summarize([]) == "No recognizable objects detected."


def test_annotate_returns_decodable_jpeg():
    jpeg = _jpeg()
    dets = [{"label": "cup", "confidence": 0.7, "box": [5, 5, 40, 40], "position": "left"}]
    out = yolo.annotate(jpeg, dets)
    assert out[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) is not None


def test_annotate_supports_png_format():
    jpeg = _jpeg()
    dets = [{"label": "cup", "confidence": 0.7, "box": [5, 5, 40, 40], "position": "left"}]
    out = yolo.annotate(jpeg, dets, fmt=".png")
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    assert cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) is not None


def test_detect_uses_injected_model():
    class _Box:
        def __init__(self): self.conf = [0.95]; self.cls = [0]; self.xyxy = [[1.0, 2.0, 30.0, 40.0]]
    class _Res:
        names = {0: "person"}
        boxes = [_Box()]
    fake_model = lambda arr, verbose=False: [_Res()]
    dets = yolo.detect(_jpeg(), model=fake_model)
    assert len(dets) == 1 and dets[0]["label"] == "person"
