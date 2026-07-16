"""YOLO object detection on a JPEG frame: detect -> annotate -> summarize.
The model is a lazy singleton (ultralytics). Formatting is split into the pure
`format_detections`/`summarize`/`_position` helpers so tests need no ultralytics."""
import logging
import os
import sys
from collections import OrderedDict

logger = logging.getLogger(__name__)

_model = None


def _position(cx, cy, w, h):
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def weights_path(frozen_base=None):
    """Bundled yolov8n.pt path when frozen, else 'yolov8n.pt' (ultralytics
    resolves/downloads it). `frozen_base` injectable for tests."""
    base = frozen_base
    if base is None and getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
    if base:
        p = os.path.join(base, "yolo", "yolov8n.pt")
        if os.path.isfile(p):
            return p
    return "yolov8n.pt"


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(weights_path())
    return _model


def format_detections(raw, w, h, conf=0.4):
    """raw: list of (label, confidence, x1, y1, x2, y2). Filter by conf,
    round the box, and add a grid position. Pure."""
    dets = []
    for label, c, x1, y1, x2, y2 in raw:
        if float(c) < conf:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dets.append({
            "label": label,
            "confidence": round(float(c), 2),
            "box": [round(x1), round(y1), round(x2), round(y2)],
            "position": _position(cx, cy, w, h),
        })
    return dets


def detect(jpeg, *, model=None, conf=0.4):
    """Detect objects in JPEG bytes -> list[Detection]."""
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return []
    h, w = arr.shape[:2]
    m = model or _get_model()
    raw = []
    for r in m(arr, verbose=False):
        names = r.names
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            raw.append((names[int(b.cls[0])], float(b.conf[0]), x1, y1, x2, y2))
    return format_detections(raw, w, h, conf)


def annotate(jpeg, dets):
    """Draw boxes + labels -> JPEG bytes (returns the input on decode failure)."""
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return jpeg
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(arr, f"{d['label']} {int(d['confidence'] * 100)}%",
                    (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 0), 1)
    ok, buf = cv2.imencode(".jpg", arr)
    return bytes(buf.tobytes()) if ok else jpeg


def _plural(label):
    if label == "person":
        return "people"
    return label if label.endswith("s") else label + "s"


def summarize(dets):
    """Group by label with counts, confidences, and positions -> one line."""
    if not dets:
        return "No recognizable objects detected."
    groups = OrderedDict()
    for d in dets:
        groups.setdefault(d["label"], []).append(d)
    parts = []
    for label, items in groups.items():
        n = len(items)
        confs = ", ".join(f"{int(i['confidence'] * 100)}%" for i in items)
        pos = ", ".join(sorted({i["position"] for i in items}))
        name = label if n == 1 else _plural(label)
        parts.append(f"{n} {name} ({confs}; {pos})")
    return ", ".join(parts)
