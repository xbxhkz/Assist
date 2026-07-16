# Webcam Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An admin-only `webcam_look` agent tool that grabs one webcam frame, runs YOLO object detection, and returns the detected objects + an annotated snapshot — optionally adding a vision-model scene description.

**Architecture:** Server-side capture (`cv2.VideoCapture`) + server-side YOLO (`ultralytics`), mirroring the existing `capture_screen` tool (injectable grabber, `{"output","image_url"}` result rendered as a chat image bubble). Two new units — `src/desktop/webcam.py` (capture) and `src/vision/yolo.py` (detect/annotate/summarize) — plus the tool, a Camera-access consent toggle, and packaging of `ultralytics` + `yolov8n.pt`.

**Tech Stack:** OpenCV (`cv2`, already bundled), `ultralytics` YOLO (`yolov8n`), FastAPI settings, PyInstaller. Tests use injectable grabber/model + monkeypatch — no real camera or ultralytics in unit tests.

## Global Constraints

- Tool is **admin-only** (register in `tool_security.py` admin lists) and gated by `get_setting("camera_access_enabled", False)`; off → return a clear error telling the user to enable Camera access.
- `camera_access_enabled` defaults `False` and is **reset off on every restart** (`reset_camera_access()`, mirroring `reset_screen_access`, called from `app.py`).
- **Server-side `cv2` capture (Approach A).** If the Task 1 frozen feasibility check fails (`cv2.VideoCapture` can't grab a frame in `Assist.exe`), STOP and escalate to pivot capture to browser `getUserMedia` (Approach B) — `src/vision/yolo.py` and the tool contract are unchanged.
- The camera is opened, one frame read, and **released immediately** — never held between calls.
- Detection uses **`yolov8n`**; ship `build_assets/yolo/yolov8n.pt` so it works **offline on first use**. Declare `ultralytics` in `requirements.txt` and add it to `Assist.spec` `collect_all`.
- Frames are returned **inline in the chat tool result only** — never auto-saved to the Gallery.
- `describe` param defaults from `get_setting("webcam_describe_default", False)`; when on and a vision model is served, reuse `analyze_image_with_vl_result` on the **raw** frame (off the event loop via `asyncio.to_thread`); when no vision model, append a short note.
- Every pytest run uses `--import-mode=importlib`. Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Webcam capture module (`src/desktop/webcam.py`)

**Files:**
- Create: `src/desktop/webcam.py`
- Test: `tests/test_webcam_capture.py`

**Interfaces:**
- Produces: `capture_frame_jpeg(*, grabber=None, index=None) -> bytes` (JPEG). `grabber(index) -> frame` (a BGR numpy array) is injectable; default opens `cv2.VideoCapture(index)`, reads one frame, releases. Raises `RuntimeError` on no-open / no-frame.

- [ ] **Step 1: Frozen feasibility gate (Approach A viability)**

Run against the existing frozen exe (cv2 is already bundled):
`./dist/Assist/Assist.exe --run-py -c "import cv2; c=cv2.VideoCapture(0); ok=c.isOpened(); r=c.read()[0] if ok else False; c.release(); print('OPENED', ok, 'FRAME', r)"`
Expected: `OPENED True FRAME True`. If it prints `False`/errors (no camera reachable from the frozen app), **STOP and report BLOCKED** — the feature must pivot to browser capture (Approach B) before continuing.

- [ ] **Step 2: Write the failing test**

Create `tests/test_webcam_capture.py`:

```python
import numpy as np
import pytest
import src.desktop.webcam as wc


def test_capture_encodes_injected_frame_to_jpeg():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 1] = 200  # a green frame
    out = wc.capture_frame_jpeg(grabber=lambda idx: frame, index=0)
    assert isinstance(out, bytes) and out[:2] == b"\xff\xd8"  # JPEG SOI


def test_capture_raises_when_grabber_fails():
    def boom(idx):
        raise RuntimeError("no camera 0")
    with pytest.raises(RuntimeError):
        wc.capture_frame_jpeg(grabber=boom, index=0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_webcam_capture.py --import-mode=importlib -q`
Expected: FAIL (`module 'src.desktop.webcam' has no attribute 'capture_frame_jpeg'`).

- [ ] **Step 4: Write the implementation**

Create `src/desktop/webcam.py`:

```python
"""Webcam single-frame capture -> JPEG bytes. Real capture uses OpenCV
(cv2.VideoCapture); the grabber is injectable so tests never need a device.
The camera is opened, one frame read, and released immediately -- never held."""
import logging

from src.settings import get_setting

logger = logging.getLogger(__name__)


def _default_grabber(index):
    import cv2
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"no frame from camera {index}")
        return frame  # BGR numpy array
    finally:
        cap.release()


def capture_frame_jpeg(*, grabber=None, index=None):
    """Grab one webcam frame and return JPEG bytes. Raises RuntimeError on
    failure (no camera / no frame). `grabber(index) -> frame` is injectable."""
    if index is None:
        try:
            index = int(get_setting("camera_index", 0))
        except (TypeError, ValueError):
            index = 0
    grab = grabber or _default_grabber
    frame = grab(index)
    import cv2
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("failed to JPEG-encode webcam frame")
    return bytes(buf.tobytes())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_webcam_capture.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/desktop/webcam.py tests/test_webcam_capture.py
git commit -m "feat(webcam): single-frame cv2 capture -> JPEG (injectable grabber)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: YOLO detect / annotate / summarize (`src/vision/yolo.py`)

**Files:**
- Create: `src/vision/__init__.py` (empty), `src/vision/yolo.py`
- Test: `tests/test_yolo_detect.py`

**Interfaces:**
- Consumes: JPEG bytes (from Task 1).
- Produces:
  - `format_detections(raw, w, h, conf=0.4) -> list[Detection]` where `raw` is a list of `(label, confidence, x1, y1, x2, y2)` and `Detection = {"label": str, "confidence": float, "box": [int,int,int,int], "position": str}`.
  - `detect(jpeg, *, model=None, conf=0.4) -> list[Detection]` (decodes JPEG, runs the lazy YOLO singleton, calls `format_detections`).
  - `annotate(jpeg, dets) -> bytes` (boxes+labels, JPEG).
  - `summarize(dets) -> str`.
  - `weights_path(frozen_base=None) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_yolo_detect.py`:

```python
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


def test_detect_uses_injected_model():
    class _Box:
        def __init__(self): self.conf = [0.95]; self.cls = [0]; self.xyxy = [[1.0, 2.0, 30.0, 40.0]]
    class _Res:
        names = {0: "person"}
        boxes = [_Box()]
    fake_model = lambda arr, verbose=False: [_Res()]
    dets = yolo.detect(_jpeg(), model=fake_model)
    assert len(dets) == 1 and dets[0]["label"] == "person"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_yolo_detect.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.vision'`).

- [ ] **Step 3: Write the implementation**

Create `src/vision/__init__.py` (empty file). Create `src/vision/yolo.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_yolo_detect.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vision/__init__.py src/vision/yolo.py tests/test_yolo_detect.py
git commit -m "feat(webcam): YOLO detect/annotate/summarize (pure formatting split out)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Camera-access consent (settings + reset + sidebar toggle)

**Files:**
- Modify: `src/settings.py` (add `camera_access_enabled` + `webcam_describe_default` defaults near line 41; add `reset_camera_access()` after `reset_input_control`)
- Modify: `app.py:991-994` (import + call `reset_camera_access`)
- Modify: `static/index.html` (add the `camera-access-toggle` sidebar row after the screen-access row; add the `<script>`)
- Create: `static/js/cameraAccess.js`
- Test: `tests/test_camera_access_setting.py`

**Interfaces:**
- Produces: setting `camera_access_enabled` (default False) + `webcam_describe_default` (default False); `reset_camera_access()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_access_setting.py`:

```python
import src.settings as settings


def test_camera_access_defaults_off():
    assert settings.DEFAULT_SETTINGS["camera_access_enabled"] is False
    assert settings.DEFAULT_SETTINGS["webcam_describe_default"] is False


def test_reset_camera_access_forces_off(monkeypatch):
    store = {"camera_access_enabled": True}
    monkeypatch.setattr(settings, "load_settings", lambda: dict(store))
    saved = {}
    monkeypatch.setattr(settings, "save_settings", lambda s: saved.update(s))
    settings.reset_camera_access()
    assert saved.get("camera_access_enabled") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_camera_access_setting.py --import-mode=importlib -q`
Expected: FAIL (`KeyError: 'camera_access_enabled'` / `reset_camera_access` missing).

- [ ] **Step 3: Add the settings defaults + reset function**

In `src/settings.py`, in the defaults dict beside `"screen_access_enabled": False,` (line ~41), add:

```python
    "camera_access_enabled": False,
    "webcam_describe_default": False,
```

After `reset_input_control()` (near line 362), add:

```python
def reset_camera_access():
    """Force camera access off. Called at startup so the webcam is never
    silently available across restarts (mirrors reset_screen_access)."""
    try:
        s = load_settings()
        if s.get("camera_access_enabled"):
            s["camera_access_enabled"] = False
            save_settings(s)
    except Exception:
        pass
```

- [ ] **Step 4: Wire the boot-time reset**

In `app.py` (line ~991, where `reset_screen_access, reset_input_control, reset_shell_exec` are imported and called), add `reset_camera_access` to the import and call it:

```python
        from src.settings import reset_screen_access, reset_input_control, reset_shell_exec, reset_camera_access
        reset_screen_access()
        reset_input_control()
        reset_shell_exec()
        reset_camera_access()
```

- [ ] **Step 5: Run the settings test**

Run: `python -m pytest tests/test_camera_access_setting.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Add the sidebar toggle UI**

In `static/index.html`, find the screen-access consent row (`id="screen-access-toggle"`, ~line 953-961) and add an analogous row right after it:

```html
        <div class="list-item">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
          <span class="grow">Camera access</span>
          <span id="camera-access-indicator" class="sidebar-notif-dot" style="display:none;background:var(--green,#50fa7b);margin-right:6px;" title="Camera access is ON"></span>
          <label class="admin-switch" style="flex-shrink:0;" title="Let the AI look through the webcam (object detection)">
            <input type="checkbox" id="camera-access-toggle">
            <span class="admin-slider"></span>
          </label>
        </div>
```

Add the script tag next to `screenAccess.js` (search `static/index.html` for `screenAccess.js`):

```html
<script type="module" src="/static/js/cameraAccess.js"></script>
```

- [ ] **Step 7: Create the toggle wiring**

Create `static/js/cameraAccess.js` (mirror of `screenAccess.js`):

```javascript
// cameraAccess.js — the sidebar "Camera access" switch, backing the
// `camera_access_enabled` setting that gates the desktop `webcam_look` tool.
// Reset to off server-side on every restart (src/settings.py); this widget
// reflects and updates the persisted setting.
(function () {
  function $(id) { return document.getElementById(id); }

  function reflect(on) {
    const t = $('camera-access-toggle');
    if (t) t.checked = !!on;
    const dot = $('camera-access-indicator');
    if (dot) dot.style.display = on ? '' : 'none';
  }

  async function load() {
    try {
      const res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
      const settings = await res.json();
      reflect(!!settings.camera_access_enabled);
    } catch (e) { /* leave default */ }
  }

  async function save(enabled) {
    try {
      await fetch('/api/auth/settings', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_access_enabled: enabled }),
      });
    } catch (e) { /* ignore */ }
    reflect(enabled);
  }

  document.addEventListener('DOMContentLoaded', () => {
    load();
    $('camera-access-toggle')?.addEventListener('change', (e) => {
      save(!!e.target.checked);
    });
  });
})();
```

- [ ] **Step 8: Commit**

```bash
git add src/settings.py app.py static/index.html static/js/cameraAccess.js tests/test_camera_access_setting.py
git commit -m "feat(webcam): camera-access consent (setting + reset-on-restart + sidebar toggle)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `webcam_look` tool + registration

**Files:**
- Modify: `src/agent_tools/desktop_tools.py` (imports + `WebcamLookTool`)
- Modify: `src/agent_tools/__init__.py` (dispatch map line ~67 + admin name list line ~101)
- Modify: `src/tool_schemas.py` (schema after `capture_screen` ~line 264 + name list ~line 1585)
- Modify: `src/tool_security.py` (add to the admin lists at ~line 56 and ~line 138)
- Modify: `src/tool_index.py` (description ~line 86)
- Test: `tests/test_webcam_look_tool.py`

**Interfaces:**
- Consumes: `capture_frame_jpeg` (Task 1); `detect`, `annotate`, `summarize` (Task 2); `camera_access_enabled`/`webcam_describe_default` (Task 3); existing `_vision_ready`, `analyze_image_with_vl_result`, `get_setting`, `_args`.
- Produces: tool name `webcam_look`, result `{"output": str, "image_url": str, "exit_code": int}` or `{"error": str, "exit_code": 1}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_webcam_look_tool.py`:

```python
import asyncio
import src.agent_tools.desktop_tools as dt


def _run(coro):
    return asyncio.run(coro)


def _patch_capture(monkeypatch, dets=None, vision=False, desc_text="a desk"):
    monkeypatch.setattr(dt, "capture_frame_jpeg", lambda: b"\xff\xd8jpeg")
    monkeypatch.setattr(dt, "detect", lambda jpeg: dets if dets is not None else [])
    monkeypatch.setattr(dt, "annotate", lambda jpeg, d: b"\xff\xd8annot")
    monkeypatch.setattr(dt, "summarize", lambda d: "1 person (92%; center)" if d else "No recognizable objects detected.")
    monkeypatch.setattr(dt, "_vision_ready", lambda: vision)
    monkeypatch.setattr(dt, "analyze_image_with_vl_result", lambda p, o: {"text": desc_text})


def test_refuses_when_camera_access_off(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: False)
    r = _run(dt.WebcamLookTool().execute("{}", {"owner": "u"}))
    assert r["exit_code"] == 1 and "Camera access" in r["error"]


def test_returns_objects_and_image(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    _patch_capture(monkeypatch, dets=[{"label": "person"}])
    r = _run(dt.WebcamLookTool().execute("{}", {"owner": "u"}))
    assert r["exit_code"] == 0
    assert "person" in r["output"]
    assert r["image_url"].startswith("data:image/jpeg;base64,")


def test_describe_appends_vision_text(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    _patch_capture(monkeypatch, dets=[{"label": "person"}], vision=True, desc_text="a person at a desk")
    r = _run(dt.WebcamLookTool().execute('{"describe": true}', {"owner": "u"}))
    assert "a person at a desk" in r["output"]


def test_describe_without_vision_notes_it(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    _patch_capture(monkeypatch, dets=[], vision=False)
    r = _run(dt.WebcamLookTool().execute('{"describe": true}', {"owner": "u"}))
    assert "no vision model" in r["output"].lower()


def test_camera_error_is_reported(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    def boom():
        raise RuntimeError("no camera 0")
    monkeypatch.setattr(dt, "capture_frame_jpeg", boom)
    r = _run(dt.WebcamLookTool().execute("{}", {"owner": "u"}))
    assert r["exit_code"] == 1 and "webcam_look" in r["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webcam_look_tool.py --import-mode=importlib -q`
Expected: FAIL (`module 'src.agent_tools.desktop_tools' has no attribute 'WebcamLookTool'`).

- [ ] **Step 3: Add imports + the tool class**

In `src/agent_tools/desktop_tools.py`, add these imports below the existing `from src.desktop.capture import capture_png` line:

```python
from src.desktop.webcam import capture_frame_jpeg
from src.vision.yolo import detect, annotate, summarize
```

Add the tool class at the end of the file:

```python
class WebcamLookTool:
    async def execute(self, content, ctx):
        if not get_setting("camera_access_enabled", False):
            return {"error": "Camera access is off. Ask the user to enable 'Camera "
                             "access' in the sidebar before using the webcam.",
                    "exit_code": 1}
        describe = _args(content).get("describe")
        if describe is None:
            describe = bool(get_setting("webcam_describe_default", False))
        try:
            jpeg = capture_frame_jpeg()
        except Exception as e:
            return {"error": f"webcam_look: no camera or capture failed ({e}).",
                    "exit_code": 1}
        dets = detect(jpeg)
        out = summarize(dets)
        annotated = annotate(jpeg, dets)
        if describe:
            if _vision_ready():
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                        tf.write(jpeg)
                        temp_path = tf.name
                    result = await asyncio.to_thread(
                        analyze_image_with_vl_result, temp_path, ctx.get("owner"))
                    desc = (result or {}).get("text", "")
                    if desc:
                        out = out + "\n\nScene: " + desc
                finally:
                    if temp_path:
                        os.remove(temp_path)
            else:
                out = out + "\n\n(no vision model served for a scene description)"
        uri = "data:image/jpeg;base64," + base64.b64encode(annotated).decode("ascii")
        logger.info("webcam_look: %d objects, describe=%s", len(dets), describe)
        return {"output": out, "image_url": uri, "exit_code": 0}
```

- [ ] **Step 4: Run the tool test to verify it passes**

Run: `python -m pytest tests/test_webcam_look_tool.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Register the tool (dispatch + schema + security + index)**

In `src/agent_tools/__init__.py`: add to the dispatch map beside `"capture_screen": CaptureScreenTool().execute,` (line ~67):

```python
    "webcam_look": WebcamLookTool().execute,
```

Import `WebcamLookTool` wherever `CaptureScreenTool` is imported in that file, and add `"webcam_look"` to the admin tool-name list at line ~101 (the one containing `"capture_screen"`).

In `src/tool_schemas.py`, add this schema object immediately after the `capture_screen` function schema (ends ~line 264):

```python
    {
        "type": "function",
        "function": {
            "name": "webcam_look",
            "description": "Look through the webcam and detect objects (YOLO). Requires the user to enable camera access. Set describe=true to also get a vision-model scene description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "describe": {"type": "boolean", "description": "also add a vision-model scene description (default from settings)"}
                },
                "required": []
            }
        }
    },
```

Add `"webcam_look"` to the tool-name list at line ~1585 (beside `"capture_screen"`).

In `src/tool_security.py`, add `"webcam_look",` to BOTH admin lists (beside `"capture_screen"` at ~line 56 and ~line 138).

In `src/tool_index.py`, add beside the `capture_screen` entry (~line 86):

```python
    "webcam_look": "Look through the webcam and detect objects (YOLO). Requires the user to enable camera access. Optional describe=true adds a vision-model scene description.",
```

- [ ] **Step 6: Verify registration is consistent**

Run: `python -m pytest tests/test_webcam_look_tool.py --import-mode=importlib -q` and
`python -c "from src.agent_tools import *" ` (import smoke — no error).
Also confirm `webcam_look` appears once in each of the 5 registration files:
`grep -rc '"webcam_look"' src/agent_tools/__init__.py src/tool_schemas.py src/tool_security.py src/tool_index.py`
Expected: `__init__.py` ≥1 (dispatch + list), `tool_schemas.py` 2, `tool_security.py` 2, `tool_index.py` 1.

- [ ] **Step 7: Commit**

```bash
git add src/agent_tools/desktop_tools.py src/agent_tools/__init__.py src/tool_schemas.py src/tool_security.py src/tool_index.py tests/test_webcam_look_tool.py
git commit -m "feat(webcam): webcam_look tool (admin-only) + registration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Package (ultralytics + yolov8n weights) + live-verify

**Files:**
- Modify: `requirements.txt` (declare `ultralytics`)
- Modify: `Assist.spec` (`collect_all` += `ultralytics`; `datas` += the bundled weights)
- Create: `build_assets/yolo/yolov8n.pt` (fetched, ~6 MB)
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt; force-added)

**Interfaces:**
- Consumes: Tasks 1-4.

- [ ] **Step 1: Fetch the YOLO weight into build_assets**

Run:
```bash
mkdir -p build_assets/yolo
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"   # downloads yolov8n.pt to CWD
mv yolov8n.pt build_assets/yolo/yolov8n.pt
ls -la build_assets/yolo/yolov8n.pt
```
Expected: `build_assets/yolo/yolov8n.pt` exists (~6 MB).

- [ ] **Step 2: Declare + collect ultralytics; bundle the weight**

In `requirements.txt`, add (with a short comment):
```
# Webcam object detection (services via the webcam_look tool): YOLO via
# ultralytics. Ships by default so detection works offline; the yolov8n weight
# is bundled under build_assets/yolo (Assist.spec).
ultralytics
```

In `Assist.spec`, add `"ultralytics"` to the `collect_all` package tuple (the loop over `chromadb`/`onnxruntime`/`faster_whisper`/...), and add the weights to `datas`:
```python
    ('build_assets/yolo', 'yolo'),
```

- [ ] **Step 3: Full affected-suite run**

Run: `python -m pytest tests/test_webcam_capture.py tests/test_yolo_detect.py tests/test_camera_access_setting.py tests/test_webcam_look_tool.py --import-mode=importlib -q`
Expected: PASS (14 passed).

- [ ] **Step 4: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File ./build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 5: Frozen verification (ultralytics bundled + weights + real detect)**

Run:
```
./dist/Assist/Assist.exe --run-py -c "import ultralytics, cv2, numpy as np; from src.vision.yolo import weights_path, detect; print('WEIGHTS', weights_path()); ok,buf=cv2.imencode('.jpg', np.zeros((64,64,3),np.uint8)); print('DETECT_OK', isinstance(detect(buf.tobytes()), list))"
```
Expected: `WEIGHTS` ends in a real bundled path (`...\yolo\yolov8n.pt`, not the bare filename), and `DETECT_OK True` (ultralytics loaded the bundled weight and ran on a blank frame with no crash / no network).

- [ ] **Step 6: Live-verify in the running app (manual)**

Reinstall, then as an admin: enable **Camera access** in the sidebar → ask the agent "look through my webcam / what do you see?" → it returns a list of detected objects with an annotated snapshot in chat; try "describe what you see" (with a vision model served) → adds a scene description; turn Camera access **off** → the tool refuses with the enable-camera message; confirm Camera access is **off again after restarting** the app.

- [ ] **Step 7: Commit the installer**

```bash
git add requirements.txt Assist.spec
git add -f build_assets/yolo/yolov8n.pt installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with webcam object detection (ultralytics + yolov8n)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- Every pytest run uses `--import-mode=importlib`.
- Unit tests never open a real camera or load ultralytics — capture uses an injected grabber, detect uses an injected model, and the tool test monkeypatches `capture_frame_jpeg`/`detect`/`annotate`/`summarize`/`_vision_ready`/`analyze_image_with_vl_result`.
- Task 1 Step 1 is the go/no-go for Approach A: if `cv2.VideoCapture` can't grab a frame in the frozen exe, STOP and escalate — the pivot to browser capture only changes `src/desktop/webcam.py`'s source, not the tool contract or Tasks 2-4.
- Do not auto-save webcam frames to the Gallery; the annotated frame is returned inline via `image_url` only.
