# Webcam Object Detection Design

**Goal:** An admin-only agent tool that lets the AI "look through the webcam":
capture a single frame, run YOLO object detection, and return the detected
objects (with an annotated snapshot shown in chat) — optionally adding a
vision-model scene description.

**Scope:** On-demand, single-frame detection exposed as an agent tool. Object
detection is **YOLO** (ultralytics); a `describe` option additionally runs the
frame through the already-served vision model. **Not** live/continuous
detection, face recognition, tracking, or a multi-camera UI — those are out of
scope (see Non-goals). This is one self-contained sub-project.

---

## Background — what exists (and is reused)

- **`capture_screen` tool** (`src/agent_tools/desktop_tools.py`, `CaptureScreenTool`)
  is the template. It: gates on `get_setting("screen_access_enabled", False)`,
  captures via an injectable grabber, base64-encodes to a `data:` URI, runs the
  vision model off the event loop, and returns
  `{"output": <text>, "image_url": <uri>, "exit_code": 0}`. The UI renders
  `result["image_url"]` as an image bubble (`src/tool_execution.py:471-492`).
- **`src/desktop/capture.py`** — `capture_png(target, *, grabber=None)` with a
  `_default_grabber` using `mss`. The **injectable-grabber** pattern is what the
  webcam capture module mirrors (so tests never need a real device).
- **VLM description** — `analyze_image_with_vl_result(path, owner) -> {"text": …}`,
  invoked via `asyncio.to_thread` (the first call can auto-serve the vision model
  and take tens of seconds). `_vision_ready()` = `vision_enabled` + `vision_model`.
- **Consent pattern** — `screen_access_enabled` defaults `False`
  (`src/settings.py:41`); `reset_screen_access()` (`src/settings.py:351`) forces
  it off on every boot so a capability is never silently on across restarts. The
  sidebar has a matching `screen-access-toggle` row.
- **Libraries** — `cv2` (OpenCV 4.13) is importable **and already bundled** in
  the frozen exe. `ultralytics` (YOLO 8.4) is importable in dev but **transitive
  and NOT bundled** — it must be declared + collected (the faster-whisper lesson).
- **Tool shape** — tools are classes with `async def execute(self, content, ctx)`
  registered via `tool_index` / `tool_schemas` / `tool_security` (admin-gated).

## Architecture (server-side capture + server-side YOLO)

The camera is on the same machine as the server, so the tool captures and
detects synchronously in-process — no browser round-trip (mirrors
`capture_screen`/`mss`).

```
AI calls webcam_look(describe?)
  └─ WebcamLookTool.execute (admin-gated by registration)
       ├─ gate: get_setting("camera_access_enabled", False)  → error if off
       ├─ webcam.capture_frame_jpeg()        # cv2.VideoCapture(0), one frame, release
       ├─ yolo.detect(jpeg)                  # ultralytics yolov8n → [Detection]
       ├─ yolo.annotate(jpeg, dets)          # draw boxes/labels (cv2)
       ├─ yolo.summarize(dets)               # "1 person (92%, center), 1 laptop (88%, left)"
       ├─ if describe and _vision_ready():    # reuse analyze_image_with_vl_result on the RAW frame
       │     desc = await asyncio.to_thread(analyze_image_with_vl_result, raw_path, owner)
       └─ return {"output": summary [+ "\n\nScene: " + desc], "image_url": <annotated data URI>, "exit_code": 0}
```

### Unit 1 — `src/desktop/webcam.py`

- `capture_frame_jpeg(*, grabber=None) -> bytes` — grabs one frame and returns
  JPEG bytes. `_default_grabber()` opens `cv2.VideoCapture(0)` (with the Windows
  MSMF/DirectShow default), reads one frame, and **releases the device
  immediately** (never held open between calls). Raises a clear exception if no
  camera opens or no frame is read. `grabber` is injectable — tests pass a fake
  returning a synthetic frame (an `numpy` array), so no real camera is needed.
- A module-level `camera_index` (default 0) read from settings so a machine with
  multiple cameras can pick one; no UI picker in v1.

### Unit 2 — `src/vision/yolo.py`

- `detect(jpeg, *, model=None, conf=0.4) -> list[Detection]` where
  `Detection = {"label": str, "confidence": float, "box": [x1,y1,x2,y2], "position": str}`.
  `position` is derived from the box center → one of
  `{"top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right"}`.
  The YOLO model is a **lazy-loaded singleton** (`ultralytics.YOLO(<weights>)`),
  injectable for tests. Only detections `>= conf` are returned.
- `annotate(jpeg, dets) -> bytes` — draws boxes + `label conf%` labels with `cv2`
  and re-encodes JPEG.
- `summarize(dets) -> str` — groups by label with counts and the rounded
  confidence + position, e.g. `"2 people (91%, 88%), 1 laptop (84%, center)"`;
  empty → `"No recognizable objects detected."`

### Unit 3 — `WebcamLookTool` (`src/agent_tools/desktop_tools.py`)

`async def execute(self, content, ctx)`:
1. If `not get_setting("camera_access_enabled", False)` → return
   `{"error": "Camera access is off. Ask the user to enable 'Camera access' in the sidebar.", "exit_code": 1}`.
2. Parse `describe` from `_args(content)` (bool; default from
   `get_setting("webcam_describe_default", False)`).
3. `jpeg = webcam.capture_frame_jpeg()` (wrapped; camera errors → clear error).
4. `dets = yolo.detect(jpeg)`; `summary = yolo.summarize(dets)`;
   `annotated = yolo.annotate(jpeg, dets)`.
5. If `describe`: if `_vision_ready()`, run the VLM on the **raw** frame (temp
   file, `asyncio.to_thread(analyze_image_with_vl_result, …)`) and append its
   text; else append a one-line note that no vision model is served.
6. Return `{"output": <summary [+ description]>, "image_url": "data:image/jpeg;base64," + b64(annotated), "exit_code": 0}`.

Registered **admin-only** in `tool_index`/`tool_schemas`/`tool_security`
alongside the other desktop tools.

### Unit 4 — Camera-access consent toggle

- Setting `camera_access_enabled` (default `False`, `src/settings.py`).
- `reset_camera_access()` mirroring `reset_screen_access()`, called at boot so it
  is never silently on across restarts.
- Sidebar `camera-access-toggle` row mirroring `screen-access-toggle` (its own
  green "on" dot; posts to the same settings-toggle route the screen-access
  switch uses). Admin-only visibility.

### Packaging

- `requirements.txt` += `ultralytics` (no longer transitive-only).
- `Assist.spec` `collect_all` loop += `ultralytics` (pulls its submodules/data;
  native deps of torch/cv2 already handled).
- Ship the **`yolov8n.pt`** weight (~6 MB) under `build_assets/yolo/` and add it
  to the spec `datas`, so `yolo.py` loads a bundled path and detection works
  **offline on first use** (ultralytics otherwise downloads it from GitHub).

## Error handling

- Consent off → clear "enable Camera access" error (step 1).
- No camera / `cv2` open or read fails → `"No camera detected or camera access failed."`
- `describe` requested but no vision model served → return YOLO objects plus a
  short note ("no vision model served for a description").
- YOLO weight missing in dev → ultralytics downloads it; in the frozen build the
  bundled weight is used (no network needed).
- The camera is released after every grab; a failure never leaves it open.

## Testing

- **`webcam.py`** — `capture_frame_jpeg` with an injected fake grabber (synthetic
  numpy frame) → valid JPEG bytes; the no-frame path raises a clear error. No
  real camera in tests.
- **`yolo.py`** — `detect` with an injected fake model (canned boxes/classes) →
  correct `Detection` list + `conf` filtering; `position` mapping for known box
  centers; `summarize` grouping/formatting incl. the empty case; `annotate`
  returns decodable JPEG.
- **`WebcamLookTool`** — gating (consent off → error), `describe` off (YOLO only),
  `describe` on with a mocked `analyze_image_with_vl_result`, and `describe` on
  with no vision model (note appended); camera-error path. All with injected
  capture/detect so no device or model is loaded.
- **Feasibility check (first task):** confirm `cv2.VideoCapture` grabs a frame in
  the **frozen exe** (`Assist.exe --run-py`). If it fails, capture pivots to
  browser `getUserMedia` (Approach B) — the detect/annotate/summarize units and
  the tool contract are unchanged.
- **Live-verify:** real webcam in the running app — `webcam_look`, `webcam_look`
  with `describe`, consent-off refusal, no-camera message.

## Non-goals (v1)

- Live/continuous detection or a real-time overlay panel (separate sub-project).
- Face recognition / identity / tracking; custom or non-COCO models.
- A multi-camera picker UI (a `camera_index` setting only).
- Auto-saving webcam frames to the Gallery (frames are returned inline in chat
  only, for privacy).
