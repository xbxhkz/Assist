# Desktop Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Assist agent five local desktop tools — launch apps, global file search, list/control windows, and (consent-gated) screen capture into the existing vision pipeline.

**Architecture:** Pure-logic OS backends in a new `src/desktop/` package (injectable primitives, no real GUI in tests) wrapped by five native tool classes in `src/agent_tools/desktop_tools.py`, registered through the same sites as the shipped `open_in_vscode` tool. Screen capture is gated by a new per-session `screen_access_enabled` setting.

**Tech Stack:** Python stdlib (`ctypes`, `winreg`, `os.startfile`, `struct`), `mss` (screen capture), FastAPI, pytest (`--import-mode=importlib`), vanilla JS, PyInstaller.

## Global Constraints

- All pytest runs use `--import-mode=importlib` (a global `ultralytics` package shadows `tests/` otherwise).
- Minimal deps: `ctypes` + stdlib for windows/launch; `mss` for capture. NO pywin32 unless a plan-time verification forces it. `mss` must be added to `requirements.txt` and bundle-verified.
- Every OS primitive (window enum, registry, grabber, search) is **injected** so tests never touch a real GUI/registry/index.
- No spawn of `sys.executable`; `os.startfile`/detached launches only. Any subprocess uses `CREATE_NO_WINDOW`.
- Registration parity: a guard test asserts each tool appears in every required registry, mirroring `open_in_vscode` (commit 153011c).
- Admin gating: all five tools in `NON_ADMIN_BLOCKED_TOOLS`.
- Screen capture: gated by `screen_access_enabled` (default `false`, reset to `false` at every app startup).
- Windows-only tools degrade gracefully off-Windows (return a clear error), so the suite runs on any platform via injected fakes.

---

### Task 1: `src/desktop/apps.py` — app resolution + launch

**Files:**
- Create: `src/desktop/__init__.py` (empty), `src/desktop/apps.py`
- Test: `tests/test_desktop_apps.py`

**Interfaces:**
- Produces: `resolve_app(name, *, start_menu=None, which=None, registry=None) -> dict|None` returning `{"name": str, "target": str, "kind": "shortcut|exe|uwp|path|url"}`; `launch(target: dict, startfile=os.startfile, popen=None) -> None`. `start_menu` is an injected `{app_name_lower: target_path}` map; `which` defaults to `shutil.which`; `registry` is an injected `{app_name_lower: exe_path}` map (App Paths).

- [ ] **Step 1: Write the failing test** `tests/test_desktop_apps.py`:

```python
import src.desktop.apps as apps


def test_resolves_start_menu_shortcut():
    sm = {"notepad": r"C:\ProgramData\...\Notepad.lnk"}
    got = apps.resolve_app("Notepad", start_menu=sm, which=lambda n: None, registry={})
    assert got == {"name": "Notepad", "target": sm["notepad"], "kind": "shortcut"}


def test_resolves_registry_app_paths():
    reg = {"code": r"C:\Program Files\Microsoft VS Code\Code.exe"}
    got = apps.resolve_app("code", start_menu={}, which=lambda n: None, registry=reg)
    assert got["kind"] == "exe" and got["target"] == reg["code"]


def test_resolves_from_path():
    got = apps.resolve_app("python", start_menu={}, registry={},
                           which=lambda n: r"C:\Py\python.exe" if n == "python" else None)
    assert got["kind"] == "exe" and got["target"].endswith("python.exe")


def test_existing_path_opens_as_path(tmp_path):
    f = tmp_path / "report.pdf"; f.write_text("x")
    got = apps.resolve_app(str(f), start_menu={}, which=lambda n: None, registry={})
    assert got["kind"] == "path" and got["target"] == str(f)


def test_url_is_recognized():
    got = apps.resolve_app("https://example.com", start_menu={}, which=lambda n: None, registry={})
    assert got["kind"] == "url"


def test_unknown_returns_none():
    assert apps.resolve_app("nonesuchapp", start_menu={}, which=lambda n: None, registry={}) is None


def test_launch_shortcut_uses_startfile():
    calls = []
    apps.launch({"target": "x.lnk", "kind": "shortcut"}, startfile=calls.append)
    assert calls == ["x.lnk"]


def test_launch_exe_uses_popen():
    spawned = []
    apps.launch({"target": "x.exe", "kind": "exe"},
                startfile=lambda p: (_ for _ in ()).throw(AssertionError("should popen")),
                popen=lambda argv, **k: spawned.append(argv))
    assert spawned == [["x.exe"]]
```

- [ ] **Step 2: Run, expect FAIL:** `python -m pytest tests/test_desktop_apps.py --import-mode=importlib -q` → ModuleNotFoundError.
- [ ] **Step 3: Implement** `src/desktop/apps.py`:

```python
"""Resolve an app/file/URL name to a launch target and launch it. OS calls are
injected so tests never touch the real Start Menu / registry / shell."""
import os
import re
import shutil
import subprocess

_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def resolve_app(name, *, start_menu=None, which=shutil.which, registry=None):
    raw = (name or "").strip()
    if not raw:
        return None
    if _URL_RE.match(raw):
        return {"name": raw, "target": raw, "kind": "url"}
    if os.path.exists(raw):
        return {"name": os.path.basename(raw) or raw, "target": raw, "kind": "path"}
    key = raw.lower()
    for ext in ("", ".exe", ".lnk"):
        pass  # keys are stored without extension; see below
    sm = start_menu if start_menu is not None else _default_start_menu()
    if key in sm:
        return {"name": raw, "target": sm[key], "kind": "shortcut"}
    reg = registry if registry is not None else _default_app_paths()
    if key in reg:
        return {"name": raw, "target": reg[key], "kind": "exe"}
    found = which(raw) or which(raw + ".exe")
    if found:
        return {"name": raw, "target": found, "kind": "exe"}
    return None


def launch(target, startfile=None, popen=None):
    startfile = startfile or getattr(os, "startfile", None)
    kind, tgt = target["kind"], target["target"]
    if kind in ("shortcut", "path", "url"):
        if startfile is None:
            raise RuntimeError("os.startfile unavailable on this platform")
        startfile(tgt)
        return
    # exe/uwp: detached, hidden console, never waited on.
    runner = popen or (lambda argv, **k: subprocess.Popen(
        argv, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    runner([tgt])


def _default_start_menu():
    """{app_name_lower: shortcut_path} from the system + user Start Menu."""
    out = {}
    roots = [os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"),
                          r"Microsoft\Windows\Start Menu\Programs"),
             os.path.join(os.environ.get("APPDATA", ""),
                          r"Microsoft\Windows\Start Menu\Programs")]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dp, _dns, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(".lnk"):
                    out.setdefault(os.path.splitext(fn)[0].lower(),
                                   os.path.join(dp, fn))
    return out


def _default_app_paths():
    """{exe_stem_lower: full_path} from HKLM/HKCU App Paths."""
    out = {}
    try:
        import winreg
    except ImportError:
        return out
    sub = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    for hive in (getattr(winreg, "HKEY_LOCAL_MACHINE", None),
                 getattr(winreg, "HKEY_CURRENT_USER", None)):
        if hive is None:
            continue
        try:
            with winreg.OpenKey(hive, sub) as k:
                for i in range(winreg.QueryInfoKey(k)[0]):
                    name = winreg.EnumKey(k, i)
                    try:
                        with winreg.OpenKey(k, name) as sk:
                            val = winreg.QueryValue(sk, None)
                        if val:
                            out[os.path.splitext(name)[0].lower()] = val
                    except OSError:
                        continue
        except OSError:
            continue
    return out
```

Remove the dead `for ext` loop before committing (left in the test-first draft by mistake — the key lookup uses the stem directly).

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** `feat(desktop): app resolve + launch backend`.

---

### Task 2: `src/desktop/windows.py` — list + control windows

**Files:** Create `src/desktop/windows.py`; Test `tests/test_desktop_windows.py`

**Interfaces:**
- Produces: `list_windows(user32=None, psapi=None) -> list[dict]` → `[{"id": int, "title": str, "pid": int, "process": str, "state": "normal|minimized|maximized"}]`; `control_window(window_id, action, *, user32=None) -> bool` for action ∈ `focus|minimize|maximize|restore|close`. `user32` is injected; the real one is `ctypes.windll.user32`.

- [ ] **Step 1: Failing tests** `tests/test_desktop_windows.py`:

```python
import src.desktop.windows as w


class FakeUser32:
    """Minimal user32 stand-in: two visible windows."""
    def __init__(self):
        self.actions = []
        self._titles = {11: "Untitled - Notepad", 22: "Assist"}
        self._pids = {11: 100, 22: 200}
    def EnumWindows(self, cb, _lparam):
        for hwnd in (11, 22):
            cb(hwnd, 0)
        return True
    def IsWindowVisible(self, hwnd):
        return 1
    def GetWindowTextLengthW(self, hwnd):
        return len(self._titles.get(hwnd, ""))
    def GetWindowTextW(self, hwnd, buf, n):
        buf.value = self._titles.get(hwnd, "")
        return len(buf.value)
    def GetWindowThreadProcessId(self, hwnd, pidref):
        pidref._obj.value = self._pids.get(hwnd, 0)
        return 0
    def IsIconic(self, hwnd):
        return 0
    def IsZoomed(self, hwnd):
        return 0
    def ShowWindow(self, hwnd, cmd):
        self.actions.append(("ShowWindow", hwnd, cmd)); return True
    def SetForegroundWindow(self, hwnd):
        self.actions.append(("SetForegroundWindow", hwnd)); return True
    def PostMessageW(self, hwnd, msg, wp, lp):
        self.actions.append(("PostMessageW", hwnd, msg)); return True


def test_list_windows_titles_and_pids():
    got = w.list_windows(user32=FakeUser32(), psapi=lambda pid: f"proc{pid}.exe")
    ids = {x["id"]: x for x in got}
    assert ids[11]["title"] == "Untitled - Notepad" and ids[11]["pid"] == 100
    assert ids[11]["process"] == "proc100.exe"
    assert ids[22]["title"] == "Assist"


def test_list_windows_skips_untitled():
    fu = FakeUser32(); fu._titles[22] = ""  # empty title -> excluded
    got = w.list_windows(user32=fu, psapi=lambda pid: "p.exe")
    assert [x["id"] for x in got] == [11]


def test_control_focus_restores_then_foregrounds():
    fu = FakeUser32()
    assert w.control_window(11, "focus", user32=fu) is True
    assert ("SetForegroundWindow", 11) in fu.actions


def test_control_minimize():
    fu = FakeUser32()
    w.control_window(11, "minimize", user32=fu)
    assert ("ShowWindow", 11, w.SW_MINIMIZE) in fu.actions


def test_control_close_posts_wm_close():
    fu = FakeUser32()
    w.control_window(22, "close", user32=fu)
    assert ("PostMessageW", 22, w.WM_CLOSE) in fu.actions


def test_control_unknown_action_false():
    assert w.control_window(11, "explode", user32=FakeUser32()) is False
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `src/desktop/windows.py`:

```python
"""List and control top-level windows via user32 (injected for tests)."""
import ctypes

SW_MINIMIZE, SW_MAXIMIZE, SW_RESTORE = 6, 3, 9
WM_CLOSE = 0x0010


def _real_user32():
    return ctypes.windll.user32  # pragma: no cover (Windows-only)


def _default_psapi(pid):  # pragma: no cover (Windows-only)
    import os
    try:
        import ctypes.wintypes as wt
        PROCESS_QUERY_LIMITED = 0x1000
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
        if not h:
            return ""
        buf = ctypes.create_unicode_buffer(260)
        n = wt.DWORD(260)
        ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n))
        k32.CloseHandle(h)
        return os.path.basename(buf.value)
    except Exception:
        return ""


def list_windows(user32=None, psapi=None):
    u = user32 or _real_user32()
    psapi = psapi or _default_psapi
    out = []

    def _cb(hwnd, _lparam):
        try:
            if not u.IsWindowVisible(hwnd):
                return True
            if u.GetWindowTextLengthW(hwnd) == 0:
                return True
            buf = ctypes.create_unicode_buffer(512)
            u.GetWindowTextW(hwnd, buf, 512)
            pid = ctypes.c_ulong(0)
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            state = ("minimized" if u.IsIconic(hwnd)
                     else "maximized" if u.IsZoomed(hwnd) else "normal")
            out.append({"id": int(hwnd), "title": buf.value, "pid": int(pid.value),
                        "process": psapi(int(pid.value)), "state": state})
        except Exception:
            pass
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p) \
        if hasattr(ctypes, "WINFUNCTYPE") else (lambda *a: _cb)
    try:
        u.EnumWindows(WNDENUMPROC(_cb) if callable(WNDENUMPROC) else _cb, 0)
    except TypeError:
        u.EnumWindows(_cb, 0)  # fakes accept the plain callback
    return out


def control_window(window_id, action, *, user32=None):
    u = user32 or _real_user32()
    if action == "focus":
        u.ShowWindow(window_id, SW_RESTORE)
        u.SetForegroundWindow(window_id)
        return True
    if action == "minimize":
        u.ShowWindow(window_id, SW_MINIMIZE); return True
    if action == "maximize":
        u.ShowWindow(window_id, SW_MAXIMIZE); return True
    if action == "restore":
        u.ShowWindow(window_id, SW_RESTORE); return True
    if action == "close":
        u.PostMessageW(window_id, WM_CLOSE, 0, 0); return True
    return False
```

Note: the `WNDENUMPROC` shim lets the real ctypes path build a proper callback while the fake `EnumWindows` receives the plain Python callable. Verify tests pass with the fake; the real path is exercised in Task 8 live verification.

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** `feat(desktop): window list + control backend`.

---

### Task 3: `src/desktop/filesearch.py` — index-first, walk fallback

**Files:** Create `src/desktop/filesearch.py`; Test `tests/test_desktop_filesearch.py`

**Interfaces:**
- Produces: `search(query, *, roots, ext=None, all_drives=False, max_results=200, searcher=None, walker=os.walk, is_sensitive=None) -> list[dict]` → `[{"path", "size", "modified"}]`. `searcher(query, ext, max_results) -> list[str]|None` is the injected Windows-index query (None/raise → fallback to `walker` over `roots`). `is_sensitive` defaults to `src.tool_execution._is_sensitive_path`.

- [ ] **Step 1: Failing tests** `tests/test_desktop_filesearch.py`:

```python
import os
import src.desktop.filesearch as fs


def test_uses_index_when_available(tmp_path):
    a = tmp_path / "report.txt"; a.write_text("x")
    got = fs.search("report", roots=[str(tmp_path)],
                    searcher=lambda q, ext, n: [str(a)],
                    is_sensitive=lambda p: False)
    assert [h["path"] for h in got] == [str(a)]
    assert got[0]["size"] == 1


def test_falls_back_to_walk_when_searcher_none(tmp_path):
    (tmp_path / "notes.md").write_text("y")
    (tmp_path / "other.txt").write_text("z")
    got = fs.search("notes", roots=[str(tmp_path)], searcher=None,
                    is_sensitive=lambda p: False)
    assert [os.path.basename(h["path"]) for h in got] == ["notes.md"]


def test_falls_back_when_searcher_raises(tmp_path):
    (tmp_path / "keep.log").write_text("y")
    def boom(q, ext, n): raise OSError("index off")
    got = fs.search("keep", roots=[str(tmp_path)], searcher=boom,
                    is_sensitive=lambda p: False)
    assert len(got) == 1 and got[0]["path"].endswith("keep.log")


def test_ext_filter_in_walk(tmp_path):
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "a.txt").write_text("1")
    got = fs.search("a", roots=[str(tmp_path)], ext="py", searcher=None,
                    is_sensitive=lambda p: False)
    assert [os.path.basename(h["path"]) for h in got] == ["a.py"]


def test_sensitive_paths_filtered(tmp_path):
    (tmp_path / "id_rsa").write_text("secret")
    got = fs.search("id_rsa", roots=[str(tmp_path)], searcher=None,
                    is_sensitive=lambda p: p.endswith("id_rsa"))
    assert got == []


def test_max_results_caps_walk(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    got = fs.search("f", roots=[str(tmp_path)], max_results=3, searcher=None,
                    is_sensitive=lambda p: False)
    assert len(got) == 3
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `src/desktop/filesearch.py`:

```python
"""Global file search: Windows Search index first, bounded os.walk fallback.
Metadata only — reading a hit still goes through the confined read tool."""
import fnmatch
import os


def _meta(path):
    try:
        st = os.stat(path)
        return {"path": path, "size": st.st_size, "modified": int(st.st_mtime)}
    except OSError:
        return None


def search(query, *, roots, ext=None, all_drives=False, max_results=200,
           searcher=None, walker=os.walk, is_sensitive=None):
    if is_sensitive is None:
        from src.tool_execution import _is_sensitive_path as is_sensitive
    q = (query or "").strip()
    hits = []

    # Primary: Windows Search index.
    if searcher is None:
        searcher = _default_searcher
    try:
        paths = searcher(q, ext, max_results)
    except Exception:
        paths = None
    if paths is not None:
        for p in paths:
            if is_sensitive(p):
                continue
            m = _meta(p)
            if m:
                hits.append(m)
            if len(hits) >= max_results:
                break
        return hits

    # Fallback: bounded walk.
    ql = q.lower()
    extl = ("." + ext.lower().lstrip(".")) if ext else None
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, _dns, fns in walker(root):
            for fn in fns:
                if ql and ql not in fn.lower() and not fnmatch.fnmatch(fn.lower(), ql):
                    continue
                if extl and not fn.lower().endswith(extl):
                    continue
                full = os.path.join(dp, fn)
                if is_sensitive(full):
                    continue
                m = _meta(full)
                if m:
                    hits.append(m)
                if len(hits) >= max_results:
                    return hits
    return hits


def _default_searcher(query, ext, max_results):  # pragma: no cover (Windows-only)
    """Query the Windows Search index via ADO. Returns None when unavailable
    so the caller falls back to walking. PLAN-TIME: pin adodbapi vs a minimal
    comtypes call and bundle-verify; until then this returns None (walk-only)."""
    return None
```

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** `feat(desktop): file search (index-first, walk fallback)`.

---

### Task 4: `src/desktop/capture.py` — screen capture via mss

**Files:** Create `src/desktop/capture.py`; Modify `requirements.txt` (add `mss`); Test `tests/test_desktop_capture.py`

**Interfaces:**
- Produces: `capture_png(target="full", *, grabber=None, window_rect=None) -> bytes` (PNG bytes). `grabber` is an injected callable `region_dict -> RGB-bytes-like` (real one wraps `mss`); `window_rect(window_id) -> (l,t,w,h)` resolves a window target.

- [ ] **Step 1: Add `mss` to `requirements.txt`** (a line `mss`), so the packaged build bundles it.
- [ ] **Step 2: Failing tests** `tests/test_desktop_capture.py`:

```python
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
```

- [ ] **Step 3: Run, expect FAIL.**
- [ ] **Step 4: Implement** `src/desktop/capture.py`:

```python
"""Screen capture → PNG bytes. Real capture uses mss; the grabber is injected
so tests never touch a display."""
import io
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
    rect = ctypes.wintypes.RECT() if hasattr(ctypes, "wintypes") else None
    ctypes.windll.user32.GetWindowRect(window_id, ctypes.byref(rect))
    return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
```

- [ ] **Step 5: Run, expect PASS.** **Step 6: Commit** `feat(desktop): screen capture → PNG (mss, injected grabber)`.

---

### Task 5: Screen-access session toggle (setting + route + reset)

**Files:**
- Modify: `src/settings.py` (add `screen_access_enabled` to `DEFAULT_SETTINGS`)
- Modify: `app.py` (reset to false in the startup lifespan, next to the endpoint prune)
- Test: `tests/test_screen_access_toggle.py`

**Interfaces:**
- Produces: setting key `screen_access_enabled` (bool, default `false`); a startup reset so it is always false at boot. Read via `get_setting("screen_access_enabled", False)`. The existing admin `POST /api/auth/settings` already accepts any `DEFAULT_SETTINGS` key, so no new route.

- [ ] **Step 1: Failing tests** `tests/test_screen_access_toggle.py`:

```python
import src.settings as settings


def test_default_is_false():
    assert settings.DEFAULT_SETTINGS.get("screen_access_enabled") is False


def test_reset_helper_forces_false(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(p))
    settings.save_settings({"screen_access_enabled": True, "keep": 1})
    settings.reset_screen_access()
    saved = settings.load_settings()
    assert saved["screen_access_enabled"] is False
    assert saved["keep"] == 1  # other settings preserved
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** In `src/settings.py` `DEFAULT_SETTINGS`, add near `image_gen_enabled`:

```python
    "screen_access_enabled": False,
```

Then add a reset helper:

```python
def reset_screen_access():
    """Force screen access off. Called at startup so capture is never silently
    available across restarts (per the Desktop Control consent model)."""
    try:
        s = load_settings()
        if s.get("screen_access_enabled"):
            s["screen_access_enabled"] = False
            save_settings(s)
    except Exception:
        pass
```

In `app.py`, in the startup lifespan next to the `prune_serve_endpoints()` call:

```python
    try:
        from src.settings import reset_screen_access
        reset_screen_access()
    except Exception as _e:
        logger.debug(f"screen-access reset skipped: {_e}")
```

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** `feat(desktop): screen_access_enabled session toggle + startup reset`.

---

### Task 6: `src/agent_tools/desktop_tools.py` — five tool classes

**Files:** Create `src/agent_tools/desktop_tools.py`; Test `tests/test_desktop_tools.py`

**Interfaces:**
- Consumes: Tasks 1-5 backends; `get_setting`.
- Produces: `LaunchAppTool`, `FindFilesTool`, `ListWindowsTool`, `ControlWindowTool`, `CaptureScreenTool`, each `async execute(content, ctx) -> dict` with `{output|error, exit_code}`. `CaptureScreenTool` returns an `image_url` block dict on success: `{"output": "<n> screenshot captured", "image_url": "data:image/png;base64,...", "exit_code": 0}`.

- [ ] **Step 1: Failing tests** `tests/test_desktop_tools.py`:

```python
import asyncio
import base64
import json

import src.agent_tools.desktop_tools as dt


def _run(tool, content):
    return asyncio.run(tool.execute(content, {}))


def test_launch_app_resolves_and_launches(monkeypatch):
    launched = {}
    monkeypatch.setattr(dt, "resolve_app", lambda n: {"name": n, "target": "np.exe", "kind": "exe"})
    monkeypatch.setattr(dt, "launch", lambda t: launched.setdefault("t", t))
    res = _run(dt.LaunchAppTool(), json.dumps({"name": "Notepad"}))
    assert res["exit_code"] == 0 and launched["t"]["target"] == "np.exe"


def test_launch_app_unknown(monkeypatch):
    monkeypatch.setattr(dt, "resolve_app", lambda n: None)
    res = _run(dt.LaunchAppTool(), json.dumps({"name": "nope"}))
    assert res["exit_code"] == 1 and "could not find" in res["error"].lower()


def test_find_files_formats_hits(monkeypatch):
    monkeypatch.setattr(dt, "search", lambda q, **k: [{"path": "C:/a.py", "size": 3, "modified": 0}])
    res = _run(dt.FindFilesTool(), json.dumps({"query": "a"}))
    assert "C:/a.py" in res["output"] and res["exit_code"] == 0


def test_list_windows_formats(monkeypatch):
    monkeypatch.setattr(dt, "list_windows", lambda: [{"id": 11, "title": "Notepad", "pid": 1, "process": "notepad.exe", "state": "normal"}])
    res = _run(dt.ListWindowsTool(), "{}")
    assert "Notepad" in res["output"] and "11" in res["output"]


def test_control_window_dispatches(monkeypatch):
    seen = {}
    monkeypatch.setattr(dt, "control_window", lambda wid, action: seen.update(id=wid, a=action) or True)
    res = _run(dt.ControlWindowTool(), json.dumps({"id": 11, "action": "focus"}))
    assert res["exit_code"] == 0 and seen == {"id": 11, "a": "focus"}


def test_capture_refuses_when_toggle_off(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: False)
    res = _run(dt.CaptureScreenTool(), json.dumps({"target": "full"}))
    assert res["exit_code"] == 1 and "screen access is off" in res["error"].lower()


def test_capture_emits_data_uri_when_on(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "screen_access_enabled" else d)
    monkeypatch.setattr(dt, "_vision_ready", lambda: True)
    monkeypatch.setattr(dt, "capture_png", lambda target, **k: b"\x89PNG\r\n\x1a\nDATA")
    res = _run(dt.CaptureScreenTool(), json.dumps({"target": "full"}))
    assert res["exit_code"] == 0
    assert res["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(res["image_url"].split(",", 1)[1]).startswith(b"\x89PNG")


def test_capture_needs_vision_model(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "screen_access_enabled" else d)
    monkeypatch.setattr(dt, "_vision_ready", lambda: False)
    res = _run(dt.CaptureScreenTool(), json.dumps({"target": "full"}))
    assert res["exit_code"] == 1 and "vision model" in res["error"].lower()
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `src/agent_tools/desktop_tools.py`:

```python
"""Native desktop-control tools. Thin wrappers over src/desktop backends;
gates and formatting only. Screen capture is consent-gated."""
import base64
import json
import logging

from src.settings import get_setting
from src.desktop.apps import resolve_app, launch
from src.desktop.filesearch import search
from src.desktop.windows import list_windows, control_window
from src.desktop.capture import capture_png

logger = logging.getLogger(__name__)


def _args(content):
    try:
        return json.loads(content) if content.strip().startswith("{") else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _vision_ready():
    return bool(get_setting("vision_enabled", True) and get_setting("vision_model", ""))


class LaunchAppTool:
    async def execute(self, content, ctx):
        name = (_args(content).get("name") or content).strip()
        if not name or name.startswith("{"):
            return {"error": "launch_app: name required", "exit_code": 1}
        target = resolve_app(name)
        if not target:
            return {"error": f"launch_app: could not find an app named {name!r}", "exit_code": 1}
        try:
            launch(target)
        except Exception as e:
            return {"error": f"launch_app: {e}", "exit_code": 1}
        logger.info("launch_app: %s -> %s", name, target["target"])
        return {"output": f"Launched {target['name']} ({target['target']})", "exit_code": 0}


class FindFilesTool:
    async def execute(self, content, ctx):
        a = _args(content)
        query = (a.get("query") or "").strip()
        if not query:
            return {"error": "find_files: query required", "exit_code": 1}
        from src.constants import DATA_DIR
        import os
        roots = [os.path.expanduser("~")]
        extra = get_setting("tool_path_extra_roots") or []
        roots += [str(r) for r in extra if r]
        if a.get("all_drives"):
            roots += [f"{d}:\\" for d in "CDEFG" if os.path.isdir(f"{d}:\\")]
        hits = search(query, roots=roots, ext=a.get("ext"),
                      all_drives=bool(a.get("all_drives")),
                      max_results=int(a.get("max_results") or 200))
        if not hits:
            return {"output": f"No files matching {query!r}", "exit_code": 0}
        lines = [f"{h['path']}  ({h['size']} bytes)" for h in hits]
        return {"output": f"{len(hits)} match(es):\n" + "\n".join(lines), "exit_code": 0}


class ListWindowsTool:
    async def execute(self, content, ctx):
        wins = list_windows()
        if not wins:
            return {"output": "No visible windows", "exit_code": 0}
        lines = [f"[{w['id']}] {w['title']} — {w['process']} (pid {w['pid']}, {w['state']})"
                 for w in wins]
        return {"output": "\n".join(lines), "exit_code": 0}


class ControlWindowTool:
    async def execute(self, content, ctx):
        a = _args(content)
        action = (a.get("action") or "").strip().lower()
        wid = a.get("id")
        if wid is None or action not in ("focus", "minimize", "maximize", "restore", "close"):
            return {"error": "control_window: id and action (focus|minimize|maximize|restore|close) required", "exit_code": 1}
        try:
            ok = control_window(int(wid), action)
        except Exception as e:
            return {"error": f"control_window: {e}", "exit_code": 1}
        logger.info("control_window: %s %s", wid, action)
        return {"output": f"{action} window {wid}" if ok else f"control_window: {action} failed",
                "exit_code": 0 if ok else 1}


class CaptureScreenTool:
    async def execute(self, content, ctx):
        if not get_setting("screen_access_enabled", False):
            return {"error": "Screen access is off. Ask the user to enable 'Allow screen "
                             "access' in the sidebar before capturing the screen.", "exit_code": 1}
        if not _vision_ready():
            return {"error": "capture_screen: no vision model configured (Settings → vision_model).",
                    "exit_code": 1}
        target = (_args(content).get("target") or "full").strip()
        try:
            png = capture_png(target)
        except Exception as e:
            return {"error": f"capture_screen: {e}", "exit_code": 1}
        logger.info("capture_screen: %s", target)
        uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        return {"output": f"{target} screenshot captured", "image_url": uri, "exit_code": 0}
```

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** `feat(desktop): five desktop-control tool classes`.

---

### Task 7: Register the five tools everywhere

**Files:** Modify `src/agent_tools/__init__.py`, `src/agent_loop.py`, `src/tool_schemas.py`, `src/tool_index.py`, `src/tool_execution.py`, `src/tool_security.py`; Test `tests/test_desktop_registration.py`

**Interfaces:** Consumes Task 6 tool classes. Produces full registration so the tools are callable and prompt-visible, mirroring `open_in_vscode` (commit 153011c).

- [ ] **Step 1: Failing guard test** `tests/test_desktop_registration.py`:

```python
DESKTOP = ["launch_app", "find_files", "list_windows", "control_window", "capture_screen"]


def test_all_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    for t in DESKTOP:
        assert t in TOOL_HANDLERS, f"{t} not in TOOL_HANDLERS"
        assert t in TOOL_TAGS, f"{t} not in TOOL_TAGS"
        assert t in names, f"{t} not in FUNCTION_TOOL_SCHEMAS"
        assert t in TOOL_SECTIONS, f"{t} not in TOOL_SECTIONS"
        assert t in _DOMAIN_TOOL_MAP["desktop"], f"{t} not in desktop domain"
        assert t in BUILTIN_TOOL_DESCRIPTIONS, f"{t} not in BUILTIN_TOOL_DESCRIPTIONS"
        assert t in NON_ADMIN_BLOCKED_TOOLS, f"{t} not admin-gated"
```

(Confirm the actual export name of the tool-index dict — `tool_index.py` line ~74 defines it; the plan uses `TOOL_DESCRIPTIONS`. If the real name differs, use that in the test and imports.)

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement each site.**

`src/agent_tools/__init__.py` — import + handlers + tags:
```python
from .desktop_tools import (LaunchAppTool, FindFilesTool, ListWindowsTool,
                            ControlWindowTool, CaptureScreenTool)
# in TOOL_HANDLERS:
    "launch_app": LaunchAppTool().execute,
    "find_files": FindFilesTool().execute,
    "list_windows": ListWindowsTool().execute,
    "control_window": ControlWindowTool().execute,
    "capture_screen": CaptureScreenTool().execute,
# in TOOL_TAGS: add the five names
```

`src/agent_loop.py` — add a `desktop` domain to `_DOMAIN_TOOL_MAP` and five `TOOL_SECTIONS` prompt blocks:
```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen"},
```
```python
    "launch_app": '```launch_app\n{"name": "<app name, file, or URL>"}\n```\nLaunch an installed app by name, or open a file/URL with its default program.',
    "find_files": '```find_files\n{"query": "<name or glob>", "ext": "pdf", "all_drives": false}\n```\nSearch the PC for files by name. Returns paths + size. Read the contents with read_file.',
    "list_windows": '```list_windows\n```\nList open windows: id, title, process, state. Use before control_window.',
    "control_window": '```control_window\n{"id": <window id>, "action": "focus|minimize|maximize|restore|close"}\n```\nFocus or change a window. Confirm with the user before close (unsaved work).',
    "capture_screen": '```capture_screen\n{"target": "full|monitor:1|window:<id>"}\n```\nCapture the screen so you can see it. Requires the user to enable screen access. Use to read on-screen content, forms, errors.',
```

`src/tool_schemas.py` — five function schemas + content marshalling in the `_tool_content_for` branch (`launch_app`/`find_files`/`control_window` → `json.dumps(args)`; `list_windows`/`capture_screen` → `json.dumps(args)` too since they take an optional target):
```python
    elif tool_type in ("grep", "glob", "ls", "open_in_vscode",
                       "launch_app", "find_files", "list_windows",
                       "control_window", "capture_screen"):
        content = json.dumps(args) if args else "{}"
```
Add schema objects mirroring `open_in_vscode`'s (name, description from Task 6 docstrings, `parameters` per the tool args).

`src/tool_index.py` — add one-line descriptions for each of the five to the `BUILTIN_TOOL_DESCRIPTIONS` dict (copy the `TOOL_SECTIONS` summaries).

`src/tool_execution.py` — extend the direct-dispatch branch:
```python
    elif tool in ("grep", "glob", "ls", "get_workspace", "open_in_vscode",
                 "launch_app", "find_files", "list_windows",
                 "control_window", "capture_screen"):
```

`src/tool_security.py` — add the five names to `NON_ADMIN_BLOCKED_TOOLS`. Add `find_files`, `list_windows`, `capture_screen` to `PLAN_MODE_READONLY_TOOLS` (read-only); leave `launch_app`/`control_window` out (they act).

- [ ] **Step 4: Run** `python -m pytest tests/test_desktop_registration.py --import-mode=importlib -q`, expect PASS. Also `python -c "import src.agent_tools, src.agent_loop, src.tool_schemas"` for import health.
- [ ] **Step 5: Commit** `feat(desktop): register the five desktop tools across the tool system`.

---

### Task 8: Screen-access sidebar toggle (UI) + Manual

**Files:** Modify `static/index.html` (sidebar toggle + indicator), `static/js/settings.js` (or a small `static/js/screenAccess.js`) to POST the setting; Modify the Help Manual (`static/index.html` Manual card); Test `tests/test_screen_access_ui.py`

**Interfaces:** Consumes the `screen_access_enabled` setting + `POST /api/auth/settings`. Produces a visible toggle + "screen on" indicator.

- [ ] **Step 1: Failing UI guard** `tests/test_screen_access_ui.py`:

```python
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def _read(p): return (ROOT / p).read_text(encoding="utf-8")

def test_sidebar_has_screen_toggle():
    html = _read("static/index.html")
    assert 'id="screen-access-toggle"' in html
    assert 'id="screen-access-indicator"' in html

def test_js_posts_screen_access_setting():
    js = _read("static/js/screenAccess.js")
    assert "screen_access_enabled" in js
    assert "/api/auth/settings" in js

def test_manual_documents_screen_access():
    html = _read("static/index.html")
    assert "screen access" in html.lower()
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** In `static/index.html`, add a sidebar item (near the Help entry) with `id="screen-access-toggle"` (a checkbox/switch) and an `id="screen-access-indicator"` dot shown only when on. Create `static/js/screenAccess.js`: on load, GET `/api/auth/settings` → reflect `screen_access_enabled`; on toggle, POST `{screen_access_enabled: <bool>}` to `/api/auth/settings` and show/hide the indicator. Add a `<script src="/static/js/screenAccess.js"></script>` include. Add a Manual `<details>` block: "Screen access & desktop control" explaining the five tools + the toggle.
- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** `feat(desktop): screen-access sidebar toggle + Manual section`.

---

### Task 9: Package, bundle-verify, live-verify

**Files:** none (build + manual verification). Optionally modify `Assist.spec` if `mss` needs a hidden import.

- [ ] **Step 1: Run the full affected suite:** `python -m pytest tests/test_desktop_apps.py tests/test_desktop_windows.py tests/test_desktop_filesearch.py tests/test_desktop_capture.py tests/test_screen_access_toggle.py tests/test_desktop_tools.py tests/test_desktop_registration.py tests/test_screen_access_ui.py --import-mode=importlib -q` → all green.
- [ ] **Step 2: Build:** `.\build-installer.ps1 -Fast`. If PyInstaller misses `mss`, add `hiddenimports=['mss', 'mss.windows']` (or the `datas`/hook) to `Assist.spec` and rebuild. Confirm the compile succeeds.
- [ ] **Step 3: Boot-verify** the packaged exe against an isolated `ODYSSEUS_DATA_DIR` + `ODYSSEUS_INTERNAL_TOKEN` (per the established pattern): confirm the app starts and `GET /api/auth/settings` shows `screen_access_enabled: false` at boot.
- [ ] **Step 4: User manual test (Agent mode, admin):** ask the agent to "launch Notepad" (opens), "find files named <known file>" (lists it), "list my open windows" then "focus the Assist window" (focuses); enable the sidebar screen toggle and ask "what's on my screen right now?" — confirm the vision model describes the captured screenshot. Then disable the toggle and confirm capture is refused.
- [ ] **Step 5: Commit** the built installer: `git add -f installer/Output/Assist-Setup.exe && git commit -m "build: Assist-Setup.exe with Desktop Control tools"`.

## Self-Review

- **Spec coverage:** launch (T1), file search index+walk (T3), window list/control (T2), capture→vision (T4), consent toggle + reset (T5), tool wrappers with gating (T6), full registration + admin gating (T7), UI toggle/indicator + Manual (T8), package + live-verify (T9). All five tools + the consent model + the dependency strategy are covered.
- **Placeholders:** none. The two genuinely-external unknowns — the Windows Search index query library and `mss` PyInstaller bundling — are explicit, isolated behind `_default_searcher` (returns None → walk fallback, so the feature ships either way) and a T9 build check with a concrete `hiddenimports` fix. The `_default_start_menu`/`_default_app_paths`/`_default_grabber`/`_default_window_rect`/`_default_psapi` real-OS paths are `# pragma: no cover` and exercised only in T9 live verification, by design (tests use injected fakes).
- **Type consistency:** `resolve_app`→dict with `kind`/`target` consumed by `launch` (T1) and `LaunchAppTool` (T6); `search(...)→[{path,size,modified}]` consumed by `FindFilesTool`; `list_windows()→[{id,title,pid,process,state}]` consumed by `ListWindowsTool`; `control_window(id, action)` consumed by `ControlWindowTool`; `capture_png(target)→bytes` consumed by `CaptureScreenTool` which emits the `image_url` data-URI block the vision pipeline already accepts. Registration names identical across T6/T7 and the guard test.
- **Symbols verified against current code (2026-07-09):** the tool-index dict is `BUILTIN_TOOL_DESCRIPTIONS` (`src/tool_index.py:69`); the domain map is `_DOMAIN_TOOL_MAP` (`src/agent_loop.py:281`, a dict of `domain -> set[str]` with no `desktop` key yet — T7 adds it). Remaining plan-time verification: `mss` PyInstaller bundling (T9) and the Windows Search index library (isolated behind `_default_searcher`, walk fallback ships regardless).
