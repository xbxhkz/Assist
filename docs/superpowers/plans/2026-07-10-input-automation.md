# Input Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent *hands* on Windows — mouse/keyboard via SendInput plus UI-Automation element targeting — behind an explicit session consent toggle, as the actuator layer the future AI Operator will drive.

**Architecture:** Two `src/desktop/` backends with injectable primitives (tests never touch a real GUI/COM): `inputraw.py` (ctypes `SendInput` actuators) and `uia.py` (UI Automation over a duck-typed element surface; all `comtypes` complexity isolated in one real adapter). Five thin tool wrappers (`src/agent_tools/input_tools.py`) with the standard `execute` shape, gated by two session settings and registered across the tool system's 8 sites.

**Tech Stack:** Python, `ctypes` (zero-dep raw input), `comtypes` (UI Automation), pytest (`--import-mode=importlib`), the existing tool-registration + settings + sidebar patterns.

## Global Constraints

- All pytest runs use `--import-mode=importlib` (a global `ultralytics` package shadows `tests/`).
- Windows-only capability, but all UNIT tests must pass on the dev machine without a real GUI/COM — every backend takes an injectable primitive; the real ctypes/comtypes boundary is exercised only at packaging live-verify.
- Consent model = **session toggle**, mirroring the shipped screen-access toggle exactly: setting defaults `false`, resets to `false` on every launch, sidebar toggle + persistent "ON" indicator, no per-action dialog.
- **Read/act gating split:** the reading tool `list_ui_elements` is gated by the EXISTING `screen_access_enabled` setting; the four acting tools (`click_element`, `set_element_text`, `mouse`, `keyboard`) are gated by the NEW `input_control_enabled` setting.
- Admin-gated: all five tools go in `NON_ADMIN_BLOCKED_TOOLS`. Plan mode: only `list_ui_elements` goes in `PLAN_MODE_READONLY_TOOLS`; the four acting tools stay OUT (mutators, blocked in plan mode).
- Every acting call logs one INFO line to `app.log` (tool + target/coords).
- Follow the existing patterns verbatim — the shipped Desktop Control tools (`launch_app`/`find_files`/`list_windows`/`control_window`/`capture_screen`) are the template for every registration site.
- Frontend: plain-IIFE script mirroring `static/js/screenAccess.js`; listeners via `addEventListener`; no inline handlers; no `type="module"`.

---

### Task 1: `input_control_enabled` setting + startup reset

**Files:**
- Modify: `src/settings.py` (`DEFAULT_SETTINGS`; add `reset_input_control`)
- Modify: `app.py` (startup reset block near line 981)
- Test: `tests/test_input_control_setting.py`

**Interfaces:**
- Produces: setting key `input_control_enabled` (bool, default `False`); `reset_input_control()` forces it off.

- [ ] **Step 1 — failing test** `tests/test_input_control_setting.py`:

```python
import src.settings as settings


def test_default_input_control_is_false():
    assert settings.DEFAULT_SETTINGS.get("input_control_enabled") is False


def test_reset_input_control_forces_off(monkeypatch):
    store = {"input_control_enabled": True}
    monkeypatch.setattr(settings, "load_settings", lambda: dict(store))
    saved = {}
    monkeypatch.setattr(settings, "save_settings", lambda s: saved.update(s))
    settings.reset_input_control()
    assert saved.get("input_control_enabled") is False
```

- [ ] **Step 2 — run, expect FAIL:** `python -m pytest tests/test_input_control_setting.py --import-mode=importlib -q`
- [ ] **Step 3 — implement.** In `src/settings.py` `DEFAULT_SETTINGS`, directly after the `"screen_access_enabled": False,` line (currently line 41) add:

```python
    "input_control_enabled": False,
```

Then, directly after the existing `reset_screen_access()` function (near line 353), add its parallel:

```python
def reset_input_control():
    """Force input control off. Called at startup so mouse/keyboard automation
    is never silently available across restarts (mirrors reset_screen_access)."""
    try:
        s = load_settings()
        if s.get("input_control_enabled"):
            s["input_control_enabled"] = False
            save_settings(s)
    except Exception:
        pass
```

In `app.py`, the existing startup block (lines 980-984) is:

```python
    try:
        from src.settings import reset_screen_access
        reset_screen_access()
    except Exception as _e:
        logger.debug(f"screen-access reset skipped: {_e}")
```

Replace it with:

```python
    try:
        from src.settings import reset_screen_access, reset_input_control
        reset_screen_access()
        reset_input_control()
    except Exception as _e:
        logger.debug(f"screen-access/input-control reset skipped: {_e}")
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(input): input_control_enabled setting + startup reset`.

---

### Task 2: `inputraw.py` — SendInput actuators

**Files:**
- Create: `src/desktop/inputraw.py`
- Test: `tests/test_inputraw.py`

**Interfaces:**
- Produces: `move(x,y,*,emit=...)`, `click(x,y,button="left",double=False,*,emit=...)`, `drag(x1,y1,x2,y2,button="left",*,emit=...)`, `scroll(amount,*,emit=...)`, `type_text(s,*,emit=...)`, `press_keys(keys,*,emit=...)`. Each builds a list of event descriptors and passes it to `emit` (default emitter calls `SendInput`). Event descriptor tuples: mouse `("mouse", flags:int, dx:int, dy:int, data:int)`, key `("key", vk:int, scan:int, flags:int)`. Module functions `_screen_size()->(w,h)` and `VK` (key-name→virtual-key dict) are monkeypatchable.

- [ ] **Step 1 — failing test** `tests/test_inputraw.py`:

```python
import src.desktop.inputraw as ir


def _rec():
    events = []
    return events, (lambda evs: events.extend(evs))


def test_type_text_emits_unicode_down_up_per_char():
    events, emit = _rec()
    ir.type_text("Hi", emit=emit)
    # 2 chars -> 4 key events (down+up each), all KEYEVENTF_UNICODE, vk=0, scan=codepoint
    assert [e[0] for e in events] == ["key", "key", "key", "key"]
    assert events[0] == ("key", 0, ord("H"), ir.KEYEVENTF_UNICODE)
    assert events[1] == ("key", 0, ord("H"), ir.KEYEVENTF_UNICODE | ir.KEYEVENTF_KEYUP)


def test_click_emits_move_then_down_up(monkeypatch):
    monkeypatch.setattr(ir, "_screen_size", lambda: (1921, 1081))  # -> divisor 1920/1080
    events, emit = _rec()
    ir.click(960, 540, emit=emit)
    assert events[0] == ("mouse", ir.MOUSEEVENTF_MOVE | ir.MOUSEEVENTF_ABSOLUTE, 32767, 32767, 0)
    assert events[1] == ("mouse", ir.MOUSEEVENTF_LEFTDOWN, 0, 0, 0)
    assert events[2] == ("mouse", ir.MOUSEEVENTF_LEFTUP, 0, 0, 0)


def test_double_click_emits_two_down_up_pairs(monkeypatch):
    monkeypatch.setattr(ir, "_screen_size", lambda: (1001, 1001))
    events, emit = _rec()
    ir.click(0, 0, double=True, emit=emit)
    # move + (down,up) + (down,up)
    assert len(events) == 5
    assert [e[1] for e in events[1:]] == [ir.MOUSEEVENTF_LEFTDOWN, ir.MOUSEEVENTF_LEFTUP,
                                          ir.MOUSEEVENTF_LEFTDOWN, ir.MOUSEEVENTF_LEFTUP]


def test_right_click_uses_right_flags(monkeypatch):
    monkeypatch.setattr(ir, "_screen_size", lambda: (101, 101))
    events, emit = _rec()
    ir.click(0, 0, button="right", emit=emit)
    assert events[1][1] == ir.MOUSEEVENTF_RIGHTDOWN and events[2][1] == ir.MOUSEEVENTF_RIGHTUP


def test_drag_emits_down_move_up(monkeypatch):
    monkeypatch.setattr(ir, "_screen_size", lambda: (1001, 1001))
    events, emit = _rec()
    ir.drag(0, 0, 500, 500, emit=emit)
    flags = [e[1] for e in events]
    assert ir.MOUSEEVENTF_LEFTDOWN in flags and ir.MOUSEEVENTF_LEFTUP in flags
    assert flags.index(ir.MOUSEEVENTF_LEFTDOWN) < flags.index(ir.MOUSEEVENTF_LEFTUP)
    # a MOVE happens between down and up
    move_idxs = [i for i, f in enumerate(flags) if f & ir.MOUSEEVENTF_MOVE]
    assert any(flags.index(ir.MOUSEEVENTF_LEFTDOWN) < i < flags.index(ir.MOUSEEVENTF_LEFTUP)
               for i in move_idxs)


def test_scroll_positive_is_wheel_up():
    events, emit = _rec()
    ir.scroll(3, emit=emit)
    assert events[0][0] == "mouse" and events[0][1] == ir.MOUSEEVENTF_WHEEL
    assert events[0][4] == 3 * ir.WHEEL_DELTA  # data carries the signed delta


def test_press_keys_presses_modifiers_then_releases_in_reverse():
    events, emit = _rec()
    ir.press_keys(["ctrl", "s"], emit=emit)
    # down ctrl, down s, up s, up ctrl
    vks = [(e[1], e[3] & ir.KEYEVENTF_KEYUP) for e in events]  # (vk, is_up)
    assert vks == [(ir.VK["ctrl"], 0), (ir.VK["s"], 0),
                   (ir.VK["s"], ir.KEYEVENTF_KEYUP), (ir.VK["ctrl"], ir.KEYEVENTF_KEYUP)]


def test_press_keys_unknown_key_raises():
    import pytest
    with pytest.raises(ValueError):
        ir.press_keys(["ctrl", "nope-key"], emit=lambda e: None)
```

- [ ] **Step 2 — run, expect FAIL** (module missing).
- [ ] **Step 3 — implement** `src/desktop/inputraw.py`:

```python
"""Raw mouse/keyboard actuators over Win32 SendInput (zero third-party dep).

Public functions build a list of event descriptors and hand them to an
injectable `emit`. The default emitter converts them to ctypes INPUT structs
and calls SendInput; tests inject a recorder and assert the descriptors, so
the actuator logic is verified without moving a real cursor.
"""
import ctypes

# --- Win32 constants -------------------------------------------------------
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
WHEEL_DELTA = 120
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

_BUTTON = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
}

# Key-name -> Virtual-Key code. Extend as needed; unknown names raise.
VK = {
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}
VK.update({chr(c): c for c in range(ord("A"), ord("Z") + 1)})          # 'A'..'Z'
VK.update({chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)})  # 'a'..'z' -> same VK
VK.update({str(d): 0x30 + d for d in range(10)})                       # '0'..'9'
VK.update({f"f{i}": 0x70 + (i - 1) for i in range(1, 13)})             # f1..f12


def _screen_size():
    """Virtual-desktop size (spans all monitors). Monkeypatched in tests."""
    u = ctypes.windll.user32
    return (u.GetSystemMetrics(SM_CXVIRTUALSCREEN), u.GetSystemMetrics(SM_CYVIRTUALSCREEN))


def _norm(x, y):
    w, h = _screen_size()
    nx = int(x * 65535 / max(1, w - 1))
    ny = int(y * 65535 / max(1, h - 1))
    return nx, ny


# --- ctypes INPUT structs (default emitter only) ---------------------------
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.c_size_t)]


class _UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _UNION)]


def _default_emit(events):
    arr = (_INPUT * len(events))()
    for i, ev in enumerate(events):
        if ev[0] == "mouse":
            _, flags, dx, dy, data = ev
            arr[i] = _INPUT(INPUT_MOUSE, _UNION(mi=_MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, 0)))
        else:
            _, vk, scan, flags = ev
            arr[i] = _INPUT(INPUT_KEYBOARD, _UNION(ki=_KEYBDINPUT(vk, scan, flags, 0, 0)))
    ctypes.windll.user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT))


# --- public actuators ------------------------------------------------------
def move(x, y, *, emit=_default_emit):
    nx, ny = _norm(x, y)
    emit([("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, nx, ny, 0)])


def click(x, y, button="left", double=False, *, emit=_default_emit):
    if button not in _BUTTON:
        raise ValueError(f"unknown button {button!r}")
    down, up = _BUTTON[button]
    nx, ny = _norm(x, y)
    events = [("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, nx, ny, 0)]
    for _ in range(2 if double else 1):
        events.append(("mouse", down, 0, 0, 0))
        events.append(("mouse", up, 0, 0, 0))
    emit(events)


def drag(x1, y1, x2, y2, button="left", *, emit=_default_emit):
    down, up = _BUTTON[button]
    ax1, ay1 = _norm(x1, y1)
    ax2, ay2 = _norm(x2, y2)
    emit([
        ("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax1, ay1, 0),
        ("mouse", down, 0, 0, 0),
        ("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax2, ay2, 0),
        ("mouse", up, 0, 0, 0),
    ])


def scroll(amount, *, emit=_default_emit):
    emit([("mouse", MOUSEEVENTF_WHEEL, 0, 0, int(amount) * WHEEL_DELTA)])


def type_text(s, *, emit=_default_emit):
    events = []
    for ch in s:
        cp = ord(ch)
        events.append(("key", 0, cp, KEYEVENTF_UNICODE))
        events.append(("key", 0, cp, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    if events:
        emit(events)


def press_keys(keys, *, emit=_default_emit):
    resolved = []
    for k in keys:
        vk = VK.get(k)
        if vk is None:
            raise ValueError(f"unknown key {k!r}")
        resolved.append(vk)
    events = [("key", vk, 0, 0) for vk in resolved]                     # all down, in order
    events += [("key", vk, 0, KEYEVENTF_KEYUP) for vk in reversed(resolved)]  # up, reversed
    emit(events)
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(input): SendInput raw actuators (inputraw.py)`.

---

### Task 3: `uia.py` — UI Automation backend (injectable element tree) + comtypes declared

**Files:**
- Create: `src/desktop/uia.py`
- Modify: `requirements.txt` (add `comtypes`), `Assist.spec` (`hiddenimports`)
- Test: `tests/test_uia.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `list_elements(root, *, interactable_only=True)->list[dict]`, `find_element(root, *, name=None, automation_id=None, control_type=None, nth=0)->element|None`, `invoke(element)->None`, `set_value(element, text)->None`, `get_value(element)->str`, `get_root(window_id, *, automation=None)->element`. All operate on a **duck-typed Element** exposing: `.name:str`, `.control_type:str`, `.automation_id:str`, `.bounds:tuple`, `.interactable:bool`, `.children()->list`, `.invoke()`, `.set_value(text)`, `.get_value()->str`. The real comtypes adapter lives in `_real_automation()` (not unit-tested; verified at packaging).

- [ ] **Step 1 — failing test** `tests/test_uia.py`:

```python
import src.desktop.uia as uia


class FakeEl:
    def __init__(self, name="", control_type="", automation_id="", bounds=(0, 0, 0, 0),
                 interactable=True, kids=None):
        self.name, self.control_type, self.automation_id = name, control_type, automation_id
        self.bounds, self.interactable, self._kids = bounds, interactable, kids or []
        self.invoked = False
        self.value = ""
    def children(self):
        return self._kids
    def invoke(self):
        self.invoked = True
    def set_value(self, text):
        self.value = text
    def get_value(self):
        return self.value


def _tree():
    save = FakeEl(name="Save", control_type="Button", automation_id="btnSave")
    edit = FakeEl(name="Body", control_type="Edit", automation_id="txt")
    label = FakeEl(name="Title", control_type="Text", interactable=False)
    return FakeEl(name="Win", control_type="Window", kids=[save, edit, label]), save, edit


def test_list_elements_interactable_only_skips_static():
    root, _s, _e = _tree()
    got = uia.list_elements(root)
    names = {g["name"] for g in got}
    assert "Save" in names and "Body" in names and "Title" not in names
    assert all({"name", "control_type", "automation_id", "bounds"} <= set(g) for g in got)


def test_find_element_by_name():
    root, save, _e = _tree()
    assert uia.find_element(root, name="Save") is save


def test_find_element_by_automation_id_and_type():
    root, save, edit = _tree()
    assert uia.find_element(root, automation_id="txt") is edit
    assert uia.find_element(root, control_type="Button") is save


def test_find_element_missing_returns_none():
    root, _s, _e = _tree()
    assert uia.find_element(root, name="Nope") is None


def test_find_element_nth():
    a = FakeEl(name="Item", control_type="ListItem")
    b = FakeEl(name="Item", control_type="ListItem")
    root = FakeEl(kids=[a, b])
    assert uia.find_element(root, name="Item", nth=1) is b


def test_invoke_and_set_and_get_value():
    root, save, edit = _tree()
    uia.invoke(save)
    assert save.invoked is True
    uia.set_value(edit, "hello")
    assert uia.get_value(edit) == "hello"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `src/desktop/uia.py`:

```python
"""UI Automation backend. Pure tree logic over a duck-typed Element; all
comtypes/IUIAutomation complexity is isolated in _real_automation() so unit
tests run against a trivial fake tree with no COM.

Duck-typed Element: .name, .control_type, .automation_id, .bounds,
.interactable, .children(), .invoke(), .set_value(text), .get_value().
"""
import logging

logger = logging.getLogger(__name__)

# Control types considered actionable by default (buttons, fields, lists, menus).
_INTERACTABLE_TYPES = {
    "Button", "Edit", "CheckBox", "RadioButton", "ComboBox", "List", "ListItem",
    "MenuItem", "Hyperlink", "Tab", "TabItem", "Slider", "TreeItem", "SplitButton",
}


def _walk(root):
    """Depth-first iterator over root's descendants (excluding root)."""
    for child in root.children():
        yield child
        yield from _walk(child)


def list_elements(root, *, interactable_only=True):
    out = []
    for el in _walk(root):
        if interactable_only and not (getattr(el, "interactable", True)
                                      or el.control_type in _INTERACTABLE_TYPES):
            continue
        out.append({
            "name": el.name,
            "control_type": el.control_type,
            "automation_id": el.automation_id,
            "bounds": el.bounds,
        })
    return out


def find_element(root, *, name=None, automation_id=None, control_type=None, nth=0):
    matches = []
    for el in _walk(root):
        if name is not None and el.name != name:
            continue
        if automation_id is not None and el.automation_id != automation_id:
            continue
        if control_type is not None and el.control_type != control_type:
            continue
        matches.append(el)
    return matches[nth] if nth < len(matches) else None


def invoke(element):
    element.invoke()


def set_value(element, text):
    element.set_value(text)


def get_value(element):
    return element.get_value()


def get_root(window_id, *, automation=None):
    """Return the root Element for a top-level window. `automation` is the
    duck-typed provider (injected in tests); production uses _real_automation()."""
    auto = automation or _real_automation()
    return auto.element_from_handle(int(window_id))


def _real_automation():
    """comtypes-backed IUIAutomation provider adapting real UIA elements to the
    duck-typed Element surface. NOT unit-tested — verified live at packaging
    (plan-time verification #1). If comtypes fails to bundle in the frozen
    build, fall back per the spec's ladder (raw-ctypes vtable wrapper)."""
    import comtypes.client
    uia_mod = comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient as UIA
    auto = comtypes.client.CreateObject(
        "{ff48dba4-60ef-4201-aa87-54103eef594e}", interface=UIA.IUIAutomation)

    class _El:
        def __init__(self, raw):
            self._raw = raw
        @property
        def name(self):
            return self._raw.CurrentName or ""
        @property
        def control_type(self):
            return _CT.get(self._raw.CurrentControlType, str(self._raw.CurrentControlType))
        @property
        def automation_id(self):
            return self._raw.CurrentAutomationId or ""
        @property
        def bounds(self):
            r = self._raw.CurrentBoundingRectangle
            return (r.left, r.top, r.right, r.bottom)
        @property
        def interactable(self):
            return self.control_type in _INTERACTABLE_TYPES
        def children(self):
            walker = auto.RawViewWalker
            kids, ch = [], walker.GetFirstChildElement(self._raw)
            while ch:
                kids.append(_El(ch))
                ch = walker.GetNextSiblingElement(ch)
            return kids
        def invoke(self):
            pat = self._raw.GetCurrentPattern(UIA.UIA_InvokePatternId)
            if pat:
                pat.QueryInterface(UIA.IUIAutomationInvokePattern).Invoke()
            else:
                l, t, r, b = self.bounds
                from src.desktop.inputraw import click
                click((l + r) // 2, (t + b) // 2)
        def set_value(self, text):
            pat = self._raw.GetCurrentPattern(UIA.UIA_ValuePatternId)
            if pat:
                pat.QueryInterface(UIA.IUIAutomationValuePattern).SetValue(text)
            else:
                self._raw.SetFocus()
                from src.desktop.inputraw import type_text
                type_text(text)
        def get_value(self):
            pat = self._raw.GetCurrentPattern(UIA.UIA_ValuePatternId)
            if not pat:
                return ""
            return pat.QueryInterface(UIA.IUIAutomationValuePattern).CurrentValue or ""

    class _Auto:
        def element_from_handle(self, hwnd):
            return _El(auto.ElementFromHandle(hwnd))

    return _Auto()


# UIA numeric ControlTypeId -> friendly name (the subset we surface).
_CT = {
    50000: "Button", 50004: "Edit", 50002: "CheckBox", 50013: "RadioButton",
    50003: "ComboBox", 50008: "List", 50007: "ListItem", 50011: "MenuItem",
    50005: "Hyperlink", 50018: "Tab", 50019: "TabItem", 50015: "Slider",
    50023: "TreeItem", 50032: "Window", 50020: "Text",
}
```

Add `comtypes` to `requirements.txt` (its own line). In `Assist.spec`, add `'comtypes'` (and `'comtypes.client'`, `'comtypes.gen'`) to the `hiddenimports` list so PyInstaller bundles it.

- [ ] **Step 4 — run, expect PASS** (unit tests use FakeEl; the real adapter is untouched). Also `python -c "import comtypes"` to confirm the dep installs in the dev env.
- [ ] **Step 5 — commit** `feat(input): UI Automation backend (uia.py) + comtypes dep`.

---

### Task 4: raw-input tools — `mouse` + `keyboard`

**Files:**
- Create: `src/agent_tools/input_tools.py`
- Test: `tests/test_input_tools_raw.py`

**Interfaces:**
- Consumes: `src.desktop.inputraw.*`; `get_setting("input_control_enabled")`.
- Produces: `MouseTool`, `KeyboardTool` with `async execute(content, ctx)->{output|error, exit_code}`. Module-level names `inputraw` (for monkeypatch) and a shared `_args`/`_input_gate` helper.

- [ ] **Step 1 — failing test** `tests/test_input_tools_raw.py`:

```python
import asyncio
import json
import src.agent_tools.input_tools as it


def _run(tool, payload, enabled=True, monkeypatch=None):
    monkeypatch.setattr(it, "get_setting",
                        lambda k, d=None: enabled if k == "input_control_enabled" else d)
    return asyncio.run(tool.execute(json.dumps(payload), {"owner": "u"}))


def test_mouse_refuses_when_input_control_off(monkeypatch):
    r = _run(it.MouseTool(), {"action": "click", "x": 1, "y": 2}, enabled=False, monkeypatch=monkeypatch)
    assert r["exit_code"] == 1 and "input control" in r["error"].lower()


def test_mouse_click_calls_backend(monkeypatch):
    calls = {}
    monkeypatch.setattr(it.inputraw, "click", lambda x, y, **k: calls.update(x=x, y=y, kw=k))
    r = _run(it.MouseTool(), {"action": "click", "x": 40, "y": 50}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and calls["x"] == 40 and calls["y"] == 50


def test_mouse_scroll_calls_backend(monkeypatch):
    got = {}
    monkeypatch.setattr(it.inputraw, "scroll", lambda amount, **k: got.update(a=amount))
    r = _run(it.MouseTool(), {"action": "scroll", "amount": -3}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and got["a"] == -3


def test_mouse_bad_action_errors(monkeypatch):
    r = _run(it.MouseTool(), {"action": "teleport"}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 1


def test_keyboard_type_calls_backend(monkeypatch):
    got = {}
    monkeypatch.setattr(it.inputraw, "type_text", lambda s, **k: got.update(s=s))
    r = _run(it.KeyboardTool(), {"action": "type", "text": "hello"}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and got["s"] == "hello"


def test_keyboard_hotkey_calls_backend(monkeypatch):
    got = {}
    monkeypatch.setattr(it.inputraw, "press_keys", lambda keys, **k: got.update(keys=keys))
    r = _run(it.KeyboardTool(), {"action": "hotkey", "keys": ["ctrl", "s"]}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and got["keys"] == ["ctrl", "s"]


def test_keyboard_refuses_when_off(monkeypatch):
    r = _run(it.KeyboardTool(), {"action": "type", "text": "x"}, enabled=False, monkeypatch=monkeypatch)
    assert r["exit_code"] == 1 and "input control" in r["error"].lower()
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `src/agent_tools/input_tools.py`:

```python
"""Input-automation tools: raw mouse/keyboard (SendInput) and UI-Automation
element targeting. Acting tools are gated by input_control_enabled; the reading
tool list_ui_elements is gated by screen_access_enabled. Thin wrappers over
src.desktop.inputraw / src.desktop.uia."""
import json
import logging

from src.settings import get_setting
from src.desktop import inputraw
from src.desktop import uia

logger = logging.getLogger(__name__)

_OFF_MSG = ("Input control is off. Ask the user to enable 'Allow input control' "
            "in the sidebar before moving the mouse or typing.")


def _args(content):
    try:
        return json.loads(content) if content.strip().startswith("{") else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _input_on():
    return bool(get_setting("input_control_enabled", False))


class MouseTool:
    async def execute(self, content, ctx):
        if not _input_on():
            return {"error": _OFF_MSG, "exit_code": 1}
        a = _args(content)
        action = (a.get("action") or "").strip().lower()
        try:
            if action in ("click", "double", "right"):
                inputraw.click(int(a["x"]), int(a["y"]),
                               button="right" if action == "right" else "left",
                               double=(action == "double"))
            elif action == "move":
                inputraw.move(int(a["x"]), int(a["y"]))
            elif action == "drag":
                inputraw.drag(int(a["x"]), int(a["y"]), int(a["to_x"]), int(a["to_y"]))
            elif action == "scroll":
                inputraw.scroll(int(a.get("amount", 0)))
            else:
                return {"error": "mouse: action must be move|click|double|right|drag|scroll",
                        "exit_code": 1}
        except (KeyError, ValueError, TypeError) as e:
            return {"error": f"mouse: bad args ({e})", "exit_code": 1}
        logger.info("mouse: %s %s", action, {k: a.get(k) for k in ("x", "y", "to_x", "to_y", "amount")})
        return {"output": f"mouse {action} ok", "exit_code": 0}


class KeyboardTool:
    async def execute(self, content, ctx):
        if not _input_on():
            return {"error": _OFF_MSG, "exit_code": 1}
        a = _args(content)
        action = (a.get("action") or "").strip().lower()
        try:
            if action == "type":
                inputraw.type_text(str(a.get("text", "")))
            elif action == "hotkey":
                keys = a.get("keys") or []
                if not isinstance(keys, list) or not keys:
                    return {"error": "keyboard: hotkey needs a non-empty keys list", "exit_code": 1}
                inputraw.press_keys([str(k) for k in keys])
            else:
                return {"error": "keyboard: action must be type|hotkey", "exit_code": 1}
        except ValueError as e:
            return {"error": f"keyboard: {e}", "exit_code": 1}
        logger.info("keyboard: %s", action)
        return {"output": f"keyboard {action} ok", "exit_code": 0}
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(input): mouse + keyboard tools`.

---

### Task 5: UIA tools — `list_ui_elements` + `click_element` + `set_element_text`

**Files:**
- Modify: `src/agent_tools/input_tools.py` (add three tool classes)
- Test: `tests/test_input_tools_uia.py`

**Interfaces:**
- Consumes: `src.desktop.uia.*`; `get_setting("screen_access_enabled")` (read gate) and `get_setting("input_control_enabled")` (act gate).
- Produces: `ListUiElementsTool` (screen-access gated), `ClickElementTool`, `SetElementTextTool` (input-control gated). All resolve the window root via `uia.get_root`.

- [ ] **Step 1 — failing test** `tests/test_input_tools_uia.py`:

```python
import asyncio
import json
import src.agent_tools.input_tools as it


class FakeEl:
    def __init__(self, name="", control_type="", automation_id="", bounds=(0, 0, 1, 1), kids=None):
        self.name, self.control_type, self.automation_id = name, control_type, automation_id
        self.bounds, self._kids = bounds, kids or []
        self.invoked, self.value = False, ""
    def children(self):
        return self._kids
    def invoke(self):
        self.invoked = True
    def set_value(self, text):
        self.value = text
    def get_value(self):
        return self.value


def _gates(monkeypatch, screen=True, inp=True):
    monkeypatch.setattr(it, "get_setting",
                        lambda k, d=None: {"screen_access_enabled": screen,
                                           "input_control_enabled": inp}.get(k, d))


def _root(monkeypatch, root):
    monkeypatch.setattr(it.uia, "get_root", lambda wid, **k: root)


def test_list_ui_elements_refuses_without_screen_access(monkeypatch):
    _gates(monkeypatch, screen=False)
    r = asyncio.run(it.ListUiElementsTool().execute(json.dumps({"window_id": 1}), {}))
    assert r["exit_code"] == 1 and "screen access" in r["error"].lower()


def test_list_ui_elements_returns_controls(monkeypatch):
    _gates(monkeypatch)
    save = FakeEl(name="Save", control_type="Button", automation_id="s")
    _root(monkeypatch, FakeEl(control_type="Window", kids=[save]))
    r = asyncio.run(it.ListUiElementsTool().execute(json.dumps({"window_id": 7}), {}))
    assert r["exit_code"] == 0 and "Save" in r["output"]


def test_click_element_refuses_without_input_control(monkeypatch):
    _gates(monkeypatch, inp=False)
    _root(monkeypatch, FakeEl(kids=[FakeEl(name="Save", control_type="Button")]))
    r = asyncio.run(it.ClickElementTool().execute(json.dumps({"window_id": 1, "name": "Save"}), {}))
    assert r["exit_code"] == 1 and "input control" in r["error"].lower()


def test_click_element_invokes_match(monkeypatch):
    _gates(monkeypatch)
    save = FakeEl(name="Save", control_type="Button")
    _root(monkeypatch, FakeEl(kids=[save]))
    r = asyncio.run(it.ClickElementTool().execute(json.dumps({"window_id": 1, "name": "Save"}), {}))
    assert r["exit_code"] == 0 and save.invoked is True


def test_click_element_missing_errors(monkeypatch):
    _gates(monkeypatch)
    _root(monkeypatch, FakeEl(kids=[]))
    r = asyncio.run(it.ClickElementTool().execute(json.dumps({"window_id": 1, "name": "Nope"}), {}))
    assert r["exit_code"] == 1 and "no matching" in r["error"].lower()


def test_set_element_text_sets_value(monkeypatch):
    _gates(monkeypatch)
    edit = FakeEl(name="Body", control_type="Edit", automation_id="txt")
    _root(monkeypatch, FakeEl(kids=[edit]))
    r = asyncio.run(it.SetElementTextTool().execute(
        json.dumps({"window_id": 1, "automation_id": "txt", "text": "hi"}), {}))
    assert r["exit_code"] == 0 and edit.value == "hi"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** — append to `src/agent_tools/input_tools.py`:

```python
def _screen_on():
    return bool(get_setting("screen_access_enabled", False))


def _resolve(a):
    """Find the target element from args; returns (element, error_str)."""
    root = uia.get_root(a.get("window_id") or "focused")
    el = uia.find_element(root, name=a.get("name"), automation_id=a.get("automation_id"),
                          control_type=a.get("control_type"), nth=int(a.get("nth", 0)))
    if el is None:
        return None, "no matching UI element"
    return el, None


class ListUiElementsTool:
    async def execute(self, content, ctx):
        if not _screen_on():
            return {"error": "Screen access is off. Ask the user to enable 'Allow screen "
                             "access' in the sidebar before reading UI elements.", "exit_code": 1}
        a = _args(content)
        try:
            root = uia.get_root(a.get("window_id") or "focused")
            els = uia.list_elements(root)
        except Exception as e:
            return {"error": f"list_ui_elements: {e}", "exit_code": 1}
        if not els:
            return {"output": "No interactable UI elements found", "exit_code": 0}
        lines = [f"[{e['control_type']}] {e['name']!r} id={e['automation_id']!r} @{e['bounds']}"
                 for e in els]
        return {"output": f"{len(els)} element(s):\n" + "\n".join(lines), "exit_code": 0}


class ClickElementTool:
    async def execute(self, content, ctx):
        if not _input_on():
            return {"error": _OFF_MSG, "exit_code": 1}
        a = _args(content)
        try:
            el, err = _resolve(a)
            if err:
                return {"error": f"click_element: {err}", "exit_code": 1}
            uia.invoke(el)
        except Exception as e:
            return {"error": f"click_element: {e}", "exit_code": 1}
        logger.info("click_element: %s", {k: a.get(k) for k in ("name", "automation_id", "control_type")})
        return {"output": "click_element ok", "exit_code": 0}


class SetElementTextTool:
    async def execute(self, content, ctx):
        if not _input_on():
            return {"error": _OFF_MSG, "exit_code": 1}
        a = _args(content)
        if "text" not in a:
            return {"error": "set_element_text: text required", "exit_code": 1}
        try:
            el, err = _resolve(a)
            if err:
                return {"error": f"set_element_text: {err}", "exit_code": 1}
            uia.set_value(el, str(a["text"]))
        except Exception as e:
            return {"error": f"set_element_text: {e}", "exit_code": 1}
        logger.info("set_element_text: %s", {k: a.get(k) for k in ("name", "automation_id")})
        return {"output": "set_element_text ok", "exit_code": 0}
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(input): UIA tools (list_ui_elements, click_element, set_element_text)`.

---

### Task 6: Register all five tools across the tool system + guard test

**Files:**
- Modify: `src/agent_tools/__init__.py`, `src/agent_loop.py`, `src/tool_schemas.py`, `src/tool_index.py`, `src/tool_execution.py`, `src/tool_security.py`
- Test: `tests/test_input_tools_registration.py`

**Interfaces:**
- Consumes: the five tool classes from Tasks 4-5.
- Produces: the five tools callable through the normal agent path, admin-gated, plan-mode-partitioned.

- [ ] **Step 1 — failing guard test** `tests/test_input_tools_registration.py`:

```python
import src.agent_tools as agent_tools
import src.tool_security as ts
import src.tool_index as ti
import src.agent_loop as al

ACT = {"click_element", "set_element_text", "mouse", "keyboard"}
ALL5 = ACT | {"list_ui_elements"}


def test_all_five_in_tool_handlers_and_tags():
    for name in ALL5:
        assert name in agent_tools.TOOL_HANDLERS, f"{name} missing from TOOL_HANDLERS"
        assert name in agent_tools.TOOL_TAGS, f"{name} missing from TOOL_TAGS"


def test_all_five_admin_blocked():
    for name in ALL5:
        assert name in ts.NON_ADMIN_BLOCKED_TOOLS, f"{name} not admin-gated"


def test_plan_mode_partition():
    # Only the reader is plan-mode read-only; the four actors are blocked.
    assert "list_ui_elements" in ts.PLAN_MODE_READONLY_TOOLS
    for name in ACT:
        assert name not in ts.PLAN_MODE_READONLY_TOOLS, f"{name} must be blocked in plan mode"


def test_index_and_prompt_sections_present():
    for name in ALL5:
        assert name in ti.BUILTIN_TOOL_DESCRIPTIONS, f"{name} missing from tool index"
        assert name in al.TOOL_SECTIONS, f"{name} missing from TOOL_SECTIONS"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the six edits (follow the `capture_screen`/`control_window` precedent verbatim at each site):

**`src/agent_tools/__init__.py`** — add the import near the other desktop-tool imports (grep for `LaunchAppTool` to find it), then into `TOOL_HANDLERS` (after the `"capture_screen": ...` line, line 60):

```python
    "list_ui_elements": ListUiElementsTool().execute,
    "click_element": ClickElementTool().execute,
    "set_element_text": SetElementTextTool().execute,
    "mouse": MouseTool().execute,
    "keyboard": KeyboardTool().execute,
```

Import line (place with the desktop imports):

```python
from .input_tools import (
    ListUiElementsTool, ClickElementTool, SetElementTextTool, MouseTool, KeyboardTool,
)
```

Add the five names to the `TOOL_TAGS` set (extend the desktop line, currently line 86).

**`src/tool_security.py`** — add all five to `NON_ADMIN_BLOCKED_TOOLS` (after `"capture_screen",` at line 56):

```python
    "list_ui_elements",
    "click_element",
    "set_element_text",
    "mouse",
    "keyboard",
```

Add ONLY the reader to `PLAN_MODE_READONLY_TOOLS` (after `"capture_screen",` at line 130):

```python
    "list_ui_elements",
```

**`src/tool_index.py`** — add to the `BUILTIN_TOOL_DESCRIPTIONS` dict (after the `capture_screen` entry, line 84):

```python
    "list_ui_elements": "List interactable UI controls of a window (name, type, id, bounds). Requires screen access. Use before click_element to find targets.",
    "click_element": "Click a UI control by name/automation_id/control_type via UI Automation. Requires input control.",
    "set_element_text": "Set the text of a UI control (e.g. a text field) via UI Automation. Requires input control.",
    "mouse": "Move/click/double/right/drag/scroll the mouse at coordinates. Requires input control. Confirm risky actions with the user.",
    "keyboard": "Type text or press a hotkey (e.g. ctrl+s). Requires input control. Confirm risky actions with the user.",
```

**`src/tool_execution.py`** — extend the direct-dispatch tuple (lines 749-751) to include the five names:

```python
    elif tool in ("grep", "glob", "ls", "get_workspace", "open_in_vscode",
                 "launch_app", "find_files", "list_windows",
                 "control_window", "capture_screen",
                 "list_ui_elements", "click_element", "set_element_text",
                 "mouse", "keyboard"):
```

**`src/agent_loop.py`** — add the five to the `desktop` domain set (line 300):

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

Append to the `_DOMAIN_RULES["desktop"]` string (line 279-285) two lines:

```
- `list_ui_elements` requires screen access; `click_element`/`set_element_text`/`mouse`/`keyboard` require the user to have enabled input control.
- Prefer `click_element`/`set_element_text` (targeting controls by name) over raw `mouse` coordinates; confirm irreversible actions with the user first.
```

Add five `TOOL_SECTIONS` entries mirroring the `capture_screen` block's fenced style (grep `capture_screen` in `agent_loop.py`, ~line 409):

```python
    "list_ui_elements": """\
```list_ui_elements
{"window_id": <id from list_windows> | "focused"}
```
List a window's interactable UI controls (name, type, automation_id, bounds). Requires screen access.""",
    "click_element": """\
```click_element
{"window_id": <id> | "focused", "name": "Save"}  // or automation_id / control_type, optional nth
```
Click a control by identity via UI Automation. Requires input control.""",
    "set_element_text": """\
```set_element_text
{"window_id": <id>, "automation_id": "txt", "text": "hello"}
```
Set a control's text via UI Automation. Requires input control.""",
    "mouse": """\
```mouse
{"action": "click", "x": 840, "y": 220}  // move|click|double|right|drag(to_x,to_y)|scroll(amount)
```
Raw mouse actuator at screen coordinates. Requires input control.""",
    "keyboard": """\
```keyboard
{"action": "type", "text": "hello"}  // or {"action":"hotkey","keys":["ctrl","s"]}
```
Type text or press a hotkey. Requires input control.""",
```

**`src/tool_schemas.py`** — add five function schemas mirroring the `control_window`/`capture_screen` entries (grep `capture_screen`, ~line 225), and add the five names to the desktop content-marshalling tuple (line 1430-1431, currently ends `"control_window", "capture_screen")`):

```python
        {"type": "function", "function": {
            "name": "list_ui_elements",
            "description": "List interactable UI controls of a window via UI Automation.",
            "parameters": {"type": "object", "properties": {
                "window_id": {"description": "Window id from list_windows, or 'focused'"}}}}},
        {"type": "function", "function": {
            "name": "click_element",
            "description": "Click a UI control by name/automation_id/control_type via UI Automation.",
            "parameters": {"type": "object", "properties": {
                "window_id": {"description": "Window id or 'focused'"},
                "name": {"type": "string"}, "automation_id": {"type": "string"},
                "control_type": {"type": "string"}, "nth": {"type": "integer"}}}}},
        {"type": "function", "function": {
            "name": "set_element_text",
            "description": "Set the text of a UI control via UI Automation.",
            "parameters": {"type": "object", "properties": {
                "window_id": {"description": "Window id or 'focused'"},
                "name": {"type": "string"}, "automation_id": {"type": "string"},
                "text": {"type": "string"}}, "required": ["text"]}}},
        {"type": "function", "function": {
            "name": "mouse",
            "description": "Move/click/drag/scroll the mouse at screen coordinates (SendInput).",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["move", "click", "double", "right", "drag", "scroll"]},
                "x": {"type": "integer"}, "y": {"type": "integer"},
                "to_x": {"type": "integer"}, "to_y": {"type": "integer"},
                "amount": {"type": "integer"}}, "required": ["action"]}}},
        {"type": "function", "function": {
            "name": "keyboard",
            "description": "Type text or press a hotkey (SendInput).",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["type", "hotkey"]},
                "text": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}}}, "required": ["action"]}}},
```

(Insert these into the same list/structure the existing desktop schemas live in — match the surrounding brackets exactly.)

- [ ] **Step 4 — run the guard test, expect PASS.** Also `python -c "import src.agent_tools, src.agent_loop, src.tool_schemas, src.tool_execution, src.tool_security, src.tool_index"` to confirm every edited module imports cleanly.
- [ ] **Step 5 — commit** `feat(input): register input tools across the tool system`.

---

### Task 7: Sidebar "Allow input control" toggle + indicator + inputControl.js

**Files:**
- Modify: `static/index.html` (sidebar block near the Screen-access item, lines 953-961)
- Create: `static/js/inputControl.js`
- Test: `tests/test_input_control_ui.py`

**Interfaces:**
- Consumes: `GET`/`POST /api/auth/settings` with key `input_control_enabled`.
- Produces: sidebar elements `input-control-toggle`, `input-control-indicator`, and the script include.

- [ ] **Step 1 — failing UI guard** `tests/test_input_control_ui.py`:

```python
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def _read(p): return (ROOT / p).read_text(encoding="utf-8")


def test_index_has_input_control_toggle():
    html = _read("static/index.html")
    for el in ('id="input-control-toggle"', 'id="input-control-indicator"'):
        assert el in html, f"{el} missing from index.html"
    assert 'src="/static/js/inputControl.js"' in html


def test_inputcontrol_js_posts_setting():
    js = _read("static/js/inputControl.js")
    assert "input_control_enabled" in js
    assert "/api/auth/settings" in js
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement.** In `static/index.html`, the Screen-access sidebar item is at lines 953-961 (the `<div class="list-item">` containing `id="screen-access-toggle"`). Directly AFTER that item's closing `</div>` (line 961), add a parallel block:

```html
        <div class="list-item">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><path d="M9 11V6a3 3 0 0 1 6 0v5M5 11h14l-1 10H6z"/></svg>
          <span class="grow">Input control</span>
          <span id="input-control-indicator" class="sidebar-notif-dot" style="display:none;background:var(--green,#50fa7b);margin-right:6px;" title="Input control is ON"></span>
          <label class="admin-switch" style="flex-shrink:0;" title="Enable mouse/keyboard automation">
            <input type="checkbox" id="input-control-toggle">
            <span class="admin-slider"></span>
          </label>
        </div>
```

Add `<script src="/static/js/inputControl.js"></script>` beside the `screenAccess.js` include (grep `screenAccess.js` in index.html). Create `static/js/inputControl.js` (mirror `screenAccess.js` verbatim, swapping the ids and setting key):

```javascript
// Input-control sidebar toggle: reflects and updates the admin-only
// `input_control_enabled` setting that gates the mouse/keyboard/UIA acting
// tools. Mirrors screenAccess.js. Defaults off and is reset off server-side
// on every restart (src/settings.py); this widget just displays and flips it.
(function () {
  function $(id) { return document.getElementById(id); }

  function reflect(enabled) {
    const toggle = $('input-control-toggle');
    const indicator = $('input-control-indicator');
    if (toggle) toggle.checked = !!enabled;
    if (indicator) indicator.style.display = enabled ? '' : 'none';
  }

  async function load() {
    try {
      const res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
      if (!res.ok) return;
      const settings = await res.json();
      reflect(!!settings.input_control_enabled);
    } catch (e) { console.warn('Failed to load input control setting', e); }
  }

  async function save(enabled) {
    try {
      await fetch('/api/auth/settings', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_control_enabled: enabled })
      });
    } catch (e) { console.warn('Failed to save input control setting', e); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    load();
    $('input-control-toggle')?.addEventListener('change', (e) => {
      const enabled = !!e.target.checked;
      reflect(enabled);
      save(enabled);
    });
  });
})();
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(input): sidebar input-control toggle + indicator`.

---

### Task 8: Package + live-verify (resolves comtypes bundling definitively)

- [ ] **Step 1 — full affected suite:** `python -m pytest tests/test_input_control_setting.py tests/test_inputraw.py tests/test_uia.py tests/test_input_tools_raw.py tests/test_input_tools_uia.py tests/test_input_tools_registration.py tests/test_input_control_ui.py --import-mode=importlib -q` → all green.
- [ ] **Step 2 — build:** full clean `.\build-installer.ps1` (never `-Fast` — it has dropped bundled deps before). Confirm compile succeeds.
- [ ] **Step 3 — boot-verify the frozen exe** via `Assist.exe --run-py <probe.py> <out>`. The probe (write to the scratchpad) must, inside the frozen process: (a) confirm `DEFAULT_SETTINGS["input_control_enabled"] is False`; (b) import `src.desktop.inputraw` and `src.desktop.input_tools` and confirm all five tools are in `TOOL_HANDLERS`; (c) **exercise the real comtypes path** — `import comtypes.client; comtypes.client.GetModule("UIAutomationCore.dll")` and `CreateObject(... IUIAutomation)` succeed (this is plan-time verification #1 — if it fails in the frozen build, apply the spec's fallback ladder: pin/bundle the comtypes gen module, else the raw-ctypes vtable wrapper). Print `INPUT_BOOT=OK` / `FAIL`.
- [ ] **Step 4 — user manual test** (packaged exe): reinstall; toggle **Allow input control** on; open Notepad; `list_ui_elements` on its window finds the edit control; `set_element_text`/`keyboard type` enters text; `click_element` opens a menu; `mouse` drag + scroll behave; toggle **off** → every acting tool refuses; confirm `list_ui_elements` refuses when **screen access** is off. Confirm the shipped Desktop Control tools + Plugins hub still work.
- [ ] **Step 5 — commit** the installer: `git add -f installer/Output/Assist-Setup.exe && git commit -m "build: Assist-Setup.exe with input automation"`.

## Self-Review

- **Spec coverage:** inputraw SendInput actuators (T2), uia UIA backend + comtypes (T3), the 5 tools with read/act gating split (T4 mouse/keyboard behind input_control; T5 list_ui_elements behind screen_access + click/set behind input_control), input_control_enabled setting + startup reset (T1), sidebar toggle + indicator (T7), registration across all 8 sites + admin-gating + plan-mode partition + guard test (T6), audit logging (T4/T5 `logger.info`), package + live-verify with comtypes bundling resolved (T8). All spec sections covered.
- **Placeholders:** none — every code step carries complete code; the one untestable boundary (`uia._real_automation`) is fully written and explicitly designated for live verification in T8 with the spec's fallback ladder named.
- **Type consistency:** the duck-typed Element surface (`name/control_type/automation_id/bounds/interactable/children()/invoke()/set_value()/get_value()`) is identical across T3 (uia.py + tests) and T5 (tool tests' FakeEl). Event-descriptor tuples (`("mouse",flags,dx,dy,data)` / `("key",vk,scan,flags)`) are identical between T2's implementation and tests. Setting key `input_control_enabled` and the gate helpers `_input_on`/`_screen_on` are consistent across T1/T4/T5/T7. Tool names (`list_ui_elements`, `click_element`, `set_element_text`, `mouse`, `keyboard`) are identical across T4-T8.
