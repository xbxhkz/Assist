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
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
MOUSEEVENTF_VIRTUALDESK = 0x4000

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


def _virtual_rect():
    """Virtual-desktop rect (left, top, width, height), spanning all monitors.
    Monkeypatched in tests. left/top may be negative for monitors above/left of
    the primary."""
    u = ctypes.windll.user32
    return (u.GetSystemMetrics(SM_XVIRTUALSCREEN), u.GetSystemMetrics(SM_YVIRTUALSCREEN),
            u.GetSystemMetrics(SM_CXVIRTUALSCREEN), u.GetSystemMetrics(SM_CYVIRTUALSCREEN))


def _norm(x, y):
    left, top, w, h = _virtual_rect()
    nx = int((x - left) * 65535 / max(1, w - 1))
    ny = int((y - top) * 65535 / max(1, h - 1))
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
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    n = user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT))
    if n != len(events):
        raise OSError(f"SendInput injected {n}/{len(events)} events "
                      f"(GetLastError={ctypes.get_last_error()})")


# --- public actuators ------------------------------------------------------
def move(x, y, *, emit=_default_emit):
    nx, ny = _norm(x, y)
    emit([("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, nx, ny, 0)])


def click(x, y, button="left", double=False, *, emit=_default_emit):
    if button not in _BUTTON:
        raise ValueError(f"unknown button {button!r}")
    down, up = _BUTTON[button]
    nx, ny = _norm(x, y)
    events = [("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, nx, ny, 0)]
    for _ in range(2 if double else 1):
        events.append(("mouse", down, 0, 0, 0))
        events.append(("mouse", up, 0, 0, 0))
    emit(events)


def drag(x1, y1, x2, y2, button="left", *, emit=_default_emit):
    if button not in _BUTTON:
        raise ValueError(f"unknown button {button!r}")
    down, up = _BUTTON[button]
    ax1, ay1 = _norm(x1, y1)
    ax2, ay2 = _norm(x2, y2)
    emit([
        ("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax1, ay1, 0),
        ("mouse", down, 0, 0, 0),
        ("mouse", MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax2, ay2, 0),
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
