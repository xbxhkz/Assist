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
