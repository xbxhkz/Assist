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
