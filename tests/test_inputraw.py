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
