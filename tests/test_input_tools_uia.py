import asyncio
import json
import src.agent_tools.input_tools as it


class FakeEl:
    def __init__(self, name="", control_type="", automation_id="", bounds=(0, 0, 1, 1),
                 kids=None, interactable=True):
        self.name, self.control_type, self.automation_id = name, control_type, automation_id
        self.bounds, self._kids = bounds, kids or []
        self.interactable = interactable
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


def test_set_element_text_refuses_without_input_control(monkeypatch):
    _gates(monkeypatch, inp=False)
    _root(monkeypatch, FakeEl(kids=[FakeEl(name="Body", control_type="Edit", automation_id="txt")]))
    r = asyncio.run(it.SetElementTextTool().execute(
        json.dumps({"window_id": 1, "automation_id": "txt", "text": "x"}), {}))
    assert r["exit_code"] == 1 and "input control" in r["error"].lower()
