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
