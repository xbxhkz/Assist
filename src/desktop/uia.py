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
