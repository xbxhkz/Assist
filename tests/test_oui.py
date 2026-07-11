import src.desktop.oui as oui


def test_known_prefix_maps_to_vendor():
    assert oui.oui_vendor("B8:27:EB:12:34:56") == "Raspberry Pi Foundation"


def test_accepts_dash_and_lowercase():
    assert oui.oui_vendor("b8-27-eb-aa-bb-cc") == "Raspberry Pi Foundation"


def test_unknown_prefix_is_none():
    assert oui.oui_vendor("02:00:00:00:00:00") is None


def test_malformed_mac_is_none():
    assert oui.oui_vendor("") is None
    assert oui.oui_vendor(None) is None
    assert oui.oui_vendor("zz") is None


def test_table_keys_are_uppercase_prefixes():
    for k in oui.OUI_VENDORS:
        assert k == k.upper() and len(k.split(":")) == 3
