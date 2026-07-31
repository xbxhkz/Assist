import os
from src.image_dataset_tools import working_set as ws


def test_new_working_set_is_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    a, b = ws.new_working_set(), ws.new_working_set()
    assert a != b and len(a) >= 8


def test_add_list_get_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    added = ws.add_images(wid, [("photo one.png", b"AAA"), ("two.jpg", b"BBB", "seeded caption")])
    assert len(added) == 2
    imgs = ws.list_images(wid)
    assert len(imgs) == 2
    ids_by_caption = {img["caption"]: img["id"] for img in imgs}
    assert ids_by_caption["seeded caption"]
    for img in imgs:
        path = ws.get_image_path(wid, img["id"])
        assert path and os.path.isfile(path)


def test_set_caption_updates_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    added = ws.add_images(wid, [("a.png", b"AAA")])
    iid = added[0]["id"]
    assert ws.set_caption(wid, iid, "a new caption") is True
    imgs = ws.list_images(wid)
    assert imgs[0]["caption"] == "a new caption"


def test_delete_working_set(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    ws.add_images(wid, [("a.png", b"AAA")])
    assert ws.delete_working_set(wid) is True
    assert ws.list_images(wid) == []


def test_path_traversal_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    assert ws.get_image_path("../../evil", "x") is None
    assert ws.get_image_path(wid, "../../etc/passwd") is None


def test_never_raises_on_hostile_input(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    assert ws.add_images("bad id", None) == []
    assert ws.add_images("bad id", [("x",), 42, ("y.png", "not-bytes")]) == []
    assert ws.list_images("does-not-exist") == []
    assert ws.set_caption("does-not-exist", "x", "c") is False
    assert ws.delete_working_set("does-not-exist") is False
    assert ws.remove_image("does-not-exist", "x") is False


def test_safe_id_never_raises_on_hostile_object(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    class Hostile:
        def __str__(self): raise RuntimeError("str() blew up")
        def __bool__(self): raise RuntimeError("bool() blew up")
    assert ws.get_image_path(Hostile(), "x") is None
    assert ws.list_images(Hostile()) == []
    assert ws.set_caption(Hostile(), "x", "c") is False
    assert ws.delete_working_set(Hostile()) is False
    assert ws.add_images(Hostile(), [("a.png", b"A")]) == []
    assert ws.remove_image(Hostile(), "x") is False


def test_tampered_non_list_images_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    ws_dir = os.path.join(str(tmp_path), wid)
    os.makedirs(ws_dir, exist_ok=True)
    with open(os.path.join(ws_dir, "state.json"), "w", encoding="utf-8") as f:
        f.write('{"images": "not-a-list"}')
    assert ws.list_images(wid) == []
    assert ws.get_image_path(wid, "x") is None
    assert ws.set_caption(wid, "x", "c") is False
    assert ws.remove_image(wid, "x") is False


def test_remove_image(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    added = ws.add_images(wid, [("a.png", b"AAA"), ("b.png", b"BBB")])
    assert len(added) == 2
    id_a, id_b = added[0]["id"], added[1]["id"]
    path_a = ws.get_image_path(wid, id_a)
    assert path_a and os.path.isfile(path_a)

    assert ws.remove_image(wid, id_a) is True
    remaining = ws.list_images(wid)
    assert len(remaining) == 1 and remaining[0]["id"] == id_b
    assert not os.path.isfile(path_a)  # underlying file removed too

    # removing an id that no longer exists (or never existed) reports failure
    assert ws.remove_image(wid, id_a) is False
    assert ws.remove_image(wid, "does-not-exist") is False
    assert ws.remove_image("does-not-exist-ws", id_b) is False


def test_add_images_and_set_caption_report_failure_when_write_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()

    real_replace = ws.os.replace
    def flaky_replace(*a, **k):
        raise OSError("simulated state-write failure")
    monkeypatch.setattr(ws.os, "replace", flaky_replace)

    added = ws.add_images(wid, [("a.png", b"AAA")])
    assert added == []  # must NOT claim success when the state write failed
    assert ws.list_images(wid) == []

    monkeypatch.setattr(ws.os, "replace", real_replace)
    added2 = ws.add_images(wid, [("b.png", b"BBB")])
    assert len(added2) == 1
    iid = added2[0]["id"]

    monkeypatch.setattr(ws.os, "replace", flaky_replace)
    assert ws.set_caption(wid, iid, "new caption") is False  # must NOT claim success
    assert ws.list_images(wid)[0]["caption"] == ""  # unchanged
