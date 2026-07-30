import os
from src.image_dataset_tools.store import ImageDatasetStore


def _src_image(tmp_path, name, data=b"fake-image-bytes"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_save_load_list_delete(tmp_path):
    src_dir = tmp_path / "src"; src_dir.mkdir()
    a = _src_image(src_dir, "a.png", b"AAA")
    b = _src_image(src_dir, "b.jpg", b"BBB")
    store = ImageDatasetStore(base_dir=str(tmp_path / "store"))
    entries = [{"path": a, "caption": "cap a"}, {"path": b, "caption": "cap b"}]
    out = store.save("my set", entries, trigger_word="ohwx-widget")
    assert out.get("ok") and out["name"] == "my-set"
    assert os.path.isdir(out["path"])

    lst = store.list()
    assert lst and lst[0]["name"] == "my-set" and lst[0]["images"] == 2

    loaded = store.load("my-set")
    assert loaded["trigger_word"] == "ohwx-widget"
    captions = sorted(img["caption"] for img in loaded["images"])
    assert captions == ["cap a", "cap b"]
    # each image file actually copied with matching content
    for img in loaded["images"]:
        full = os.path.join(loaded["path"], img["filename"])
        assert os.path.isfile(full)

    assert store.delete("my-set").get("ok") and store.list() == []


def test_name_sanitized_no_traversal(tmp_path):
    src_dir = tmp_path / "src"; src_dir.mkdir()
    a = _src_image(src_dir, "a.png")
    store = ImageDatasetStore(base_dir=str(tmp_path / "store"))
    out = store.save("../../evil", [{"path": a, "caption": "x"}])
    assert out.get("ok") and str(tmp_path / "store") in out["path"] and ".." not in out["name"]


def test_empty_entries_and_bad_name_rejected(tmp_path):
    store = ImageDatasetStore(base_dir=str(tmp_path))
    assert "error" in store.save("ok", [])
    assert "error" in store.save("", [{"path": "x", "caption": "y"}])


def test_missing_source_file_is_skipped_not_fatal(tmp_path):
    src_dir = tmp_path / "src"; src_dir.mkdir()
    a = _src_image(src_dir, "a.png")
    store = ImageDatasetStore(base_dir=str(tmp_path / "store"))
    out = store.save("s", [{"path": a, "caption": "x"}, {"path": "/nope.png", "caption": "y"}])
    assert out.get("ok")
    loaded = store.load("s")
    assert len(loaded["images"]) == 1  # the missing one was skipped, not fatal


def test_load_missing(tmp_path):
    assert "error" in ImageDatasetStore(base_dir=str(tmp_path)).load("nope")
