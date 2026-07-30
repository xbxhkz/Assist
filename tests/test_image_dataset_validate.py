import io
from PIL import Image
from src.image_dataset_tools.validate import validate_image_set


def _make_png(path, size=(512, 512), color=(255, 0, 0)):
    Image.new("RGB", size, color).save(path, format="PNG")


def test_validate_clean_set(tmp_path):
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    _make_png(p1); _make_png(p2, color=(0, 255, 0))
    entries = [{"id": "a", "path": str(p1), "caption": "a red square"},
              {"id": "b", "path": str(p2), "caption": "a green square"}]
    rep = validate_image_set(entries)
    assert rep["total"] == 2 and rep["valid"] == 2 and rep["invalid"] == 0
    assert rep["stats"]["duplicates"] == 0 and rep["stats"]["missing_captions"] == 0


def test_validate_flags_missing_caption(tmp_path):
    p = tmp_path / "a.png"; _make_png(p)
    rep = validate_image_set([{"id": "a", "path": str(p), "caption": ""}])
    assert rep["invalid"] == 1 and rep["errors"][0]["id"] == "a"
    assert "caption" in rep["errors"][0]["message"].lower()


def test_validate_flags_corrupt_image(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a real png")
    rep = validate_image_set([{"id": "bad", "path": str(p), "caption": "x"}])
    assert rep["invalid"] == 1 and "corrupt" in rep["errors"][0]["message"].lower() or \
          "unreadable" in rep["errors"][0]["message"].lower()


def test_validate_flags_duplicates(tmp_path):
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    _make_png(p1); _make_png(p2)  # identical content
    entries = [{"id": "a", "path": str(p1), "caption": "x"},
              {"id": "b", "path": str(p2), "caption": "y"}]
    rep = validate_image_set(entries)
    assert rep["stats"]["duplicates"] == 1
    assert any("duplicate" in e["message"].lower() for e in rep["errors"])


def test_validate_flags_small_resolution(tmp_path):
    p = tmp_path / "tiny.png"; _make_png(p, size=(32, 32))
    rep = validate_image_set([{"id": "a", "path": str(p), "caption": "x"}], min_dimension=256)
    assert rep["invalid"] == 1 and "resolution" in rep["errors"][0]["message"].lower()


def test_validate_never_raises_on_hostile_input():
    assert validate_image_set(None)["total"] == 0
    assert validate_image_set("not-a-list")["total"] == 0
    rep = validate_image_set([{"id": "a"}, "not-a-dict", 42, {"path": "/nope/x.png", "caption": "x"}])
    assert rep["total"] == 4 and rep["invalid"] == 4  # every entry unresolvable, none raise


def test_validate_never_raises_if_pillow_import_fails(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "PIL":
            raise ImportError("simulated: Pillow DLL failed to load")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from src.image_dataset_tools.validate import validate_image_set
    rep = validate_image_set([{"id": "a", "path": "whatever.png", "caption": "x"}])
    assert rep["total"] == 1 and rep["invalid"] == 1 and rep["valid"] == 0
