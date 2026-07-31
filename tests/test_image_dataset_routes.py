import io
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.image_dataset_routes as idr


def _client(monkeypatch):
    monkeypatch.setattr(idr, "require_admin", lambda: None)
    app = FastAPI(); app.include_router(idr.setup_image_dataset_routes())
    return TestClient(app)


def test_upload_then_caption_validate_save(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))
    monkeypatch.setattr(idr, "get_image_dataset_store",
                        lambda: idr.ImageDatasetStore(base_dir=str(tmp_path / "saved")))
    c = _client(monkeypatch)

    files = [("files", ("a.png", io.BytesIO(b"AAA"), "image/png")),
            ("files", ("b.png", io.BytesIO(b"BBB"), "image/png"))]
    r = c.post("/api/image-datasets/upload", files=files)
    assert r.status_code == 200
    wid = r.json()["working_set_id"]
    assert len(r.json()["images"]) == 2

    def fake_caption(path, *, vl_call=None, prompt=None, owner=None):
        return "an auto caption", None
    monkeypatch.setattr(idr, "caption_image", fake_caption)
    r2 = c.post("/api/image-datasets/caption", json={"working_set_id": wid})
    assert r2.status_code == 200 and all(img["caption"] == "an auto caption" for img in r2.json()["images"])

    captions = {img["id"]: "edited: " + img["caption"] for img in r2.json()["images"]}
    r3 = c.post("/api/image-datasets/validate", json={"working_set_id": wid, "captions": captions})
    assert r3.status_code == 200 and r3.json()["valid"] >= 0  # never-500; report shape present

    r4 = c.post("/api/image-datasets", json={"working_set_id": wid, "name": "my-set",
                                              "trigger_word": "ohwx", "captions": captions})
    assert r4.status_code == 200 and r4.json().get("ok")

    r5 = c.get("/api/image-datasets")
    assert r5.status_code == 200 and any(d["name"] == "my-set" for d in r5.json()["datasets"])


def test_working_image_served_and_path_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))
    c = _client(monkeypatch)
    files = [("files", ("a.png", io.BytesIO(b"AAA"), "image/png"))]
    r = c.post("/api/image-datasets/upload", files=files)
    wid = r.json()["working_set_id"]
    iid = r.json()["images"][0]["id"]
    r2 = c.get(f"/api/image-datasets/working/{wid}/{iid}")
    assert r2.status_code == 200 and r2.content == b"AAA"
    r3 = c.get(f"/api/image-datasets/working/{wid}/../../etc-passwd")
    assert r3.status_code == 404


def test_from_gallery_is_owner_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))

    class _FakeRow:
        def __init__(self, id, filename, owner, caption=""):
            self.id, self.filename, self.owner, self.caption = id, filename, owner, caption

    img_path = tmp_path / "gallery_img.png"
    img_path.write_bytes(b"GALLERYBYTES")

    class _FakeQuery:
        def __init__(self, rows): self._rows = rows
        def filter(self, *a, **k): return self
        def all(self): return self._rows

    class _FakeDb:
        def __init__(self, rows): self._rows = rows
        def query(self, *a, **k): return _FakeQuery(self._rows)
        def close(self): pass

    owned = _FakeRow("g1", "gallery_img.png", "admin", "gallery caption")
    other = _FakeRow("g2", "other.png", "someone-else", "")
    monkeypatch.setattr(idr, "SessionLocal", lambda: _FakeDb([owned, other]))
    monkeypatch.setattr(idr, "_gallery_image_path", lambda filename: str(img_path))

    c = _client(monkeypatch)
    r = c.post("/api/image-datasets/from-gallery", json={"ids": ["g1", "g2"], "owner": "admin"})
    assert r.status_code == 200
    assert len(r.json()["images"]) == 1  # only the admin-owned row copied in
