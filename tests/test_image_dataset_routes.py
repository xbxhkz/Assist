import io
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
        def filter(self, *a, **k): return self  # id.in_(...) filter -- accept-all in this fake
        def all(self): return self._rows

    class _FakeDb:
        def __init__(self, rows): self._rows = rows
        def query(self, *a, **k): return _FakeQuery(self._rows)
        def close(self): pass

    owned = _FakeRow("g1", "gallery_img.png", "admin", "gallery caption")
    other = _FakeRow("g2", "other.png", "someone-else", "")
    monkeypatch.setattr(idr, "SessionLocal", lambda: _FakeDb([owned, other]))
    monkeypatch.setattr(idr, "_gallery_image_path", lambda filename: str(img_path))
    # owner_filter runs the REAL filtering logic against the fake rows (in-memory,
    # since the fake query's .filter() is a no-op passthrough) -- simulate its
    # effect by monkeypatching it directly, since the fake DB has no real SQLAlchemy
    # column expressions to filter on.
    def fake_owner_filter(query, model_cls, user, *, include_shared=True):
        class _Filtered:
            def all(self_inner):
                return [r for r in query.all() if r.owner == user or (include_shared and r.owner is None)]
        return _Filtered()
    monkeypatch.setattr(idr, "owner_filter", fake_owner_filter)

    monkeypatch.setattr(idr, "require_admin", lambda: None)
    app = FastAPI()

    @app.middleware("http")
    async def _set_user(request, call_next):
        request.state.current_user = "admin"  # the REAL authenticated identity
        return await call_next(request)

    app.include_router(idr.setup_image_dataset_routes())
    c = TestClient(app)

    # Even though the client tries to claim a different owner in the body, the
    # route must use the REAL authenticated identity (request.state), not this.
    r = c.post("/api/image-datasets/from-gallery", json={"ids": ["g1", "g2"], "owner": "someone-else"})
    assert r.status_code == 200
    assert len(r.json()["images"]) == 1  # only the admin-owned row, regardless of the spoofed body field

    # Omitting the body owner field entirely must ALSO stay scoped to the real user.
    r2 = c.post("/api/image-datasets/from-gallery", json={"ids": ["g1", "g2"]})
    assert r2.status_code == 200
    assert len(r2.json()["images"]) == 1


def test_remove_working_image_excludes_from_validate_and_save(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))
    monkeypatch.setattr(idr, "get_image_dataset_store",
                        lambda: idr.ImageDatasetStore(base_dir=str(tmp_path / "saved")))
    c = _client(monkeypatch)

    files = [("files", ("a.png", io.BytesIO(b"AAA"), "image/png")),
            ("files", ("b.png", io.BytesIO(b"BBB"), "image/png")),
            ("files", ("c.png", io.BytesIO(b"CCC"), "image/png"))]
    r = c.post("/api/image-datasets/upload", files=files)
    wid = r.json()["working_set_id"]
    ids = [img["id"] for img in r.json()["images"]]
    assert len(ids) == 3

    # remove 2 of the 3 images via the DELETE endpoint (mirrors the UI's Remove button)
    r_del1 = c.delete(f"/api/image-datasets/working/{wid}/{ids[0]}")
    assert r_del1.status_code == 200 and r_del1.json().get("ok")
    r_del2 = c.delete(f"/api/image-datasets/working/{wid}/{ids[1]}")
    assert r_del2.status_code == 200 and r_del2.json().get("ok")

    # removing an id that's already gone (or never existed) 404s, not 200
    r_del_again = c.delete(f"/api/image-datasets/working/{wid}/{ids[0]}")
    assert r_del_again.status_code == 404

    # validate must only see the one remaining image
    r_val = c.post("/api/image-datasets/validate", json={"working_set_id": wid, "captions": {}})
    assert r_val.status_code == 200 and r_val.json()["total"] == 1

    # save must only write the one remaining image
    r_save = c.post("/api/image-datasets", json={"working_set_id": wid, "name": "remove-test",
                                                  "trigger_word": "", "captions": {}})
    assert r_save.status_code == 200 and r_save.json().get("ok")
    loaded = idr.ImageDatasetStore(base_dir=str(tmp_path / "saved")).load("remove-test")
    assert len(loaded["images"]) == 1


def test_second_upload_with_same_working_set_id_accumulates(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))
    c = _client(monkeypatch)

    r1 = c.post("/api/image-datasets/upload",
                files=[("files", ("a.png", io.BytesIO(b"AAA"), "image/png"))])
    wid = r1.json()["working_set_id"]
    assert len(r1.json()["images"]) == 1

    # a second upload call passing the SAME working_set_id must ADD to it, not
    # orphan the first batch into a separate, now-unreachable working set.
    r2 = c.post("/api/image-datasets/upload",
                files=[("files", ("b.png", io.BytesIO(b"BBB"), "image/png"))],
                data={"working_set_id": wid})
    assert r2.status_code == 200
    assert r2.json()["working_set_id"] == wid

    # both batches' images are visible to the server -- reflected in /validate.
    r_val = c.post("/api/image-datasets/validate", json={"working_set_id": wid, "captions": {}})
    assert r_val.status_code == 200 and r_val.json()["total"] == 2


def test_router_is_admin_gated():
    # Deliberately does NOT monkeypatch require_admin -- every other test in
    # this file does, so none of them actually prove the router requires
    # admin. This test inspects the real APIRouter object's dependency wiring.
    # Note: `fastapi.Depends` is a FACTORY FUNCTION (returns a `params.Depends`
    # instance), not a class -- isinstance() needs the real class from
    # fastapi.params.
    from fastapi.params import Depends as DependsClass
    router = idr.setup_image_dataset_routes()
    assert any(
        isinstance(dep, DependsClass) and dep.dependency is idr.require_admin
        for dep in router.dependencies
    )
    # confirm the dependency actually propagates to every individual route,
    # not just the router object itself
    assert router.routes
    for route in router.routes:
        assert any(
            isinstance(dep, DependsClass) and dep.dependency is idr.require_admin
            for dep in route.dependencies
        ), f"route {route.path} missing require_admin dependency"
