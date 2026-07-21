import asyncio
import json

import src.agent_tools.industrial_manuals as im


def _run(coro):
    return asyncio.run(coro)


class FakeStore:
    def __init__(self):
        self.ingested = []
        self.removed = []
    def ingest_file(self, path, title=None):
        self.ingested.append((path, title))
        return {"manual_id": "man_abc", "title": title or "x", "pages": 2, "chunks_indexed": 5}
    def list_manuals(self):
        return [{"manual_id": "man_abc", "title": "VFD", "source": "C:/m/vfd.pdf", "chunk_count": 5}]
    def remove_manual(self, manual_id):
        self.removed.append(manual_id)
        return {"removed_count": 3}


def _exec(content, store=None, ctx=None):
    store = store or FakeStore()
    out = _run(im.ingest_equipment_manual(content, ctx or {"owner": "admin"}, store=store))
    return out, store


def test_add_happy_path(tmp_path):
    f = tmp_path / "vfd.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    out, store = _exec(json.dumps({"action": "add", "path": str(f), "title": "VFD"}))
    assert out["output"]["manual_id"] == "man_abc"
    assert store.ingested == [(str(f), "VFD")]


def test_add_missing_file_is_error():
    out, _ = _exec(json.dumps({"action": "add", "path": "C:/nope/missing.pdf"}))
    assert "error" in out


def test_add_non_str_path_is_error():
    out, _ = _exec(json.dumps({"action": "add", "path": 5}))
    assert "error" in out


def test_add_null_byte_path_never_raises():
    out, _ = _exec(json.dumps({"action": "add", "path": "C:/x" + chr(0) + ".pdf"}))
    assert "error" in out


def test_list_action():
    out, _ = _exec(json.dumps({"action": "list"}))
    assert out["output"]["manuals"][0]["manual_id"] == "man_abc"


def test_remove_action():
    out, store = _exec(json.dumps({"action": "remove", "manual_id": "man_abc"}))
    assert out["output"]["removed_count"] == 3 and store.removed == ["man_abc"]


def test_remove_missing_id_is_error():
    out, _ = _exec(json.dumps({"action": "remove"}))
    assert "error" in out


def test_bad_json_is_error():
    out, _ = _exec("not json")
    assert "error" in out


def test_unknown_action_is_error():
    out, _ = _exec(json.dumps({"action": "explode"}))
    assert "error" in out


def test_store_raising_never_raises():
    class Boom(FakeStore):
        def list_manuals(self):
            raise RuntimeError("chroma down")
    out, _ = _exec(json.dumps({"action": "list"}), store=Boom())
    assert "error" in out and "chroma down" in out["error"]
