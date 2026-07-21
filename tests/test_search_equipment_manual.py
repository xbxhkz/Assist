import asyncio
import json

import src.agent_tools.industrial_manuals as im


def _run(coro):
    return asyncio.run(coro)


class FakeStore:
    def __init__(self, hits=None):
        self._hits = hits if hits is not None else [
            {"title": "VFD Manual", "page": 42, "source": "C:/m/vfd.pdf",
             "snippet": "F0002 = overcurrent during acceleration.", "chunk_id": 3, "score": 0.9},
            {"title": "Notes", "page": None, "source": "C:/m/notes.txt",
             "snippet": "Grease every 500 hours.", "chunk_id": 0, "score": 0.5},
        ]
        self.calls = []
    def search(self, query, k=5, manual_id=None):
        self.calls.append((query, k, manual_id))
        return self._hits


def _exec(content, store=None, ctx=None):
    store = store or FakeStore()
    out = _run(im.search_equipment_manual(content, ctx or {"owner": "admin"}, store=store))
    return out, store


def test_search_returns_page_citation():
    out, store = _exec(json.dumps({"query": "overcurrent"}))
    cites = out["output"]["citations"]
    assert "VFD Manual, p.42" in cites
    assert "Notes (excerpt 1)" in cites  # page-less -> excerpt (chunk_id 0 -> 1)
    assert out["output"]["results"][0]["page"] == 42
    assert store.calls == [("overcurrent", 5, None)]


def test_k_clamped_and_bool_rejected():
    out, store = _exec(json.dumps({"query": "x", "k": True}))
    assert store.calls[0][1] == 5  # bool k ignored -> default 5
    _exec_out, store2 = _exec(json.dumps({"query": "x", "k": 999}), store=FakeStore())
    assert store2.calls[0][1] == 20  # clamped to max 20


def test_manual_id_filter_passed_through():
    out, store = _exec(json.dumps({"query": "x", "manual_id": "man_abc"}))
    assert store.calls[0][2] == "man_abc"


def test_missing_query_is_error():
    out, _ = _exec(json.dumps({"k": 5}))
    assert "error" in out


def test_non_str_query_is_error():
    out, _ = _exec(json.dumps({"query": 5}))
    assert "error" in out


def test_no_results_is_clean_output_not_error():
    out, _ = _exec(json.dumps({"query": "nothing"}), store=FakeStore(hits=[]))
    assert "error" not in out
    assert out["output"]["results"] == []
    assert "No matching" in out["output"]["message"]


def test_bad_json_is_error():
    out, _ = _exec("not json")
    assert "error" in out


def test_store_raising_never_raises():
    class Boom(FakeStore):
        def search(self, query, k=5, manual_id=None):
            raise RuntimeError("chroma down")
    out, _ = _exec(json.dumps({"query": "x"}), store=Boom())
    assert "error" in out and "chroma down" in out["error"]
