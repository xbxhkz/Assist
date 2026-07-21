def test_split_into_chunks_module_level_splits_long_text():
    from src.rag_vector import split_into_chunks
    text = "This is a sentence. " * 200  # ~4000 chars
    chunks = split_into_chunks(text, chunk_size=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)  # chunk_size + a little slack
    assert split_into_chunks("") == []
    assert split_into_chunks("short") == ["short"]


import pytest
from tests.helpers.embedding_lanes import FakeChroma, FakeEmbedder, patch_chroma


@pytest.fixture
def store(monkeypatch):
    """A ManualStore whose lanes run on an in-memory FakeChroma + a fake embedder,
    exercising the real build_embedding_lanes/query_lanes machinery with no disk
    or model load. Only the fastembed lane exists (custom lane forced unavailable)."""
    import src.embedding_lanes as el
    patch_chroma(monkeypatch, FakeChroma())
    monkeypatch.setattr(el, "_build_fastembed_client", lambda: FakeEmbedder(8, "fake-embed", ""))

    def _no_custom():
        raise RuntimeError("no custom lane in tests")
    monkeypatch.setattr(el, "_build_custom_client", _no_custom)

    from src.industrial.manual_store import ManualStore
    return ManualStore(base_name="equipment_manuals_test")


def _fake_pages(pairs):
    def _extract(_source):
        return list(pairs)
    return _extract


def test_ingest_pdf_sets_per_page_metadata(store):
    res = store.ingest_file(
        "C:/manuals/vfd.pdf", title="VFD Manual",
        extract_pages=_fake_pages([(1, "F0002 overcurrent during accel."),
                                   (2, "Check motor cabling and load.")]),
    )
    assert res["manual_id"].startswith("man_")
    assert res["title"] == "VFD Manual"
    assert res["pages"] == 2
    assert res["chunks_indexed"] >= 2
    hits = store.search("overcurrent cabling", k=20, manual_id=res["manual_id"])
    assert {h["page"] for h in hits} == {1, 2}
    assert all(h["title"] == "VFD Manual" for h in hits)


def test_ingest_text_file_page_is_none(store, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Belt tension spec is 40 N. Grease every 500 hours.", encoding="utf-8")
    res = store.ingest_file(str(p), title="Notes")
    assert res["chunks_indexed"] >= 1
    hits = store.search("grease", k=20, manual_id=res["manual_id"])
    assert hits and all(h["page"] is None for h in hits)


def test_remove_manual_deletes_only_that_manual(store):
    a = store.ingest_file("C:/m/a.pdf", title="A", extract_pages=_fake_pages([(1, "alpha")]))
    b = store.ingest_file("C:/m/b.pdf", title="B", extract_pages=_fake_pages([(1, "bravo")]))
    assert store.remove_manual(a["manual_id"])["removed_count"] >= 1
    assert store.search("alpha", k=20, manual_id=a["manual_id"]) == []
    assert store.search("bravo", k=20, manual_id=b["manual_id"])  # B untouched
    ids = {m["manual_id"] for m in store.list_manuals()}
    assert b["manual_id"] in ids and a["manual_id"] not in ids


def test_reingest_same_file_is_idempotent(store):
    pages = _fake_pages([(1, "alpha overcurrent"), (2, "bravo cabling")])
    store.ingest_file("C:/m/a.pdf", title="A", extract_pages=pages)
    count_after_first = store._lanes[0].count()
    store.ingest_file("C:/m/a.pdf", title="A", extract_pages=pages)
    assert store._lanes[0].count() == count_after_first


def test_search_and_list_safe_when_empty(store):
    assert store.search("anything") == []
    assert store.list_manuals() == []
