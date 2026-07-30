import asyncio
from src.dataset_tools.ground import chunk_document, generate_grounded, library_chunks, document_chunks


def _run(coro):
    return asyncio.run(coro)


def test_chunk_document_labels_and_splits():
    chunks = chunk_document([(1, "hello"), (2, ""), (3, "x" * 4500)], max_chars=2000)
    assert chunks[0] == {"source": "p.1", "text": "hello"}          # page 1
    assert all(c["source"] != "p.2" for c in chunks)                # blank page skipped
    p3 = [c for c in chunks if c["source"] == "p.3"]
    assert len(p3) == 3 and all(len(c["text"]) <= 2000 for c in p3)  # long page split


def test_chunk_document_never_raises_on_garbage():
    assert chunk_document(None) == []
    assert chunk_document([("bad",), 42, (1, 5)]) == []  # non-str text / bad tuples skipped


def test_generate_grounded_tags_source_and_stops_at_count():
    async def fake(prompt, system=None):
        # echo a row that embeds which source text it saw so we can assert grounding
        return '{"text": "row"}'
    chunks = [{"source": "p.1", "text": "AAA"}, {"source": "p.2", "text": "BBB"}]
    rep = _run(generate_grounded(chunks, "text", 1, model_call=fake, per_chunk=4, batch_size=4))
    assert rep["produced"] == 1 and rep["requested"] == 1
    assert rep["rows"][0]["source"] == "p.1"          # tagged
    assert rep["chunks_used"] == 1                     # stopped after count reached


def test_generate_grounded_dedups_across_chunks():
    async def fake(prompt, system=None):
        return '{"text": "same"}'                      # every chunk yields the same row
    chunks = [{"source": "p.1", "text": "A"}, {"source": "p.2", "text": "B"}]
    rep = _run(generate_grounded(chunks, "text", 5, model_call=fake, per_chunk=1, batch_size=1))
    assert rep["produced"] == 1 and rep["duplicates"] >= 1  # 2nd chunk's row is a dup


def test_generate_grounded_model_error_and_empty_never_raise():
    async def boom(prompt, system=None):
        raise RuntimeError("no endpoint")
    rep = _run(generate_grounded([{"source": "p.1", "text": "A"}], "text", 3, model_call=boom))
    assert rep["produced"] == 0 and "error" in rep
    rep2 = _run(generate_grounded([], "text", 3, model_call=boom))
    assert rep2["produced"] == 0 and rep2["chunks_used"] == 0


class _FakeStore:
    def __init__(self, hits): self._hits = hits
    def search(self, query, k=8): return self._hits


def test_library_chunks_labels_title_and_page():
    store = _FakeStore([
        {"title": "VFD Manual", "page": 42, "snippet": "E12 = overpressure trip"},
        {"title": "VFD Manual", "page": None, "snippet": "general notes"},
        {"title": "X", "page": 1, "snippet": "   "},           # blank -> skipped
    ])
    chunks = library_chunks(store, "overpressure", k=8)
    assert chunks[0] == {"source": "VFD Manual, p.42", "text": "E12 = overpressure trip"}
    assert chunks[1]["source"] == "VFD Manual"                 # no page
    assert all(c["text"].strip() for c in chunks) and len(chunks) == 2


def test_library_chunks_never_raises():
    assert library_chunks(None, "q") == []
    assert library_chunks(_FakeStore("not-a-list"), "q") == []
    assert library_chunks(_FakeStore([]), "") == []           # empty query


def test_document_chunks_pdf_and_text_and_unsupported():
    def fake_pdf(path):
        return [(1, "page one text"), (2, "page two text")]
    pdf = document_chunks("f.pdf", ".pdf", pdf_pages=fake_pdf, filename="Guide.pdf")
    assert pdf[0] == {"source": "Guide.pdf p.1", "text": "page one text"}
    def fake_read(path, ext):
        return "plain body"
    txt = document_chunks("f.txt", ".txt", read_text=fake_read, filename="notes.txt")
    assert txt[0]["source"] == "notes.txt p.1" and txt[0]["text"] == "plain body"
    def none_read(path, ext):
        return None
    assert document_chunks("f.xyz", ".xyz", read_text=none_read) == []   # unsupported
