"""Equipment-manual knowledge base: a dedicated ChromaDB collection (isolated from
the personal-docs RAG) of ingested manuals/datasheets, searchable with page-level
citations. Built on the existing offline embedding-lane machinery. READ operations
never raise; every method degrades to an empty/error result when the store is
unhealthy. Admin-only at the tool layer."""
import hashlib
import logging
import os

from src.embedding_lanes import build_embedding_lanes, dedupe_results, query_lanes
from src.rag_vector import split_into_chunks

logger = logging.getLogger(__name__)

MANUAL_COLLECTION = "equipment_manuals"
_PLAIN_EXTS = {".txt", ".md", ".json"}


def _pdf_pages(source):
    """Yield (1-based page_number, page_text) for each page of a text-layer PDF."""
    from pypdf import PdfReader
    reader = PdfReader(source)
    for i, page in enumerate(reader.pages):
        yield (i + 1, page.extract_text() or "")


def _read_document_text(source, ext):
    from src.personal_docs import extract_office_text, read_text_file
    from src.markitdown_runtime import MARKITDOWN_EXTS
    if ext in _PLAIN_EXTS:
        return read_text_file(source)
    if ext in MARKITDOWN_EXTS:
        return extract_office_text(source)
    return None  # unsupported


class ManualStore:
    def __init__(self, base_name: str = MANUAL_COLLECTION, lanes=None):
        self.base_name = base_name
        self._lanes = lanes if lanes is not None else build_embedding_lanes(base_name)

    @property
    def healthy(self) -> bool:
        return bool(self._lanes)

    # ------------------------------------------------------------------
    def ingest_file(self, path, title=None, *, extract_pages=None) -> dict:
        if not self.healthy:
            return {"error": "manual store unavailable"}
        source = os.path.abspath(path)
        ext = os.path.splitext(source)[1].lower()
        title = title or os.path.splitext(os.path.basename(source))[0]
        manual_id = "man_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]

        if ext == ".pdf":
            try:
                pages = list((extract_pages or _pdf_pages)(source))
            except Exception as e:
                return {"error": f"could not read PDF: {e}"}
        else:
            text = _read_document_text(source, ext)
            if text is None:
                return {"error": f"unsupported manual type: {ext or '(none)'}"}
            pages = [(0, text)]  # 0 = non-paginated sentinel

        chunks_indexed = 0
        page_numbers = set()
        for page, text in pages:
            if not text or not text.strip():
                continue
            page_val = int(page) if isinstance(page, int) else 0
            if page_val >= 1:
                page_numbers.add(page_val)
            for chunk_id, chunk in enumerate(split_into_chunks(text)):
                meta = {
                    "manual_id": manual_id, "title": title, "source": source,
                    "page": page_val, "chunk_id": chunk_id, "kind": "manual",
                }
                if self._add(chunk, meta):
                    chunks_indexed += 1

        if chunks_indexed == 0:
            return {"error": "no extractable text in manual"}
        return {
            "manual_id": manual_id, "title": title,
            "pages": len(page_numbers), "chunks_indexed": chunks_indexed,
        }

    def _add(self, text: str, meta: dict) -> bool:
        doc_id = "man_" + hashlib.sha256(
            f"{meta['manual_id']}\x00{meta['page']}\x00{meta['chunk_id']}".encode("utf-8")
        ).hexdigest()[:16]
        wrote = False
        for lane in self._lanes:
            try:
                if lane.collection.get(ids=[doc_id]).get("ids"):
                    wrote = True
                    continue
                lane.collection.add(
                    ids=[doc_id], embeddings=lane.encode([text]),
                    documents=[text], metadatas=[meta],
                )
                wrote = True
            except Exception as e:
                logger.warning("manual _add failed in %s lane: %s", lane.name, e)
        return wrote

    # ------------------------------------------------------------------
    def list_manuals(self) -> list:
        if not self.healthy:
            return []
        try:
            data = self._lanes[0].collection.get(include=["metadatas"])
        except Exception as e:
            logger.warning("list_manuals failed: %s", e)
            return []
        manuals = {}
        for meta in data.get("metadatas") or []:
            if not isinstance(meta, dict):
                continue
            mid = meta.get("manual_id")
            if not mid:
                continue
            m = manuals.setdefault(mid, {
                "manual_id": mid, "title": meta.get("title"),
                "source": meta.get("source"), "chunk_count": 0,
            })
            m["chunk_count"] += 1
        return list(manuals.values())

    def remove_manual(self, manual_id) -> dict:
        if not self.healthy:
            return {"removed_count": 0}
        removed = set()
        for lane in self._lanes:
            try:
                res = lane.collection.get(where={"manual_id": manual_id}, include=[])
                ids = res.get("ids") or []
                if ids:
                    lane.collection.delete(ids=ids)
                    removed.update(ids)
            except Exception as e:
                logger.warning("remove_manual failed in %s lane: %s", lane.name, e)
        return {"removed_count": len(removed)}

    # ------------------------------------------------------------------
    def search(self, query, k: int = 5, manual_id=None) -> list:
        if not self.healthy or not query or not isinstance(query, str):
            return []
        where = {"manual_id": manual_id} if manual_id else None
        candidates = []
        try:
            for lane, results in query_lanes(
                self._lanes, query,
                n_results=lambda lane: min(max(k, 20), lane.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            ):
                for idx in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][idx] or {}
                    distance = results["distances"][0][idx]
                    page = meta.get("page")
                    page = page if isinstance(page, int) and page >= 1 else None
                    candidates.append({
                        "id": results["ids"][0][idx],
                        "title": meta.get("title"),
                        "page": page,
                        "source": meta.get("source"),
                        "snippet": results["documents"][0][idx],
                        "chunk_id": meta.get("chunk_id"),
                        "score": round(1.0 - distance, 4),
                    })
        except Exception as e:
            logger.warning("manual search failed: %s", e)
            return []
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return dedupe_results(candidates, limit=k)


_manual_store = None


def get_manual_store():
    """Lazy singleton. Returns a ManualStore (possibly unhealthy — its methods then
    return empty/error results) or None if construction raised."""
    global _manual_store
    if _manual_store is None:
        try:
            _manual_store = ManualStore()
        except Exception as e:
            logger.error("get_manual_store init failed: %s", e)
            return None
    return _manual_store
