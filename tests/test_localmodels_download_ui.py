"""Text guards that the Phase 3b download UI is wired."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_search_and_progress_elements():
    html = _read("static/index.html")
    assert 'id="localmodels-search"' in html
    assert 'id="localmodels-results"' in html
    assert 'id="localmodels-progress"' in html


def test_js_calls_catalog_and_download_endpoints():
    js = _read("static/js/localModels.js")
    for ep in ("/api/localmodels/catalog/search", "/api/localmodels/catalog/files",
               "/api/localmodels/download", "/api/localmodels/download/status",
               "/api/localmodels/download/cancel"):
        assert ep in js, f"{ep} not called in localModels.js"
