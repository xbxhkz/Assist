"""Unit tests for the HF GGUF catalog service (network injected)."""
import src.localmodels.catalog as cat


def test_search_builds_url_and_parses(monkeypatch):
    seen = {}
    def fake_get_json(url, headers):
        seen["url"] = url
        return [
            {"id": "TheBloke/foo-GGUF", "downloads": 10, "likes": 2},
            {"modelId": "bar/baz-gguf", "downloads": 5},
            {"downloads": 1},  # no id → skipped
        ]
    out = cat.search_gguf_models("qwen", sort="downloads", limit=7, get_json=fake_get_json)
    assert "filter=gguf" in seen["url"]
    assert "search=qwen" in seen["url"]
    assert "limit=7" in seen["url"]
    assert out == [
        {"repo": "TheBloke/foo-GGUF", "downloads": 10, "likes": 2},
        {"repo": "bar/baz-gguf", "downloads": 5, "likes": 0},
    ]


def test_search_clamps_bad_sort(monkeypatch):
    seen = {}
    def fake_get_json(url, headers):
        seen["url"] = url
        return []
    cat.search_gguf_models("x", sort="; rm -rf", get_json=fake_get_json)
    assert "sort=downloads" in seen["url"]  # unknown sort falls back


def test_search_network_error_returns_empty():
    def boom(url, headers):
        raise RuntimeError("network down")
    assert cat.search_gguf_models("x", get_json=boom) == []


def test_list_repo_files_filters_gguf_and_builds_url():
    def fake_get_json(url, headers):
        assert url.endswith("/api/models/acme/m/tree/main?recursive=1")
        return [
            {"type": "file", "path": "model-Q4_K_M.gguf", "size": 2000000000},
            {"type": "file", "path": "sub/model-Q8.gguf", "size": 3000000000},
            {"type": "file", "path": "README.md", "size": 100},
            {"type": "directory", "path": "sub"},
        ]
    out = cat.list_repo_gguf_files("acme/m", get_json=fake_get_json)
    assert out == [
        {"filename": "model-Q4_K_M.gguf", "size": 2000000000,
         "url": "https://huggingface.co/acme/m/resolve/main/model-Q4_K_M.gguf"},
        {"filename": "model-Q8.gguf", "size": 3000000000,
         "url": "https://huggingface.co/acme/m/resolve/main/sub/model-Q8.gguf"},
    ]


def test_hf_headers_present_when_token_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret123")
    assert cat._hf_headers() == {"Authorization": "Bearer secret123"}


def test_hf_headers_absent_when_no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert cat._hf_headers() == {}
