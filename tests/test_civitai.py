import src.imagemodels.civitai as civitai


def test_search_flattens_primary_version_and_file():
    sample = {"items": [{"id": 1, "name": "Anime Style", "modelVersions": [
        {"id": 11, "baseModel": "SDXL 1.0", "trainedWords": ["anmstyle"],
         "files": [{"name": "anime.safetensors", "downloadUrl": "http://c/dl/11",
                    "sizeKB": 1234, "primary": True}]}]}]}
    res = civitai.search("anime", get=lambda u, p, h: sample)
    assert res[0]["name"] == "Anime Style"
    assert res[0]["base_model"] == "SDXL 1.0"
    assert res[0]["trigger_words"] == ["anmstyle"]
    assert res[0]["download_url"] == "http://c/dl/11"
    assert res[0]["file_name"] == "anime.safetensors"
    assert res[0]["size_kb"] == 1234


def test_search_tolerates_missing_fields():
    res = civitai.search("x", get=lambda u, p, h: {"items": [{"id": 2, "name": "Bare"}]})
    assert res[0]["download_url"] == "" and res[0]["trigger_words"] == []
    assert civitai.search("x", get=lambda u, p, h: {}) == []


def test_search_passes_lora_type_and_query():
    seen = {}
    def get(u, p, h):
        seen["params"] = p
        return {"items": []}
    civitai.search("dog", limit=5, get=get)
    assert seen["params"]["types"] == "LORA"
    assert seen["params"]["query"] == "dog" and seen["params"]["limit"] == 5


def test_download_url_with_token():
    assert civitai.download_url_with_token("http://c/dl", "T") == "http://c/dl?token=T"
    assert civitai.download_url_with_token("http://c/dl?x=1", "T") == "http://c/dl?x=1&token=T"
    assert civitai.download_url_with_token("http://c/dl", "") == "http://c/dl"
