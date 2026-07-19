import base64

import src.document_processor as dp


def _tiny_png(tmp_path):
    # 1x1 PNG — enough for open()/base64 in the function under test
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    p = tmp_path / "img.png"
    p.write_bytes(png)
    return str(p)


def _stub_vl(monkeypatch, capture):
    # Make the function reach message construction + our fake llm_call without a network call.
    monkeypatch.setattr(dp, "_load_vl_settings", lambda: {"vision_enabled": True, "vision_model": "vl"})
    monkeypatch.setattr(dp, "_resolve_vl_model", lambda m, owner=None: ("http://x", "vl-model", {}))
    monkeypatch.setattr(dp, "resolve_vision_fallback_candidates", lambda owner=None: [], raising=False)

    def fake_llm_call(url, model, messages, headers=None, timeout=None):
        capture["messages"] = messages
        return "VL-REPLY"
    monkeypatch.setattr(dp, "llm_call", fake_llm_call)


def test_custom_prompt_is_sent_to_the_model(tmp_path, monkeypatch):
    cap = {}
    _stub_vl(monkeypatch, cap)
    out = dp.analyze_image_with_vl_result(_tiny_png(tmp_path), owner="admin",
                                          prompt="You are a controls expert. Diagnose this.")
    assert out == {"text": "VL-REPLY", "model": "vl-model"}
    text_part = cap["messages"][0]["content"][0]["text"]
    assert text_part == "You are a controls expert. Diagnose this."


def test_default_prompt_preserved_when_absent(tmp_path, monkeypatch):
    cap = {}
    _stub_vl(monkeypatch, cap)
    dp.analyze_image_with_vl_result(_tiny_png(tmp_path), owner="admin")
    assert cap["messages"][0]["content"][0]["text"] == "Describe this image in detail"
