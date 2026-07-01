"""Phase 1 (de-Docker): with the keyless defaults and no chromadb/searxng/ntfy
services running, the consolidated health report must be healthy — SearXNG and
ntfy report `disabled`, which is excluded from the overall verdict.
"""
import pytest

import src.service_health as sh
from src.settings import DEFAULT_SETTINGS


def test_searxng_disabled_under_default_provider():
    # Default provider is no longer searxng, so its probe self-disables and
    # never performs a network call.
    result = sh.searxng_health(dict(DEFAULT_SETTINGS))
    assert result["status"] == sh.DISABLED


def test_ntfy_disabled_with_no_integration():
    result = sh.ntfy_health([], dict(DEFAULT_SETTINGS))
    assert result["status"] == sh.DISABLED


@pytest.mark.asyncio
async def test_overall_health_ok_with_no_containers(monkeypatch):
    # No settings/integrations/accounts/endpoints and no vector managers:
    # nothing configured means nothing can be "down".
    monkeypatch.setattr(sh, "_gather_inputs", lambda: {
        "settings": dict(DEFAULT_SETTINGS),
        "integrations": [],
        "accounts": [],
        "endpoints": [],
    })
    report = await sh.collect_service_health(rag_manager=None, memory_vector=None)
    assert report["overall"] == sh.OK
    statuses = {s["name"]: s["status"] for s in report["services"]}
    assert statuses["searxng"] == sh.DISABLED
    assert statuses["ntfy"] == sh.DISABLED
    assert statuses["chromadb"] == sh.DISABLED  # no vector managers passed
