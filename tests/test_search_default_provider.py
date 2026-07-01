"""The out-of-the-box search provider must need no API key and no service,
so a fresh native (container-free) install can search the web immediately.
"""
from src.settings import DEFAULT_SETTINGS
from services.search.providers import PROVIDER_INFO


def test_default_search_provider_is_keyless_and_serviceless():
    provider = DEFAULT_SETTINGS["search_provider"]
    label, needs_key, needs_url = PROVIDER_INFO[provider]
    assert needs_key is False, f"default provider {provider!r} requires an API key"
    assert needs_url is False, f"default provider {provider!r} requires a service URL"


def test_default_search_provider_is_duckduckgo():
    assert DEFAULT_SETTINGS["search_provider"] == "duckduckgo"
