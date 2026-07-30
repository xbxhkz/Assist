from src.settings import DEFAULT_SETTINGS, _PER_USER_KEYS


def test_dataset_generation_keys_registered_with_empty_default():
    assert DEFAULT_SETTINGS.get("dataset_generation_endpoint_id") == ""
    assert DEFAULT_SETTINGS.get("dataset_generation_model") == ""


def test_dataset_generation_keys_not_per_user():
    # admin-only feature -- no per-user override needed
    assert "dataset_generation_endpoint_id" not in _PER_USER_KEYS
    assert "dataset_generation_model" not in _PER_USER_KEYS
