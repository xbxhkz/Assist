from src.dataset_tools.store import DatasetStore


def test_save_load_list_delete(tmp_path):
    s = DatasetStore(base_dir=str(tmp_path))
    out = s.save("my set", [{"text": "a"}, {"text": "b"}])
    assert out.get("ok") and out["name"] == "my-set" and out["path"].endswith("my-set.jsonl")
    lst = s.list()
    assert lst and lst[0]["name"] == "my-set" and lst[0]["rows"] == 2
    loaded = s.load("my-set")
    assert loaded["rows"] == [{"text": "a"}, {"text": "b"}]
    assert s.delete("my-set").get("ok") and s.list() == []


def test_name_sanitized_no_traversal(tmp_path):
    s = DatasetStore(base_dir=str(tmp_path))
    out = s.save("../../evil", [{"text": "x"}])
    # written inside base_dir under a sanitized basename, never escaping
    assert out.get("ok") and str(tmp_path) in out["path"] and ".." not in out["name"]


def test_empty_rows_and_bad_name_rejected(tmp_path):
    s = DatasetStore(base_dir=str(tmp_path))
    assert "error" in s.save("ok", [])
    assert "error" in s.save("", [{"text": "x"}])


def test_load_missing(tmp_path):
    assert "error" in DatasetStore(base_dir=str(tmp_path)).load("nope")
