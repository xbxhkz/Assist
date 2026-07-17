import pytest
import src.workflows.store as st


def test_safe_id_rejects_traversal():
    for bad in ["a/b", "a\\b", "..", "../x", ""]:
        with pytest.raises(ValueError):
            st._safe_id(bad)
    assert st._safe_id("my-flow") == "my-flow"


def test_save_get_list_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    wf = {"id": "flow1", "name": "Flow One", "nodes": [], "edges": []}
    saved = st.save_workflow(wf)
    assert saved["id"] == "flow1"
    assert st.get_workflow("flow1")["name"] == "Flow One"
    assert [w["id"] for w in st.list_workflows()] == ["flow1"]
    assert st.delete_workflow("flow1") is True
    assert st.get_workflow("flow1") is None
    assert st.list_workflows() == []
    assert st.delete_workflow("flow1") is False


def test_save_slugifies_id_from_name_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    saved = st.save_workflow({"name": "My Cool Flow!", "nodes": [], "edges": []})
    assert saved["id"] == "my-cool-flow"
    assert st.get_workflow("my-cool-flow") is not None
