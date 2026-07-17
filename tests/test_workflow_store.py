import pytest
import src.workflows.store as st


def test_safe_id_rejects_traversal():
    for bad in ["a/b", "a\\b", "..", "../x", ""]:
        with pytest.raises(ValueError):
            st._safe_id(bad)
    assert st._safe_id("my-flow") == "my-flow"


def test_safe_id_rejects_drive_letters_and_ads():
    # Bare Windows drive letters basename() to "" -- must be rejected, not
    # silently accepted as an empty id.
    for bad in ["C:", "Z:", "D:evil", "foo:bar"]:
        with pytest.raises(ValueError):
            st._safe_id(bad)


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


def test_save_dedupes_slug_collision_from_different_names(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    first = st.save_workflow({"name": "Flow#1", "nodes": [], "edges": []})
    second = st.save_workflow({"name": "Flow!1", "nodes": [], "edges": []})
    assert first["id"] == "flow-1"
    assert second["id"] != first["id"]
    assert second["id"] == "flow-1-2"
    # Both must survive on disk under distinct files.
    assert st.get_workflow(first["id"])["name"] == "Flow#1"
    assert st.get_workflow(second["id"])["name"] == "Flow!1"
    ids = {w["id"] for w in st.list_workflows()}
    assert ids == {"flow-1", "flow-1-2"}


def test_save_with_explicit_id_still_overwrites_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    st.save_workflow({"id": "flow1", "name": "Original", "nodes": [], "edges": []})
    updated = st.save_workflow({"id": "flow1", "name": "Updated", "nodes": [1], "edges": []})
    assert updated["id"] == "flow1"
    assert st.get_workflow("flow1")["name"] == "Updated"
    assert [w["id"] for w in st.list_workflows()] == ["flow1"]


def test_list_workflows_skips_non_dict_and_garbage_json(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    st.save_workflow({"id": "good", "name": "Good Flow", "nodes": [], "edges": []})
    (tmp_path / "not-a-dict.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "garbage.json").write_text("{not valid json", encoding="utf-8")
    result = st.list_workflows()
    assert [w["id"] for w in result] == ["good"]


def test_get_workflow_returns_none_for_non_dict_json(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    (tmp_path / "not-a-dict.json").write_text("[1, 2]", encoding="utf-8")
    assert st.get_workflow("not-a-dict") is None
