"""Contract tests for the admin-gated /api/workflows router.

Exercises the real router + real JSON store (redirected to a tmp dir) with the
admin gate overridden, plus the run endpoint against a mocked engine. Covers the
status-code contract the route is responsible for: invalid graph / unsafe id ->
400, missing workflow -> 404, and a runtime NODE failure -> 200 with the failure
in the returned log (a node failing is a run result, not a request error)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.workflow_routes as wr
import src.workflows.store as st
from core.middleware import require_admin
from src.workflows.model import WorkflowError


def _valid_wf(wid="flow1", name="Flow One"):
    return {"id": wid, "name": name,
            "nodes": [{"id": "i", "type": "input", "config": {"name": "q"}},
                      {"id": "o", "type": "output", "config": {"name": "answer"}}],
            "edges": [{"from_node": "i", "from_port": "value",
                       "to_node": "o", "to_port": "value"}]}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    app = FastAPI()
    app.include_router(wr.setup_workflow_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app)


def test_router_gate_is_require_admin_by_identity():
    # Truthiness alone would pass with any no-op dependency — assert the real gate.
    router = wr.setup_workflow_routes()
    assert router.prefix == "/api/workflows"
    assert router.dependencies[0].dependency is require_admin
    paths = {r.path for r in router.routes}
    assert paths == {"/api/workflows", "/api/workflows/{wid}", "/api/workflows/{wid}/run"}


def test_crud_round_trip(client):
    assert client.post("/api/workflows", json=_valid_wf()).json()["id"] == "flow1"
    assert client.get("/api/workflows/flow1").json()["name"] == "Flow One"
    assert [w["id"] for w in client.get("/api/workflows").json()["workflows"]] == ["flow1"]
    assert client.delete("/api/workflows/flow1").json() == {"deleted": True}
    assert client.get("/api/workflows/flow1").status_code == 404


def test_post_invalid_graph_is_400_with_errors_list(client):
    cyclic = {"id": "c", "name": "C",
              "nodes": [{"id": "t1", "type": "template", "config": {"template": "{x}"}},
                        {"id": "t2", "type": "template", "config": {"template": "{y}"}}],
              "edges": [{"from_node": "t1", "from_port": "text", "to_node": "t2", "to_port": "y"},
                        {"from_node": "t2", "from_port": "text", "to_node": "t1", "to_port": "x"}]}
    r = client.post("/api/workflows", json=cyclic)
    assert r.status_code == 400
    assert isinstance(r.json()["detail"]["errors"], list)


def test_post_malformed_shape_is_400_not_500(client):
    # {"nodes": [1]} once crashed validate with AttributeError -> 500; must be 400.
    r = client.post("/api/workflows", json={"name": "bad", "nodes": [1], "edges": []})
    assert r.status_code == 400
    assert r.json()["detail"]["errors"]


def test_unsafe_and_wrong_type_ids_are_400_not_500(client):
    valid = _valid_wf()
    # ValueError (path-unsafe id via ':'), TypeError (non-str id), OSError (invalid char).
    assert client.post("/api/workflows", json={**valid, "id": "a:b"}).status_code == 400
    assert client.post("/api/workflows", json={**valid, "id": 5}).status_code == 400
    assert client.post("/api/workflows", json={**valid, "id": "*"}).status_code == 400


def test_missing_workflow_is_404(client):
    assert client.get("/api/workflows/nope").status_code == 404
    assert client.post("/api/workflows/nope/run", json={"inputs": {}}).status_code == 404


def test_run_returns_outputs_and_log(client, monkeypatch):
    client.post("/api/workflows", json=_valid_wf())

    async def fake_run(wf, inputs, ctx, **kw):
        return {"outputs": {"answer": "hi"}, "log": [{"node": "o", "status": "ok"}]}
    monkeypatch.setattr(wr, "run_workflow", fake_run)

    r = client.post("/api/workflows/flow1/run", json={"inputs": {"q": "x"}})
    assert r.status_code == 200
    assert r.json() == {"outputs": {"answer": "hi"}, "log": [{"node": "o", "status": "ok"}]}


def test_run_node_failure_is_200_with_error_in_log(client, monkeypatch):
    client.post("/api/workflows", json=_valid_wf())

    async def fake_run(wf, inputs, ctx, **kw):
        return {"outputs": {}, "log": [{"node": "l", "status": "error", "error": "model down"}]}
    monkeypatch.setattr(wr, "run_workflow", fake_run)

    r = client.post("/api/workflows/flow1/run", json={"inputs": {}})
    assert r.status_code == 200          # a node failing is a run result, not a request error
    assert r.json()["log"][0]["status"] == "error"


def test_run_invalid_graph_is_400(client, monkeypatch):
    client.post("/api/workflows", json=_valid_wf())

    async def fake_run(wf, inputs, ctx, **kw):
        raise WorkflowError(["cycle detected in workflow graph"])
    monkeypatch.setattr(wr, "run_workflow", fake_run)

    r = client.post("/api/workflows/flow1/run", json={"inputs": {}})
    assert r.status_code == 400
    assert r.json()["detail"]["errors"] == ["cycle detected in workflow graph"]
