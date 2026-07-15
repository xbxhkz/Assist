from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.hwfit_routes as hr
import services.hwfit.hardware as hw


def test_usage_route(monkeypatch):
    monkeypatch.setattr(hw, "usage", lambda: {
        "cpu_percent": 5.0, "ram_used_gb": 1.0, "ram_total_gb": 8.0,
        "ram_percent": 12.5, "gpus": []})
    app = FastAPI()
    app.include_router(hr.setup_hwfit_routes())
    r = TestClient(app).get("/api/hwfit/usage")
    assert r.status_code == 200
    j = r.json()
    assert j["cpu_percent"] == 5.0 and j["ram_total_gb"] == 8.0 and j["gpus"] == []
