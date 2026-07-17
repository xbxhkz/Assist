import routes.workflow_routes as wr


def test_router_is_admin_gated_and_prefixed():
    router = wr.setup_workflow_routes()
    assert router.prefix == "/api/workflows"
    assert router.dependencies, "router must carry the require_admin dependency"
    paths = {r.path for r in router.routes}
    assert "/api/workflows" in paths
    assert "/api/workflows/{wid}" in paths
    assert "/api/workflows/{wid}/run" in paths
