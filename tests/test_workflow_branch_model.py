import src.workflows.model as m


def _branch(cases, mode="match", nid="b"):
    return {"id": nid, "type": "branch", "config": {"mode": mode, "cases": cases}}


def test_branch_in_node_types():
    assert "branch" in m.NODE_TYPES


def test_branch_ports():
    b = _branch(["yes", "no"])
    assert m.input_ports(b) == ["value"]
    assert m.output_ports(b) == ["yes", "no", "else"]


def test_output_ports_crash_safe_on_bad_cases():
    assert m.output_ports({"type": "branch", "config": {"cases": "nope"}}) == ["else"]
    assert m.output_ports({"type": "branch", "config": {}}) == ["else"]
    assert m.output_ports({"type": "branch", "config": {"cases": [1, "", "ok"]}}) == ["ok", "else"]


def _wf(branch_cfg, extra_nodes=None, edges=None):
    nodes = [{"id": "i", "type": "input", "config": {"name": "q"}},
             {"id": "b", "type": "branch", "config": branch_cfg}]
    nodes += extra_nodes or []
    base_edges = [{"from_node": "i", "from_port": "value", "to_node": "b", "to_port": "value"}]
    return {"id": "w", "name": "W", "nodes": nodes, "edges": base_edges + (edges or [])}


def test_valid_branch_workflow_passes():
    wf = _wf({"mode": "match", "cases": ["yes", "no"]},
             extra_nodes=[{"id": "o", "type": "output", "config": {"name": "r"}}],
             edges=[{"from_node": "b", "from_port": "yes", "to_node": "o", "to_port": "value"}])
    assert m.validate(wf) == []


def test_validate_flags_bad_branch_config():
    assert any("cases" in e for e in m.validate(_wf({"mode": "match", "cases": []})))
    assert any("cases" in e for e in m.validate(_wf({"mode": "match", "cases": "x"})))
    assert any("duplicate case" in e for e in m.validate(_wf({"mode": "match", "cases": ["a", "a"]})))
    assert any("reserved" in e for e in m.validate(_wf({"mode": "match", "cases": ["else"]})))
    assert any("mode" in e for e in m.validate(_wf({"mode": "bogus", "cases": ["a"]})))


def test_validate_flags_edge_from_unknown_case_port():
    wf = _wf({"mode": "match", "cases": ["yes"]},
             extra_nodes=[{"id": "o", "type": "output", "config": {"name": "r"}}],
             edges=[{"from_node": "b", "from_port": "maybe", "to_node": "o", "to_port": "value"}])
    assert any("invalid output port" in e for e in m.validate(wf))
