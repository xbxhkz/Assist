import pytest
import src.workflows.model as m


def _wf(nodes, edges):
    return {"id": "w1", "name": "W", "nodes": nodes, "edges": edges}


def test_slots_of_extracts_unique_ordered():
    assert m.slots_of("Hi {name}, you are {age}. Bye {name}") == ["name", "age"]
    assert m.slots_of("no slots") == []


def test_ports_per_type():
    assert m.input_ports({"type": "input", "config": {"name": "q"}}) == []
    assert m.output_ports({"type": "input", "config": {}}) == ["value"]
    assert m.input_ports({"type": "template", "config": {"template": "{a}-{b}"}}) == ["a", "b"]
    assert m.output_ports({"type": "template", "config": {}}) == ["text"]
    assert m.input_ports({"type": "llm", "config": {"prompt": "sum {doc}"}}) == ["doc"]
    assert m.output_ports({"type": "llm", "config": {}}) == ["text"]
    assert m.input_ports({"type": "tool", "config": {"args": "{path}"}}) == ["path"]
    assert m.output_ports({"type": "tool", "config": {}}) == ["result"]
    assert m.input_ports({"type": "output", "config": {}}) == ["value"]
    assert m.output_ports({"type": "output", "config": {}}) == []


def _linear():
    return _wf(
        [{"id": "i", "type": "input", "config": {"name": "q"}},
         {"id": "t", "type": "template", "config": {"template": "Q: {q}"}},
         {"id": "o", "type": "output", "config": {"name": "answer"}}],
        [{"from_node": "i", "from_port": "value", "to_node": "t", "to_port": "q"},
         {"from_node": "t", "from_port": "text", "to_node": "o", "to_port": "value"}],
    )


def test_validate_accepts_valid_graph():
    assert m.validate(_linear()) == []


def test_validate_flags_unknown_type_and_dup_ids():
    wf = _wf([{"id": "a", "type": "bogus", "config": {}},
              {"id": "a", "type": "input", "config": {"name": "q"}}], [])
    errs = m.validate(wf)
    assert any("unknown node type" in e for e in errs)
    assert any("duplicate node id" in e for e in errs)


def test_validate_flags_dangling_edge_and_bad_port():
    wf = _linear()
    wf["edges"].append({"from_node": "nope", "from_port": "value", "to_node": "o", "to_port": "value"})
    assert any("unknown node" in e for e in m.validate(wf))
    wf2 = _linear()
    wf2["edges"][0]["to_port"] = "zzz"
    assert any("invalid input port" in e for e in m.validate(wf2))


def test_validate_flags_unwired_slot():
    wf = _linear()
    wf["edges"] = [e for e in wf["edges"] if e["to_node"] != "t"]  # drop the wire into {q}
    assert any("unwired input port" in e for e in m.validate(wf))


def test_validate_detects_cycle():
    wf = _wf(
        [{"id": "t1", "type": "template", "config": {"template": "{x}"}},
         {"id": "t2", "type": "template", "config": {"template": "{y}"}}],
        [{"from_node": "t1", "from_port": "text", "to_node": "t2", "to_port": "y"},
         {"from_node": "t2", "from_port": "text", "to_node": "t1", "to_port": "x"}],
    )
    assert any("cycle" in e for e in m.validate(wf))


def test_topo_sort_orders_and_raises_on_cycle():
    assert m.topo_sort(_linear()) == ["i", "t", "o"]
    wf = _wf(
        [{"id": "t1", "type": "template", "config": {"template": "{x}"}},
         {"id": "t2", "type": "template", "config": {"template": "{y}"}}],
        [{"from_node": "t1", "from_port": "text", "to_node": "t2", "to_port": "y"},
         {"from_node": "t2", "from_port": "text", "to_node": "t1", "to_port": "x"}],
    )
    with pytest.raises(m.WorkflowError):
        m.topo_sort(wf)
