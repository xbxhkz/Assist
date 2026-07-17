import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")

_SETUP = (
    "import { createGraph, addNode, setConfig, canConnect, addEdge, removeEdge,"
    " removeNode, unwiredPorts, nodeById } from './static/js/workflowGraph.js';"
    "const g = createGraph();"
    "const i = addNode(g,'input'); const t = addNode(g,'template');"
    "setConfig(g, t.id, {template:'Q: {q}'});"
)


def _node_eval(body: str):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", _SETUP + body],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_add_valid_edge_and_reject_bad_ports_and_self_loop():
    out = _node_eval(
        "const good = addEdge(g, i.id, 'value', t.id, 'q');"
        "const badFromPort = addEdge(g, i.id, 'nope', t.id, 'q');"
        "const badToPort = addEdge(g, i.id, 'value', t.id, 'zzz');"
        "const self = addEdge(g, t.id, 'text', t.id, 'q');"
        "console.log(JSON.stringify({good, badFromPort, badToPort, self, edges: g.edges.length}));"
    )
    assert out == {"good": True, "badFromPort": False, "badToPort": False, "self": False, "edges": 1}


def test_one_edge_per_input_replaces():
    out = _node_eval(
        "const i2 = addNode(g,'input');"
        "addEdge(g, i.id, 'value', t.id, 'q');"
        "addEdge(g, i2.id, 'value', t.id, 'q');"   # re-wire same input port
        "console.log(JSON.stringify({count: g.edges.length, from: g.edges[0].from_node === i2.id}));"
    )
    assert out == {"count": 1, "from": True}


def test_cycle_is_rejected():
    out = _node_eval(
        "const t2 = addNode(g,'template'); setConfig(g, t2.id, {template:'{x}'});"
        "setConfig(g, t.id, {template:'{q}{y}'});"   # t now has ports q, y
        "addEdge(g, t.id, 'text', t2.id, 'x');"       # t -> t2
        "const back = canConnect(g, t2.id, 'text', t.id, 'y');"  # t2 -> t would cycle
        "const added = addEdge(g, t2.id, 'text', t.id, 'y');"
        "console.log(JSON.stringify({back, added, edges: g.edges.length}));"
    )
    assert out == {"back": False, "added": False, "edges": 1}


def test_setconfig_prunes_orphaned_edge_and_removenode_drops_edges():
    out = _node_eval(
        "addEdge(g, i.id, 'value', t.id, 'q');"
        "setConfig(g, t.id, {template:'no slots now'});"   # {q} gone -> edge orphaned
        "const afterPrune = g.edges.length;"
        "setConfig(g, t.id, {template:'{q}'}); addEdge(g, i.id, 'value', t.id, 'q');"
        "removeNode(g, i.id);"                              # dropping the source node
        "console.log(JSON.stringify({afterPrune, afterRemove: g.edges.length}));"
    )
    assert out == {"afterPrune": 0, "afterRemove": 0}


def test_unwired_ports_lists_missing_inputs():
    out = _node_eval(
        "const o = addNode(g,'output');"          # output has input port 'value', unwired
        "console.log(JSON.stringify({unwired: unwiredPorts(g)}));"
    )
    # t has 'q' unwired, o has 'value' unwired (input node has no input ports)
    pairs = {(u["node"], u["port"]) for u in out["unwired"]}
    assert pairs == {("template1", "q"), ("output1", "value")}
