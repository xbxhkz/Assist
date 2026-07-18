import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.workflows.model import input_ports, output_ports

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")


def _node_eval(source):
    r = subprocess.run(["node", "--input-type=module", "-e", source],
                       cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout)


def test_branch_in_node_types_and_ports_match_python():
    out = _node_eval(
        "import { NODE_TYPES, inputPortsOf, outputPortsOf } from './static/js/workflowGraph.js';"
        "const b = {type:'branch', config:{mode:'match', cases:['yes','no']}};"
        "console.log(JSON.stringify({has: NODE_TYPES.includes('branch'),"
        "inp: inputPortsOf(b), out: outputPortsOf(b),"
        "bad: outputPortsOf({type:'branch', config:{cases:'nope'}})}));"
    )
    b = {"type": "branch", "config": {"mode": "match", "cases": ["yes", "no"]}}
    assert out["has"] is True
    assert out["inp"] == input_ports(b) == ["value"]
    assert out["out"] == output_ports(b) == ["yes", "no", "else"]        # cross-language parity
    assert out["bad"] == output_ports({"type": "branch", "config": {"cases": "nope"}}) == ["else"]


def test_setconfig_prunes_outgoing_edge_when_case_removed():
    out = _node_eval(
        "import { createGraph, setConfig } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w', name:'W', nodes:["
        "{id:'b', type:'branch', config:{mode:'match', cases:['yes','no']}},"
        "{id:'o', type:'output', config:{name:'r'}}], edges:["
        "{from_node:'b', from_port:'no', to_node:'o', to_port:'value'}]});"
        "setConfig(g, 'b', {mode:'match', cases:['yes']});"
        "console.log(JSON.stringify({edges: g.edges.length}));"
    )
    assert out == {"edges": 0}
