import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")


def _node_eval(source: str):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_slots_of_ordered_unique():
    out = _node_eval(
        "import { slotsOf } from './static/js/workflowGraph.js';"
        "console.log(JSON.stringify({"
        "a: slotsOf('Hi {name}, you are {age}. Bye {name}'),"
        "b: slotsOf('no slots'), c: slotsOf('')}));"
    )
    assert out == {"a": ["name", "age"], "b": [], "c": []}


def test_ports_per_type():
    out = _node_eval(
        "import { inputPortsOf, outputPortsOf } from './static/js/workflowGraph.js';"
        "const mk=(t,c)=>({type:t,config:c});"
        "console.log(JSON.stringify({"
        "iIn: inputPortsOf(mk('input',{name:'q'})), iOut: outputPortsOf(mk('input',{})),"
        "tIn: inputPortsOf(mk('template',{template:'{a}-{b}'})), tOut: outputPortsOf(mk('template',{})),"
        "lIn: inputPortsOf(mk('llm',{prompt:'sum {doc}'})), lOut: outputPortsOf(mk('llm',{})),"
        "toIn: inputPortsOf(mk('tool',{args:'{path}'})), toOut: outputPortsOf(mk('tool',{})),"
        "oIn: inputPortsOf(mk('output',{})), oOut: outputPortsOf(mk('output',{}))}));"
    )
    assert out == {"iIn": [], "iOut": ["value"], "tIn": ["a", "b"], "tOut": ["text"],
                   "lIn": ["doc"], "lOut": ["text"], "toIn": ["path"], "toOut": ["result"],
                   "oIn": ["value"], "oOut": []}


def test_add_node_unique_ids_and_roundtrip():
    out = _node_eval(
        "import { createGraph, addNode, setConfig, toJSON } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w', name:'W'});"
        "const a = addNode(g,'template',10,20); const b = addNode(g,'template',30,40);"
        "setConfig(g, a.id, {template:'Q: {q}'});"
        "const wf = toJSON(g);"
        "console.log(JSON.stringify({ids:[a.id,b.id], distinct: a.id!==b.id,"
        "n0:{id:wf.nodes[0].id,type:wf.nodes[0].type,x:wf.nodes[0].x,y:wf.nodes[0].y,cfg:wf.nodes[0].config},"
        "meta:{id:wf.id,name:wf.name}}));"
    )
    assert out["distinct"] is True
    assert out["n0"]["type"] == "template" and out["n0"]["x"] == 10 and out["n0"]["y"] == 20
    assert out["n0"]["cfg"] == {"template": "Q: {q}"}
    assert out["meta"] == {"id": "w", "name": "W"}


def test_create_graph_preserves_positions_from_json():
    out = _node_eval(
        "import { createGraph, toJSON, nodeById } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w',name:'W',"
        "nodes:[{id:'i',type:'input',config:{name:'q'},x:5,y:7}],edges:[]});"
        "console.log(JSON.stringify({pos:[nodeById(g,'i').x,nodeById(g,'i').y],"
        "wf: toJSON(g).nodes[0]}));"
    )
    assert out["pos"] == [5, 7]
    assert out["wf"] == {"id": "i", "type": "input", "config": {"name": "q"}, "x": 5, "y": 7}


def test_set_node_id_rejects_empty_and_duplicate():
    out = _node_eval(
        "import { createGraph, addNode, setNodeId } from './static/js/workflowGraph.js';"
        "const g = createGraph(); const a = addNode(g,'input'); const b = addNode(g,'input');"
        "let empty=false, dup=false, ok=false;"
        "try { setNodeId(g,a.id,''); } catch(e){ empty=true; }"
        "try { setNodeId(g,a.id,b.id); } catch(e){ dup=true; }"
        "try { setNodeId(g,a.id,'fresh'); ok = (a.id==='fresh'); } catch(e){}"
        "console.log(JSON.stringify({empty,dup,ok}));"
    )
    assert out == {"empty": True, "dup": True, "ok": True}


def test_topo_order_and_autolayout_and_run_inputs():
    out = _node_eval(
        "import { createGraph, topoOrder, autoLayout, runInputNames, nodeById } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w',name:'W',nodes:["
        "{id:'i',type:'input',config:{name:'q',default:'d'}},"
        "{id:'t',type:'template',config:{template:'Q: {q}'}},"
        "{id:'o',type:'output',config:{name:'answer'}}],"
        "edges:[{from_node:'i',from_port:'value',to_node:'t',to_port:'q'},"
        "{from_node:'t',from_port:'text',to_node:'o',to_port:'value'}]});"
        "autoLayout(g);"
        "console.log(JSON.stringify({order: topoOrder(g),"
        "layers:[nodeById(g,'i').x, nodeById(g,'t').x, nodeById(g,'o').x],"
        "inputs: runInputNames(g)}));"
    )
    assert out["order"] == ["i", "t", "o"]
    assert out["layers"][0] < out["layers"][1] < out["layers"][2]  # left-to-right by depth
    assert out["inputs"] == [{"name": "q", "default": "d"}]
