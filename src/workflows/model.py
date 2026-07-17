"""Workflow graph model: node/edge shapes, port derivation, validation, topo sort.

A workflow is {"id","name","nodes":[{"id","type","config"}],"edges":[
{"from_node","from_port","to_node","to_port"}]}. Text-only wires; the input
ports of template/llm/tool nodes are DERIVED from the {slot} names in their
text config, so the editor never hand-declares them."""
import re
from collections import deque

NODE_TYPES = ("input", "template", "llm", "tool", "output")

_SLOT_RE = re.compile(r"\{(\w+)\}")

# type -> the config key whose {slots} become that node's input ports
_SLOT_SOURCE = {"template": "template", "llm": "prompt", "tool": "args"}
_OUTPUT_PORT = {"input": "value", "template": "text", "llm": "text", "tool": "result"}


class WorkflowError(Exception):
    """Invalid workflow graph. `.errors` holds the human-readable reasons."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def slots_of(text):
    """Ordered unique {slot} names in `text`."""
    out = []
    for name in _SLOT_RE.findall(text or ""):
        if name not in out:
            out.append(name)
    return out


def input_ports(node):
    t = node.get("type")
    if t == "output":
        return ["value"]
    key = _SLOT_SOURCE.get(t)
    if not key:
        return []          # input nodes (and unknown types) take no wires
    return slots_of((node.get("config") or {}).get(key, ""))


def output_ports(node):
    port = _OUTPUT_PORT.get(node.get("type"))
    return [port] if port else []


def validate(wf):
    """Return a list of human-readable errors ([] means valid)."""
    errors = []
    nodes = wf.get("nodes") or []
    edges = wf.get("edges") or []
    by_id = {}
    for n in nodes:
        nid = n.get("id")
        if nid in by_id:
            errors.append(f"duplicate node id: {nid}")
        by_id[nid] = n
        if n.get("type") not in NODE_TYPES:
            errors.append(f"unknown node type: {n.get('type')} (node {nid})")
    for e in edges:
        src, dst = e.get("from_node"), e.get("to_node")
        if src not in by_id:
            errors.append(f"edge references unknown node: {src}")
        if dst not in by_id:
            errors.append(f"edge references unknown node: {dst}")
        if src in by_id and e.get("from_port") not in output_ports(by_id[src]):
            errors.append(f"invalid output port '{e.get('from_port')}' on node {src}")
        if dst in by_id and e.get("to_port") not in input_ports(by_id[dst]):
            errors.append(f"invalid input port '{e.get('to_port')}' on node {dst}")
    # every declared input port must be wired
    wired = {(e.get("to_node"), e.get("to_port")) for e in edges}
    for n in nodes:
        for port in input_ports(n):
            if (n.get("id"), port) not in wired:
                errors.append(f"unwired input port '{port}' on node {n.get('id')}")
    if not errors:
        try:
            topo_sort(wf)
        except WorkflowError as ex:
            errors.extend(ex.errors)
    return errors


def topo_sort(wf):
    """Node ids in execution order (Kahn). Raises WorkflowError on a cycle."""
    nodes = wf.get("nodes") or []
    edges = wf.get("edges") or []
    ids = [n.get("id") for n in nodes]
    indeg = {i: 0 for i in ids}
    adj = {i: [] for i in ids}
    for e in edges:
        src, dst = e.get("from_node"), e.get("to_node")
        if src in indeg and dst in indeg:
            adj[src].append(dst)
            indeg[dst] += 1
    q = deque([i for i in ids if indeg[i] == 0])
    order = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if len(order) != len(ids):
        raise WorkflowError(["cycle detected in workflow graph"])
    return order
