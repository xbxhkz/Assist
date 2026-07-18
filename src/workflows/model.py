"""Workflow graph model: node/edge shapes, port derivation, validation, topo sort.

A workflow is {"id","name","nodes":[{"id","type","config"}],"edges":[
{"from_node","from_port","to_node","to_port"}]}. Text-only wires; the input
ports of template/llm/tool nodes are DERIVED from the {slot} names in their
text config, so the editor never hand-declares them."""
import re
from collections import deque

NODE_TYPES = ("input", "template", "llm", "tool", "output", "branch")

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
    if t in ("output", "branch"):
        return ["value"]
    key = _SLOT_SOURCE.get(t)
    if not key:
        return []          # input nodes (and unknown types) take no wires
    return slots_of((node.get("config") or {}).get(key, ""))


def output_ports(node):
    if node.get("type") == "branch":
        cfg = node.get("config")
        cfg = cfg if isinstance(cfg, dict) else {}
        cases = cfg.get("cases")
        cases = cases if isinstance(cases, list) else []
        return [c for c in cases if isinstance(c, str) and c.strip()] + ["else"]
    port = _OUTPUT_PORT.get(node.get("type"))
    return [port] if port else []


def validate(wf):
    """Return a list of human-readable errors ([] means valid).

    `wf` is untrusted JSON straight off an HTTP request, so shape is never
    assumed: a malformed graph is reported as an error string like its
    neighbouring checks, never raised as an exception."""
    errors = []
    nodes = wf.get("nodes") or []
    edges = wf.get("edges") or []
    if not isinstance(nodes, list):
        errors.append(f"nodes must be a list, got {type(nodes).__name__}")
        nodes = []
    if not isinstance(edges, list):
        errors.append(f"edges must be a list, got {type(edges).__name__}")
        edges = []
    by_id = {}
    for n in nodes:
        if not isinstance(n, dict):
            errors.append(f"node must be an object: {n!r}")
            continue
        nid = n.get("id")
        if not nid:
            errors.append(f"missing node id: {n!r}")
        elif nid in by_id:
            errors.append(f"duplicate node id: {nid}")
        by_id[nid] = n
        if n.get("type") not in NODE_TYPES:
            errors.append(f"unknown node type: {n.get('type')} (node {nid})")
        if n.get("type") == "branch":
            cfg = n.get("config")
            cfg = cfg if isinstance(cfg, dict) else {}
            cases = cfg.get("cases")
            if not isinstance(cases, list) or not cases:
                errors.append(f"branch node {nid} must have a non-empty 'cases' list")
            else:
                seen = set()
                for c in cases:
                    if not isinstance(c, str) or not c.strip():
                        errors.append(f"branch node {nid} has an empty/non-string case")
                        continue
                    if c.strip().lower() == "else":
                        errors.append(f"branch node {nid}: case 'else' is reserved (auto fallback)")
                    elif c in seen:
                        errors.append(f"branch node {nid} has duplicate case '{c}'")
                    seen.add(c)
            if cfg.get("mode", "match") not in ("match", "llm"):
                errors.append(f"branch node {nid} mode must be 'match' or 'llm'")
    for e in edges:
        if not isinstance(e, dict):
            errors.append(f"edge must be an object: {e!r}")
            continue
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
    wired = {(e.get("to_node"), e.get("to_port")) for e in edges if isinstance(e, dict)}
    for n in nodes:
        if not isinstance(n, dict):
            continue
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
