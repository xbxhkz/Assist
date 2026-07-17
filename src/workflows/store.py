"""Workflow persistence: one JSON file per workflow under DATA_DIR/workflows.
Mirrors the path-safety of the Skills/LoRA stores (ids may not traverse)."""
import json
import os
import re

from src.constants import DATA_DIR


def workflows_dir():
    d = os.path.join(DATA_DIR, "workflows")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_id(wid):
    if not wid or "/" in wid or "\\" in wid or ".." in wid or ":" in wid:
        raise ValueError("unsafe workflow id")
    base = os.path.basename(wid)
    if not base:
        raise ValueError("unsafe workflow id")
    return base


def _slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "workflow"


def _path(wid):
    return os.path.join(workflows_dir(), _safe_id(wid) + ".json")


def list_workflows():
    out = []
    for fn in sorted(os.listdir(workflows_dir())):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(workflows_dir(), fn), "r", encoding="utf-8") as f:
                wf = json.load(f)
            if not isinstance(wf, dict):
                continue
            out.append({"id": wf.get("id", fn[:-5]), "name": wf.get("name", fn[:-5]),
                        "nodes": len(wf.get("nodes") or [])})
        except (OSError, json.JSONDecodeError):
            continue
    return out


def get_workflow(wid):
    try:
        with open(_path(wid), "r", encoding="utf-8") as f:
            wf = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return wf if isinstance(wf, dict) else None


def _unique_id(slug):
    if not os.path.isfile(_path(slug)):
        return slug
    n = 2
    while os.path.isfile(_path(f"{slug}-{n}")):
        n += 1
    return f"{slug}-{n}"


def save_workflow(wf):
    explicit_id = wf.get("id")
    if explicit_id:
        wid = _safe_id(explicit_id)
    else:
        wid = _unique_id(_safe_id(_slugify(wf.get("name"))))
    wf = dict(wf, id=wid)
    with open(_path(wf["id"]), "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)
    return wf


def delete_workflow(wid):
    p = _path(wid)
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False
