"""In-memory tracker for currently-executing workflow runs (Mission Control
sub-project 2c). Mirrors src/agent_runs.py's own pattern: a live snapshot of
what's running right now, with no persisted history -- a run disappears the
instant it finishes, success or failure alike. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

_RUNS: Dict[str, Dict] = {}


def start(workflow_id: str, workflow_name: str, owner: Optional[str], trigger: str) -> str:
    run_id = str(uuid.uuid4())
    _RUNS[run_id] = {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "owner": owner,
        "trigger": trigger,
        "started_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }
    return run_id


def finish(run_id: str) -> None:
    _RUNS.pop(run_id, None)


def list_active() -> List[Dict]:
    return list(_RUNS.values())
