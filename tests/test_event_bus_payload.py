import asyncio
import types

import src.event_bus as eb


def _run(coro):
    return asyncio.run(coro)


def test_fire_event_forwards_payload_to_handler(monkeypatch):
    seen = {}

    async def fake_handle(name, owner=None, payload=None):
        seen.update(name=name, owner=owner, payload=payload)
    monkeypatch.setattr(eb, "_handle_event", fake_handle)

    async def go():
        eb.fire_event("message_sent", "admin", payload={"message": "hi"})
        await asyncio.sleep(0)
    _run(go())
    assert seen == {"name": "message_sent", "owner": "admin", "payload": {"message": "hi"}}


def test_handle_event_passes_payload_as_context_at_threshold(monkeypatch):
    # One active event task at threshold 1; assert run_task_now gets context=payload.
    task = types.SimpleNamespace(id="t1", name="t", trigger_count=1, trigger_counter=0)

    class _Q:
        def filter(self, *a): return self
        def all(self): return [task]

    class _DB:
        def query(self, *a): return _Q()
        def commit(self): pass
        def close(self): pass
    monkeypatch.setattr(eb, "SessionLocal", lambda: _DB(), raising=False)
    # _handle_event imports SessionLocal locally from core.database; patch there too.
    import core.database as cdb
    monkeypatch.setattr(cdb, "SessionLocal", lambda: _DB())
    monkeypatch.setattr(eb, "_resolve_event_owner", lambda o: o)

    captured = {}

    class _Sched:
        async def run_task_now(self, task_id, *, context=None):
            captured.update(task_id=task_id, context=context)
    monkeypatch.setattr(eb, "_task_scheduler", _Sched())

    _run(eb._handle_event("message_sent", "admin", payload={"message": "hi"}))
    assert captured == {"task_id": "t1", "context": {"message": "hi"}}
