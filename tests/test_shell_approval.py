import asyncio
import src.shell_exec.approval as a


def setup_function():
    a.reset_all()


def test_approve_flow():
    async def run():
        task = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)                       # let it register the pending
        pend = a.list_pending("s1")
        assert len(pend) == 1 and pend[0]["command"] == "del x"
        assert a.set_decision("s1", pend[0]["pending_id"], "approve") is True
        return await task
    assert asyncio.run(run()) == "approve"


def test_deny_flow():
    async def run():
        task = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)
        pid = a.list_pending("s1")[0]["pending_id"]
        a.set_decision("s1", pid, "deny")
        return await task
    assert asyncio.run(run()) == "deny"


def test_auto_approve_all_sets_flag_and_skips_staging():
    async def run():
        t1 = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)
        pid = a.list_pending("s1")[0]["pending_id"]
        a.set_decision("s1", pid, "auto_approve_all")
        first = await t1
        # a subsequent write auto-approves with no pending staged
        second = await a.await_decision("s1", "del y", "cmd", timeout=5)
        assert a.list_pending("s1") == []
        return first, second
    assert asyncio.run(run()) == ("approve", "approve")


def test_timeout_denies():
    assert asyncio.run(a.await_decision("s1", "del x", "cmd", timeout=0.05)) == "deny"


def test_reset_clears_auto_all():
    async def run():
        t1 = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)
        a.set_decision("s1", a.list_pending("s1")[0]["pending_id"], "auto_approve_all")
        await t1
        a.reset_session("s1")                        # e.g. consent turned off
        return await a.await_decision("s1", "del y", "cmd", timeout=0.05)
    assert asyncio.run(run()) == "deny"              # no longer auto-approving
