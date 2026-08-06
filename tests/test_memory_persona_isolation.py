"""Persona-scoped memory isolation at the MemoryManager layer. Hard
isolation: a persona sees its own tagged entries plus the shared
(untagged) pool, never another persona's tagged entries. Mirrors the
sub-project-2 fail-open precedent: no persona_id given -> today's exact
owner-only behavior, unchanged."""
from src.memory import MemoryManager


def test_add_entry_sets_persona_id_when_given(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("likes dark mode", owner="alice", persona_id="persona-a")
    assert entry["persona_id"] == "persona-a"


def test_add_entry_omits_persona_id_when_not_given(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("likes dark mode", owner="alice")
    assert "persona_id" not in entry


def test_load_without_persona_id_returns_everything_for_the_owner(tmp_path):
    """Fail-open / backward compatibility: omitting persona_id must behave
    exactly like today -- owner-only filtering, no persona narrowing."""
    mgr = MemoryManager(str(tmp_path))
    e1 = mgr.add_entry("fact one", owner="alice", persona_id="persona-a")
    e2 = mgr.add_entry("fact two", owner="alice", persona_id="persona-b")
    e3 = mgr.add_entry("fact three", owner="alice")
    mgr.save([e1, e2, e3])

    loaded = mgr.load(owner="alice")
    assert {m["text"] for m in loaded} == {"fact one", "fact two", "fact three"}


def test_load_with_persona_id_hides_other_personas_entries(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    e1 = mgr.add_entry("persona A's fact", owner="alice", persona_id="persona-a")
    e2 = mgr.add_entry("persona B's fact", owner="alice", persona_id="persona-b")
    e3 = mgr.add_entry("shared fact", owner="alice")
    mgr.save([e1, e2, e3])

    loaded = mgr.load(owner="alice", persona_id="persona-a")
    texts = {m["text"] for m in loaded}
    assert texts == {"persona A's fact", "shared fact"}
    assert "persona B's fact" not in texts


def test_load_with_persona_id_still_respects_owner(tmp_path):
    """Persona filtering is additive to owner filtering, not a replacement
    for it -- a different owner's same-persona-id entry must still be
    invisible (persona ids are only meaningful within one owner's CrewMember
    table, but this guards against any future ID-collision assumption)."""
    mgr = MemoryManager(str(tmp_path))
    e1 = mgr.add_entry("alice's fact", owner="alice", persona_id="persona-a")
    e2 = mgr.add_entry("bob's fact", owner="bob", persona_id="persona-a")
    mgr.save([e1, e2])

    loaded = mgr.load(owner="alice", persona_id="persona-a")
    assert {m["text"] for m in loaded} == {"alice's fact"}
