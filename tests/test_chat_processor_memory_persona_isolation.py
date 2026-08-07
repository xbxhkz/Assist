"""Auto-injected chat context (build_context_preface) must respect the
same persona isolation manage_memory does -- a persona-bound turn's
background memory priming pulls only from that persona's memories plus
the shared pool. This is a thin proof that persona_id reaches
memory_manager.load() correctly; the filtering logic itself is already
covered by tests/test_memory_persona_isolation.py."""
from unittest.mock import MagicMock

from src.chat_processor import ChatProcessor


def _make_processor(loaded_memories):
    memory_manager = MagicMock()
    memory_manager.load.return_value = loaded_memories
    memory_manager.increment_uses = MagicMock()
    processor = ChatProcessor(memory_manager, personal_docs_manager=None, memory_vector=None, skills_manager=None)
    return processor, memory_manager


def test_build_context_preface_passes_persona_id_to_memory_manager_load():
    processor, memory_manager = _make_processor(loaded_memories=[])
    processor.build_context_preface(
        message="hello", session=MagicMock(), use_memory=True, owner="alice", persona_id="persona-a",
    )
    memory_manager.load.assert_called_once_with(owner="alice", persona_id="persona-a")


def test_build_context_preface_without_persona_id_matches_todays_call_shape():
    """Fail-open / backward compatibility: omitting persona_id must call
    memory_manager.load() exactly as it's called today (owner-only)."""
    processor, memory_manager = _make_processor(loaded_memories=[])
    processor.build_context_preface(
        message="hello", session=MagicMock(), use_memory=True, owner="alice",
    )
    memory_manager.load.assert_called_once_with(owner="alice", persona_id=None)


def test_build_context_preface_still_injects_a_pinned_memory():
    """Regression guard: the persona_id plumbing must not disturb the
    existing pinned/extended memory injection behavior."""
    processor, _ = _make_processor(loaded_memories=[
        {"id": "m1", "text": "core fact", "pinned": True, "category": "fact"},
    ])
    preface, _, _ = processor.build_context_preface(
        message="hello", session=MagicMock(), use_memory=True, owner="alice", persona_id="persona-a",
    )
    joined = " ".join(m.get("content", "") for m in preface)
    assert "core fact" in joined
