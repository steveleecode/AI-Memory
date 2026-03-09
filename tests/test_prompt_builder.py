from __future__ import annotations

from memory_engine.prompt_builder import build_prompt
from memory_engine.schema import Memory


def test_prompt_builder_includes_memories_and_history() -> None:
    memories = [
        Memory(
            text="User prefers practice tests",
            embedding=[0.1, 0.2],
            importance=0.8,
            type="preference",
        )
    ]
    history = [{"role": "assistant", "content": "How can I help?"}]

    payload = build_prompt("Help me study", memories=memories, history=history)

    assert "User prefers practice tests" in payload.system_prompt
    assert payload.messages[0]["role"] == "assistant"
    assert payload.messages[-1]["content"] == "Help me study"
