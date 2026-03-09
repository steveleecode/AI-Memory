from __future__ import annotations

from pydantic import ValidationError

from memory_engine.schema import Memory


def test_memory_schema_sets_timestamp_and_accepts_category() -> None:
    memory = Memory(
        text="User studies AP Physics 2",
        embedding=[0.1, 0.2, 0.3],
        importance=0.9,
        type="fact",
    )

    assert memory.id
    assert memory.timestamp is not None
    assert memory.type == "fact"


def test_memory_schema_rejects_invalid_category() -> None:
    try:
        Memory(
            text="Invalid",
            embedding=[0.1],
            importance=0.5,
            type="note",  # type: ignore[arg-type]
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected validation error for invalid memory type")
