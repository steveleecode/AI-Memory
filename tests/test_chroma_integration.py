from __future__ import annotations

import pytest

chromadb = pytest.importorskip("chromadb")

from memory_engine.schema import Memory
from storage.vector_store import ChromaVectorStore


def test_chroma_returns_most_similar(tmp_path) -> None:
    store = ChromaVectorStore(path=str(tmp_path / "chroma"), collection_name="test_memories")

    store.upsert_memory(
        Memory(text="study physics", embedding=[1.0, 0.0], importance=0.8, type="fact")
    )
    store.upsert_memory(
        Memory(text="buy groceries", embedding=[0.0, 1.0], importance=0.6, type="task")
    )

    results = store.search_similar([0.9, 0.1], k=1)

    assert len(results) == 1
    assert "physics" in results[0].text
