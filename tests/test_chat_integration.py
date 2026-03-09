from __future__ import annotations

from memory_engine.schema import Memory
from memory_engine.service import MemoryChatService


class FakeEmbedder:
    def __init__(self) -> None:
        self.table = {
            "I prefer practice tests": [1.0, 0.0],
            "How should I study for AP Physics?": [0.9, 0.1],
            "Hi": [0.0, 0.0],
        }

    def embed_text(self, text: str) -> list[float]:
        return self.table.get(text, [0.5, 0.5])


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.items: list[Memory] = []

    def upsert_memory(self, memory: Memory) -> None:
        self.items.append(memory)

    def search_similar(self, query_embedding: list[float], k: int = 5) -> list[Memory]:
        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        ranked = sorted(self.items, key=lambda m: cosine(query_embedding, m.embedding), reverse=True)
        return ranked[:k]


class FakeChatClient:
    def generate_response(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        return f"SYS={system_prompt}\nLAST={messages[-1]['content']}"


def test_chat_turn_stores_memory_when_above_threshold() -> None:
    service = MemoryChatService(
        chat_client=FakeChatClient(),
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
        threshold=0.55,
    )

    result = service.run_turn("I prefer practice tests")

    assert result["stored"] is True


def test_end_to_end_memory_influences_prompt() -> None:
    store = InMemoryVectorStore()
    service = MemoryChatService(
        chat_client=FakeChatClient(),
        embedder=FakeEmbedder(),
        vector_store=store,
        threshold=0.55,
        top_k=3,
    )

    service.run_turn("I prefer practice tests")
    result = service.run_turn("How should I study for AP Physics?")

    assert "User prefers practice tests" not in str(result["response"])
    # Injected memory is user-authored text and should appear in system prompt serialization.
    assert "I prefer practice tests" in str(result["response"])
