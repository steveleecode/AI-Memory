from __future__ import annotations

from memory_engine.schema import Memory
from memory_engine.service import MemoryChatService


class FailingEmbedder:
    def embed_text(self, text: str) -> list[float]:
        raise ValueError("OPENAI_API_KEY is not set")


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.items: list[Memory] = []

    def upsert_memory(self, memory: Memory) -> None:
        self.items.append(memory)

    def search_similar(self, query_embedding: list[float], k: int = 5) -> list[Memory]:
        return []


class EchoChatClient:
    def generate_response(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        return "ok"


class FailingChatClient:
    def generate_response(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        raise RuntimeError("rate limited")


def test_missing_api_key_path_does_not_crash() -> None:
    service = MemoryChatService(
        chat_client=EchoChatClient(),
        embedder=FailingEmbedder(),
        vector_store=InMemoryVectorStore(),
    )

    result = service.run_turn("hello")
    assert "response" in result


def test_provider_error_returns_fallback_message() -> None:
    service = MemoryChatService(
        chat_client=FailingChatClient(),
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
    )

    result = service.run_turn("hello")
    assert "provider error" in str(result["response"]).lower()


class InMemoryEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2]
