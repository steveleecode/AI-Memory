from __future__ import annotations

from typing import Protocol

from config import settings
from llm_interface.client import safe_generate_response
from memory_engine.prompt_builder import build_prompt
from memory_engine.retrieval import MemoryRetriever
from memory_engine.schema import Memory, MemoryType
from memory_engine.scoring import score_importance


class EmbedderProtocol(Protocol):
    def embed_text(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    def upsert_memory(self, memory: Memory) -> None: ...


class ChatClientProtocol(Protocol):
    def generate_response(self, messages: list[dict[str, str]], system_prompt: str) -> str: ...


class MemoryChatService:
    def __init__(
        self,
        chat_client: ChatClientProtocol,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k if top_k is not None else settings.memory_top_k
        self.threshold = threshold if threshold is not None else settings.memory_importance_threshold
        self.history: list[dict[str, str]] = []

    def run_turn(self, user_input: str) -> dict[str, object]:
        try:
            memories = MemoryRetriever(self.embedder, self.vector_store).retrieve_for_query(
                user_input, self.top_k
            )
        except Exception:
            memories = []
        prompt_payload = build_prompt(user_input=user_input, memories=memories, history=self.history)
        response_text = safe_generate_response(
            messages=prompt_payload.messages,
            system_prompt=prompt_payload.system_prompt,
            chat_client=self.chat_client,
        )

        self.history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response_text},
            ]
        )

        importance = score_importance(user_input, context=response_text)
        stored = False
        if importance >= self.threshold:
            try:
                vector = self.embedder.embed_text(user_input)
                mem_type = infer_memory_type(user_input)
                memory = Memory(text=user_input, embedding=vector, importance=importance, type=mem_type)
                self.vector_store.upsert_memory(memory)
                stored = True
            except Exception:
                stored = False

        return {
            "response": response_text,
            "memories_used": [m.model_dump(mode="json") for m in memories],
            "importance": importance,
            "stored": stored,
        }


def infer_memory_type(text: str) -> MemoryType:
    lowered = text.lower()
    if any(token in lowered for token in ["prefer", "like", "favorite"]):
        return "preference"
    if any(token in lowered for token in ["todo", "task", "deadline", "need to"]):
        return "task"
    return "fact"
