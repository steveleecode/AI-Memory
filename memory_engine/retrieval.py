from __future__ import annotations

from typing import Protocol

from memory_engine.schema import Memory


class EmbedderProtocol(Protocol):
    def embed_text(self, text: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    def search_similar(self, query_embedding: list[float], k: int = 5) -> list[Memory]: ...


class MemoryRetriever:
    def __init__(self, embedder: EmbedderProtocol, vector_store: VectorStoreProtocol) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve_for_query(self, query: str, k: int = 5) -> list[Memory]:
        query_embedding = self._embedder.embed_text(query)
        return self._vector_store.search_similar(query_embedding, k)


def retrieve_for_query(
    query: str,
    k: int,
    embedder: EmbedderProtocol,
    vector_store: VectorStoreProtocol,
) -> list[Memory]:
    return MemoryRetriever(embedder, vector_store).retrieve_for_query(query, k)
