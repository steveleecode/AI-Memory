from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from config import settings
from memory_engine.schema import Memory


class ChromaVectorStore:
    def __init__(self, path: str | None = None, collection_name: str = "memories") -> None:
        self._client = chromadb.PersistentClient(path=path or settings.chroma_path)
        self._collection: Collection = self._client.get_or_create_collection(name=collection_name)

    def upsert_memory(self, memory: Memory) -> None:
        self._collection.upsert(
            ids=[memory.id],
            documents=[memory.text],
            embeddings=[memory.embedding],
            metadatas=[
                {
                    "timestamp": memory.timestamp.isoformat(),
                    "importance": memory.importance,
                    "type": memory.type,
                }
            ],
        )

    def search_similar(self, query_embedding: list[float], k: int = 5) -> list[Memory]:
        result: dict[str, Any] = self._collection.query(query_embeddings=[query_embedding], n_results=k)

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        embeddings_result = result.get("embeddings")
        if embeddings_result and isinstance(embeddings_result, list) and embeddings_result[0]:
            embs = embeddings_result[0]
        else:
            embs = [query_embedding] * len(ids)

        memories: list[Memory] = []
        for memory_id, doc, meta, emb in zip(ids, docs, metas, embs):
            metadata = meta or {}
            memory = Memory(
                id=memory_id,
                text=doc,
                embedding=list(emb),
                timestamp=metadata.get("timestamp"),
                importance=float(metadata.get("importance", 0.5)),
                type=metadata.get("type", "fact"),
            )
            memories.append(memory)
        return memories
