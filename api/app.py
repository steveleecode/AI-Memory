from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from llm_interface.client import OpenAIChatClient
from llm_interface.embeddings import OpenAIEmbedder
from memory_engine.service import MemoryChatService
from storage.vector_store import ChromaVectorStore


app = FastAPI(title="AI Memory API", version="0.1.0")

service = MemoryChatService(
    chat_client=OpenAIChatClient(),
    embedder=OpenAIEmbedder(),
    vector_store=ChromaVectorStore(),
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    return service.run_turn(request.message)
