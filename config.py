from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chroma_path: str = os.getenv("CHROMA_PATH", ".chroma")
    memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "5"))
    memory_importance_threshold: float = float(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", "0.55"))


settings = Settings()
