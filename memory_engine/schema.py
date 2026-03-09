from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


MemoryType = Literal["fact", "preference", "task"]


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(min_length=1)
    embedding: list[float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = Field(ge=0.0, le=1.0)
    type: MemoryType

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("embedding cannot be empty")
        return value


class PromptPayload(BaseModel):
    system_prompt: str
    messages: list[dict[str, str]]
