from __future__ import annotations

from openai import OpenAI

from config import settings


class OpenAIEmbedder:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_embed_model

    def embed_text(self, text: str) -> list[float]:
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in response.data]


def embed_text(text: str) -> list[float]:
    return OpenAIEmbedder().embed_text(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return OpenAIEmbedder().embed_texts(texts)
