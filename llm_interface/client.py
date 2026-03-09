from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import settings


class OpenAIChatClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_chat_model

    def generate_response(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=self._api_key)
        payload: list[dict[str, str]] = [{"role": "system", "content": system_prompt}, *messages]
        completion = client.chat.completions.create(model=self._model, messages=payload)
        content = completion.choices[0].message.content
        return content if isinstance(content, str) else ""


def generate_response(messages: list[dict[str, str]], system_prompt: str) -> str:
    return OpenAIChatClient().generate_response(messages, system_prompt)


def safe_generate_response(
    messages: list[dict[str, str]],
    system_prompt: str,
    chat_client: Any,
) -> str:
    try:
        return chat_client.generate_response(messages, system_prompt)
    except Exception as exc:  # pragma: no cover
        return f"I ran into an LLM provider error: {exc}"
