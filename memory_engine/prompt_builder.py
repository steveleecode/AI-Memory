from __future__ import annotations

from memory_engine.schema import Memory, PromptPayload


BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use relevant user memory context when it is helpful and factual. "
    "If memory conflicts with current user input, trust the current input."
)


def build_prompt(
    user_input: str,
    memories: list[Memory],
    history: list[dict[str, str]],
) -> PromptPayload:
    memory_lines = [f"- ({m.type}) {m.text}" for m in memories]
    memory_block = "\n".join(memory_lines) if memory_lines else "- None"

    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nRelevant memories:\n{memory_block}"

    messages = [*history, {"role": "user", "content": user_input}]
    return PromptPayload(system_prompt=system_prompt, messages=messages)
