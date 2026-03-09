from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_interface.client import OpenAIChatClient
from llm_interface.embeddings import OpenAIEmbedder
from memory_engine.service import MemoryChatService
from storage.vector_store import ChromaVectorStore


def main() -> None:
    service = MemoryChatService(
        chat_client=OpenAIChatClient(),
        embedder=OpenAIEmbedder(),
        vector_store=ChromaVectorStore(),
    )

    print("AI Memory CLI started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        result = service.run_turn(user_input)
        print(f"AI: {result['response']}")
        if result["stored"]:
            print(f"[memory] stored (importance={result['importance']:.2f})")


if __name__ == "__main__":
    main()
