# AI Memory MVP

AI memory system MVP built with FastAPI, OpenAI, and Chroma.

## Architecture

```text
User
  -> Chat Interface (CLI/API)
  -> Memory Retrieval (embed query + top-k vector search)
  -> Prompt Builder (inject relevant memories)
  -> LLM Response
  -> Importance Scoring
  -> Memory Storage (if above threshold)
```

## Milestone Status

- [x] M1 Project bootstrap (`uv`, config, structure, README)
- [x] M2 CLI chat loop + OpenAI integration + history buffer
- [x] M3 Memory object schema + categories
- [x] M4 Embedding system
- [x] M5 Chroma vector store + similarity search
- [x] M6 Retrieval-augmented prompting
- [x] M7 Importance threshold gating
- [ ] M8 Advanced phase (hierarchy, consolidation, reflection)

## Project Structure

```text
ai-memory-system/
├── api/
├── llm_interface/
├── memory_engine/
├── scripts/
├── storage/
└── tests/
```

## Setup

1. Install `uv` ([docs](https://docs.astral.sh/uv/)).
2. Create env and install deps:
   ```bash
   uv sync --extra dev
   ```
3. Configure env vars:
   ```bash
   cp .env.example .env
   # then set OPENAI_API_KEY
   ```

## Run

- CLI chat:
  ```bash
  uv run python scripts/chat_cli.py
  ```
- FastAPI:
  ```bash
  uv run uvicorn api.app:app --reload
  ```
- Tests:
  ```bash
  uv run pytest
  ```

## Environment Variables

- `OPENAI_API_KEY`: required for live LLM + embeddings.
- `OPENAI_CHAT_MODEL`: default `gpt-4.1-mini`.
- `OPENAI_EMBED_MODEL`: default `text-embedding-3-small`.
- `CHROMA_PATH`: default `.chroma`.
- `MEMORY_TOP_K`: default `5`.
- `MEMORY_IMPORTANCE_THRESHOLD`: default `0.55`.

## GitHub Project Board Initialization

1. Create a project board in GitHub (`Projects` -> `New project`).
2. Add columns: `Backlog`, `In Progress`, `Review`, `Done`.
3. Add milestone tasks M1-M8 as cards/issues.
4. Link repo issues/PRs to board items.

## Notes

- Frontend is intentionally deferred for MVP quality on memory behavior.
- Advanced features are phase-2 and should be implemented after core stability.


source .venv/bin/activate
pip install setuptools wheel
pip install --no-build-isolation -e ".[dev]"
cp .env.example .env
