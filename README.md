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

### Recommended (venv + pip)

1. Create and activate virtual env:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install build tools + project deps:
   ```bash
   pip install setuptools wheel
   pip install --no-build-isolation -e ".[dev]"
   ```
3. Configure env vars:
   ```bash
   cp .env.example .env
   # then set OPENAI_API_KEY
   ```

### Optional (uv)

```bash
uv sync --extra dev
cp .env.example .env
```

## Run

- CLI chat:
  ```bash
  python scripts/chat_cli.py
  ```
- FastAPI:
  ```bash
  uvicorn api.app:app --reload
  ```
- Tests:
  ```bash
  pytest -q
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

## Troubleshooting

- `ModuleNotFoundError: No module named 'config'` when launching CLI by absolute path:
  - Fixed in current code. You can run either:
    - `python scripts/chat_cli.py`
    - `"/absolute/path/to/.venv/bin/python" "/absolute/path/to/scripts/chat_cli.py"`
- `429 insufficient_quota` from OpenAI:
  - Confirm key/project match the enrolled data-sharing project.
  - Ensure billing balance/payment method is active.
  - Chat model (`gpt-4.1-mini`) may be eligible for complimentary usage, but other calls (like embeddings) can still require paid quota.
