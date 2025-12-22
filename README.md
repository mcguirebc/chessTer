# chessTer

FastAPI-first chess move generation server with pluggable policies (random / Stockfish / Ollama LLM), plus a codebase foundation to add RL + self-play later.

## Quickstart

Create a venv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the API:

```bash
uvicorn chesster.api.app:app --reload
```

List policies:

```bash
curl http://127.0.0.1:8000/v1/policies
```

Request a move (random policy):

```bash
curl -X POST http://127.0.0.1:8000/v1/move \
  -H 'content-type: application/json' \
  -d '{"fen":"startpos","policy_id":"random","idempotency_key":"demo-1"}'
```

## Optional backends

- Stockfish: set `STOCKFISH_PATH` or ensure `stockfish` is on your PATH.
- Ollama: set `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`) and use policy `ollama`.

