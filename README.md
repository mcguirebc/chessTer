# chessTer

A chess AI training research playground combining neural network policies, LLM-based move generation, and reinforcement learning via self-play.

**Goal**: Train chess-playing agents through imitation learning (behavior cloning from Stockfish) and reinforcement learning, comparing CNN-based and LLM-based approaches.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                     /v1/move, /v1/policies                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ RandomPolicy  │   │StockfishPolicy│   │  OllamaPolicy │
│   (baseline)  │   │   (teacher)   │   │  (LLM-based)  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SmallNetPolicy                             │
│          CNN with residual blocks + policy/value heads          │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐                 ┌───────────────────┐
│ Behavior Cloning  │                 │   RL Training     │
│ (imitate teacher) │                 │   (REINFORCE)     │
└───────────────────┘                 └───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Self-Play Loop                              │
│   generate games → train → gate against baseline → promote      │
└─────────────────────────────────────────────────────────────────┘
```

## Completed Phases

### Phase 1: API + Pluggable Policies ✅
- FastAPI server with idempotent `/v1/move` endpoint
- Pluggable policy architecture: Random, Stockfish, Ollama LLM
- SQLite-backed idempotency store

### Phase 2: Behavior Cloning ✅
- `SmallChessNet`: CNN with 4 residual blocks, policy head (4672 moves), value head
- Supervised learning to imitate Stockfish's best moves
- Training data: games annotated with teacher moves

### Phase 3: Self-Play RL Loop ✅
- REINFORCE trainer with multiple reward functions
- Self-play game generation with Stockfish annotation
- Model registry with versioned snapshots
- Gating system: only promote models that beat baseline

## Quickstart

### Installation

```bash
# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train]'

# For Apple Silicon (M1/M2/M3/M4)
pip install torch torchvision  # MPS acceleration
```

### Run the API

```bash
uvicorn chesster.api.app:app --reload
```

```bash
# List policies
curl http://127.0.0.1:8000/v1/policies

# Request a move
curl -X POST http://127.0.0.1:8000/v1/move \
  -H 'content-type: application/json' \
  -d '{"fen":"startpos","policy_id":"random","idempotency_key":"demo-1"}'
```

### Optional Backends

- **Stockfish**: `brew install stockfish` or set `STOCKFISH_PATH`
- **Ollama**: Install [Ollama](https://ollama.ai), run a model, set `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)

## Training

### Generate Training Data (Stockfish-annotated games)

```bash
python -m chesster.selfplay.generator \
  --policy runs/bc/v1/best_model.pt \
  --opponents random,stockfish \
  --games 1000 \
  --annotate \
  --teacher-depth 15 \
  --out data/teacher_games_large.jsonl \
  --device mps
```

### Behavior Cloning (Imitation Learning)

```bash
python -m chesster.train.bc \
  --data data/teacher_games.jsonl \
  --out runs/bc/v2 \
  --epochs 20 \
  --batch-size 256 \
  --device mps
```

### Self-Play RL Loop

```bash
python -m chesster.train.loop \
  --init runs/bc/v1/best_model.pt \
  --iterations 100 \
  --games-per-iter 200 \
  --reward match_bestmove \
  --include-stockfish \
  --gating-games 50 \
  --device mps
```

### Device Options
- `cpu` - CPU only
- `mps` - Apple Silicon (M1/M2/M3/M4)
- `cuda` - NVIDIA GPU

## Project Structure

```
chessTer/
├── src/chesster/
│   ├── api/           # FastAPI server
│   ├── chess/         # Board utilities
│   ├── league/        # Registry, gating, opponent sampling
│   ├── policies/      # Random, Stockfish, Ollama, SmallNet
│   ├── selfplay/      # Game generation, recording
│   └── train/         # BC, RL, rewards, training loop
├── data/              # Training data (JSONL game records)
├── runs/              # Model checkpoints & training outputs
│   ├── bc/            # Behavior cloning runs
│   ├── loop/          # Self-play RL runs
│   └── registry/      # Model registry snapshots
└── tests/             # Test suite
```

## Current Status

The CNN model is trained via behavior cloning and self-play RL. It plays legal chess but is not yet competitive with Stockfish.

See [ROADMAP.md](ROADMAP.md) for next steps and research directions.

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/

# Format
ruff format src/
```

## License

MIT