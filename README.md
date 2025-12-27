# chessTer

A chess AI training research playground combining neural network policies, LLM-based move generation, and reinforcement learning via self-play.

**Goal**: Train chess-playing agents through imitation learning (behavior cloning from Stockfish) and reinforcement learning, comparing CNN-based and LLM-based approaches.

## Architecture

```
+------------------------------------------------------------------+
|                         FastAPI Server                            |
|                     /v1/move, /v1/policies                        |
+----------------------------------+-------------------------------+
                                   |
         +-------------------------+-------------------------+
         v                         v                         v
+----------------+         +----------------+         +----------------+
|  RandomPolicy  |         |StockfishPolicy |         |  OllamaPolicy  |
|   (baseline)   |         |   (teacher)    |         |  (LLM-based)   |
+----------------+         +----------------+         +----------------+
         |                         |                         |
         +-------------------------+-------------------------+
                                   v
+------------------------------------------------------------------+
|                       SmallNetPolicy (CNN)                        |
|           Residual blocks + policy head + value head              |
+------------------------------------------------------------------+
                                   |
         +-------------------------+-------------------------+
         v                                                   v
+--------------------+                           +--------------------+
|  Behavior Cloning  |                           |    RL Training     |
|  (imitate teacher) |                           |    (REINFORCE)     |
+--------------------+                           +--------------------+
                                   |
                                   v
+------------------------------------------------------------------+
|                        Self-Play Loop                             |
|    generate games -> train -> gate against baseline -> promote    |
+------------------------------------------------------------------+
```

## Completed Phases

### Phase 1: API + Pluggable Policies
- FastAPI server with idempotent `/v1/move` endpoint
- Pluggable policy architecture: Random, Stockfish, Ollama LLM
- SQLite-backed idempotency store

### Phase 2: Behavior Cloning
- `SmallChessNet`: CNN with 4 residual blocks, policy head (4672 moves), value head
- Supervised learning to imitate Stockfish best moves
- Training data: games annotated with teacher moves

### Phase 3: Self-Play RL Loop
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
pip install -e ".[dev,train]"

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
  -H "content-type: application/json" \
  -d "{"fen":"startpos","policy_id":"random","idempotency_key":"demo-1"}"
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

---

## Roadmap and Next Steps

### Near Term

#### 1. Scale Up Training Data
Current BC model trained on only ~3K positions. Need more data:
```bash
# Generate 10K+ annotated positions
python -m chesster.selfplay.generator \
  --policy random \
  --opponents stockfish \
  --games 500 \
  --annotate \
  --teacher-depth 20 \
  --out data/teacher_games_10k.jsonl
```

#### 2. Add ELO Rating System
Track player strength over time with proper ELO calculations:
- Calculate ELO after each gating match
- Track rating history per snapshot
- Add ELO to registry metadata
- Leaderboard endpoint: `GET /v1/leaderboard`

#### 3. Improve RL Training
- Experiment with `--reward cp_delta` for denser feedback
- Implement PPO/A2C for more stable policy gradients
- Add temperature scheduling during self-play

---

### LLM Fine-Tuning with RL (Key Research Direction)

Instead of training a CNN from scratch, fine-tune a small LLM to play chess using RL:

#### Approach
1. **Base Model**: Use a small LLM via Ollama (e.g., `phi3`, `llama3.2:1b`, `qwen2.5:0.5b`)
2. **LoRA/QLoRA**: Parameter-efficient fine-tuning on chess positions
3. **Reward Signal**: Same as CNN approach (match_bestmove, cp_delta, outcome)
4. **Self-Play**: LLM vs LLM or LLM vs CNN

#### Implementation Plan
```
src/chesster/
├── policies/
│   └── llm_finetune.py    # LoRA-wrapped LLM policy
├── train/
│   └── llm_rl.py          # RL trainer for LLM weights
```

#### Key Questions to Answer
- Can a 1B parameter LLM learn to play decent chess through RL?
- Does the LLM world knowledge help or hurt?
- How does sample efficiency compare to CNN?
- Can we get emergent reasoning about chess positions?

#### Technical Requirements
- `transformers`, `peft` (LoRA), `bitsandbytes` (quantization)
- Higher VRAM requirements (8GB+ for 1B model)
- Gradient checkpointing for memory efficiency

---

### Cloud Training (GCP)

#### Phase 1: Single GPU Training
- Vertex AI Workbench or Compute Engine with T4/A100
- Containerize training with Docker
- Store checkpoints in GCS bucket

#### Phase 2: Distributed Training
- Multi-GPU with `torchrun` or Vertex AI Training
- TPU v3/v4 for larger scale experiments
- Experiment tracking with Weights and Biases or Vertex AI Experiments

#### Infrastructure Setup
```bash
# Example: Create GCP VM with GPU
gcloud compute instances create chesster-train \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release
```

---

### Frontend Dashboard

Real-time training visualization and game viewing:

#### Features
- **Leaderboard**: ELO rankings of all registered models
- **Training Progress**: Loss curves, accuracy, games played
- **Live Games**: Watch self-play games as they happen
- **Game Browser**: Replay historical games with analysis
- **Model Comparison**: Head-to-head stats between checkpoints

#### Tech Stack Options
- **Frontend**: React/Next.js or Svelte
- **Real-time**: WebSockets for live game updates
- **Visualization**: D3.js for charts, chessboard.js for board rendering
- **Backend**: Extend existing FastAPI with additional endpoints

#### API Endpoints to Add
```
GET  /v1/leaderboard           # ELO rankings
GET  /v1/training/status       # Current training progress
WS   /v1/games/live            # WebSocket for live games
GET  /v1/games/{game_id}       # Fetch specific game
GET  /v1/models/{name}/stats   # Model statistics
```

---

### Full Roadmap

| Priority | Task | Status |
|----------|------|--------|
| P0 | Scale training data (10K+ positions) | Todo |
| P0 | More RL iterations on M4 | Todo |
| P1 | ELO rating system | Todo |
| P1 | LLM policy with LoRA fine-tuning | Todo |
| P1 | Compare CNN vs LLM approaches | Todo |
| P2 | PPO/A2C implementation | Todo |
| P2 | MCTS for inference-time search | Todo |
| P2 | GCP setup with GPU/TPU | Todo |
| P3 | Frontend dashboard | Todo |
| P3 | Distributed training | Todo |

---

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
