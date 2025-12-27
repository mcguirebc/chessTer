# ChessTer Roadmap

## Current Status

- CNN model trained via behavior cloning + self-play RL
- Model plays legal chess but is not yet competitive with Stockfish
- Training data has been scaled up, but more iterations needed
- **ELO rating system implemented** (leaderboard, gating integration, API endpoint)
- **HuggingFace LLM policy implemented** (`policies/hf_llm.py`) with ELO-aware prompts
- Exploring two parallel research tracks: improve CNN vs try LLM fine-tuning

---

## Research Track A: LLM Fine-Tuning with RL (Primary Focus)

Fine-tune a small LLM to play chess using reinforcement learning. This is the primary research direction.

### Recommended Models for M4 MacBook

| Model | Params | Memory (4-bit) | Reasoning | Notes |
|-------|--------|----------------|-----------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | ~2GB | Excellent | Best for reasoning tasks |
| Phi-3-mini | 3.8B | ~3GB | Excellent | Microsoft's reasoning model |
| Qwen2.5:1.5b | 1.5B | ~2GB | Good | General capability |
| Qwen2.5:3b | 3B | ~2.5GB | Very Good | If memory allows |

For 16GB M4: Train 1B-3B models with LoRA
For 24GB+ M4: Can train up to 7B models with QLoRA

### Quick Test with Ollama

```bash
# Install and test inference first
ollama pull deepseek-r1:1.5b
ollama run deepseek-r1:1.5b "You are a chess engine. Given FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1, what is the best move in UCI format? Reply with only the move."
```

### Implementation Plan

**Phase 1 (Completed):** HuggingFace LLM inference

```
src/chesster/
├── policies/
│   └── hf_llm.py          # HuggingFace LLM policy with ELO context (DONE)
```

**Phase 2 (Next):** RL training

```
src/chesster/
├── policies/
│   └── llm_finetune.py    # LoRA-wrapped LLM for training
├── train/
│   └── llm_rl.py          # RL trainer (approach TBD - see research question below)
```

### Dependencies

```bash
pip install chesster[llm]  # transformers, accelerate, bitsandbytes
pip install peft trl       # For Phase 2 RL training
```

### Phase 2 RL Training Approach (Open Research Question)

Several approaches exist for RL fine-tuning of LLMs. Phase 2 experiments will determine what works best for chess:

| Approach | Pros | Cons | Libraries |
|----------|------|------|-----------|
| **LoRA + REINFORCE** | Simple, low memory | High variance | Custom |
| **LoRA + PPO** | More stable than REINFORCE | Complex, needs value head | TRL |
| **DPO (Direct Preference Optimization)** | No reward model needed | Needs preference pairs | TRL |
| **Full fine-tuning** | Maximum capacity | High memory, slow | transformers |

**Key questions to answer:**
- Does PPO's stability matter for chess (we have dense rewards from Stockfish)?
- Can we use Stockfish evaluations directly as rewards?
- Is LoRA sufficient or do we need more trainable parameters?
- How does sample efficiency compare to CNN approach?

**Recommendation:** Start with **LoRA + REINFORCE** (simplest), measure variance. If too unstable, try **PPO** via TRL.

### Approach (Phase 2)

1. **Base Model**: Download weights from HuggingFace (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
2. **LoRA/QLoRA**: Parameter-efficient fine-tuning (~0.1-1% of weights)
3. **Reward Signal**: Same as CNN approach (match_bestmove, cp_delta, outcome)
4. **Training**: Start with custom REINFORCE, upgrade to TRL's `PPOTrainer` if needed
5. **Self-Play**: LLM vs LLM or LLM vs CNN for comparison

### Opponent Context (LLM Advantage)

Unlike CNNs, LLMs can leverage contextual information in the prompt:

- **Opponent ELO**: Adjust strategy based on opponent strength
- **Bot vs Human**: Different play styles (bots are more consistent, humans make psychological errors)
- **Playing Style**: Aggressive, defensive, positional, tactical

Example prompt with context:
```
You are playing chess against a 1200 ELO bot opponent.
Position (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
Your ELO: 1800

Given the skill gap, what is your best move in UCI format?
```

This context could help the model:
- Play more aggressively against weaker opponents
- Be more solid/defensive against stronger opponents
- Exploit known bot patterns vs adapt to human psychology

### Research Questions

- Can a 1B parameter LLM learn to play decent chess through RL?
- Does the LLM's pre-trained world knowledge help or hurt?
- How does sample efficiency compare to CNN?
- Can we get emergent reasoning about chess positions?
- Does chain-of-thought prompting improve move quality?
- Does opponent ELO context improve adaptation and win rate?

---

## Research Track B: Improve CNN Approach

Continue improving the existing SmallNetPolicy CNN.

### Near Term Improvements

- **PPO/A2C**: More stable policy gradients than vanilla REINFORCE
- **Temperature Scheduling**: Anneal temperature during self-play
- **More Training Data**: Continue scaling up annotated positions

### Medium Term

- **MCTS**: Monte Carlo Tree Search at inference time using value head
- **Larger Network**: More residual blocks, wider channels
- **Better Rewards**: Combine match_bestmove + cp_delta + outcome

---

## ELO Rating System (Completed)

Track model strength over time with proper ELO calculations.

### Implemented Features

- **ELO Calculations** (`league/elo.py`): Standard ELO formula with K=32
- **Registry Integration**: `SnapshotInfo` includes `elo`, `elo_history`, `games_played`, `is_bot`
- **Gating Integration**: ELO updates calculated after each gating match
- **Leaderboard API**: `GET /v1/leaderboard` returns models sorted by ELO
- **Initial Ratings**: Random=800, Stockfish=2800, Default=1200

### Why ELO Matters for LLM Training

ELO data serves dual purposes:
1. **Evaluation**: Measure model improvement over time
2. **LLM Context**: Feed opponent ELO into prompts during training/inference

This creates a feedback loop where the LLM learns to adapt its play based on opponent strength.

### API Endpoint

```bash
curl http://127.0.0.1:8000/v1/leaderboard
```

Returns:
```json
{
  "rankings": [
    {"name": "v0042", "elo": 1523, "games_played": 150, "is_bot": true},
    {"name": "init", "elo": 1200, "games_played": 100, "is_bot": true}
  ]
}
```

---

## Priority 2: Infrastructure

### GCP Cloud Training

#### Phase 1: Single GPU
- Vertex AI Workbench or Compute Engine with T4/A100
- Containerize training with Docker
- Store checkpoints in GCS bucket

#### Phase 2: Distributed Training
- Multi-GPU with `torchrun` or Vertex AI Training
- TPU v3/v4 for larger scale experiments
- Experiment tracking with Weights & Biases

#### Example Setup

```bash
# Create GCP VM with GPU
gcloud compute instances create chesster-train \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release
```

### Frontend Dashboard

Real-time training visualization and game viewing.

#### Features

- **Leaderboard**: ELO rankings of all registered models
- **Training Progress**: Loss curves, accuracy, games played
- **Live Games**: Watch self-play games as they happen
- **Game Browser**: Replay historical games with analysis
- **Model Comparison**: Head-to-head stats between checkpoints

#### Tech Stack

- **Frontend**: React/Next.js or Svelte
- **Real-time**: WebSockets for live game updates
- **Visualization**: D3.js for charts, chessboard.js for board rendering
- **Backend**: Extend existing FastAPI

#### API Endpoints to Add

```
GET  /v1/leaderboard           # ELO rankings
GET  /v1/training/status       # Current training progress
WS   /v1/games/live            # WebSocket for live games
GET  /v1/games/{game_id}       # Fetch specific game
GET  /v1/models/{name}/stats   # Model statistics
```

---

## Full Roadmap

| Priority | Task | Status |
|----------|------|--------|
| P0 | ELO rating system | **Done** |
| P0 | HuggingFace LLM inference policy (`hf_llm.py`) | **Done** |
| P0 | LLM RL training experiments (DeepSeek-R1) | **Next** |
| P0 | Implement `llm_finetune.py` and `llm_rl.py` | Next |
| P1 | Compare CNN vs LLM approaches | Todo |
| P2 | PPO/A2C implementation for CNN | Todo |
| P2 | MCTS for inference-time search | Todo |
| P2 | GCP setup with GPU/TPU | Todo |
| P3 | Frontend dashboard | Todo |
| P3 | Distributed training | Todo |

---

## Timeline Estimates

| Phase | Tasks | Duration |
|-------|-------|----------|
| ~~Phase 0~~ | ~~ELO rating system~~ | **Done** |
| ~~Phase 1a~~ | ~~HuggingFace LLM inference policy~~ | **Done** |
| Phase 1b | Test inference with DeepSeek-R1 model | ~1 day |
| Phase 2 | LoRA RL training loop (`llm_rl.py`) | 2-3 weeks |
| Phase 3 | CNN vs LLM comparison | 1-2 weeks |
| Phase 4 | GCP setup, larger scale training | 2-4 weeks |
| Phase 5 | Frontend dashboard | 2-3 weeks |
