# ChessTer Roadmap

## Current Status

- CNN model trained via behavior cloning + self-play RL
- Model plays legal chess but is not yet competitive with Stockfish
- Training data has been scaled up, but more iterations needed
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

New files to create:

```
src/chesster/
├── policies/
│   └── llm_finetune.py    # LoRA-wrapped LLM policy
├── train/
│   └── llm_rl.py          # TRL-based PPO/REINFORCE trainer
```

### Dependencies

```bash
pip install transformers peft trl bitsandbytes accelerate
```

### Approach

1. **Base Model**: Download weights from HuggingFace (e.g., `Qwen/Qwen2.5-1.5B`)
2. **LoRA/QLoRA**: Parameter-efficient fine-tuning (~0.1-1% of weights)
3. **Reward Signal**: Same as CNN approach (match_bestmove, cp_delta, outcome)
4. **Training**: Use TRL library's `PPOTrainer` or custom REINFORCE
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

## Priority 1: ELO Rating System

Track model strength over time with proper ELO calculations.

### Features

- Calculate ELO after each gating match
- Track rating history per snapshot
- Add ELO to registry metadata
- Initial ratings: Random=800, Stockfish=2800
- Track `is_bot=True` flag for all trained models

### Why ELO Matters for LLM Training

ELO data serves dual purposes:
1. **Evaluation**: Measure model improvement over time
2. **LLM Context**: Feed opponent ELO into prompts during training/inference

This creates a feedback loop where the LLM learns to adapt its play based on opponent strength.

### API Endpoint

```
GET /v1/leaderboard
```

Returns:
```json
{
  "rankings": [
    {"name": "v0042", "elo": 1523, "games": 150},
    {"name": "init", "elo": 1200, "games": 100}
  ]
}
```

### Implementation

- Add `elo` field to `SnapshotInfo` in registry
- Create `league/elo.py` with ELO calculation functions
- Update gating to record ELO changes

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
| P0 | LLM fine-tuning experiments (DeepSeek-R1, Phi-3) | Todo |
| P0 | Implement `llm_finetune.py` and `llm_rl.py` | Todo |
| P1 | ELO rating system | Todo |
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
| Phase 1 | LLM inference via Ollama, basic fine-tuning setup | 1-2 weeks |
| Phase 2 | LoRA training loop, initial experiments | 2-3 weeks |
| Phase 3 | ELO system, CNN vs LLM comparison | 1-2 weeks |
| Phase 4 | GCP setup, larger scale training | 2-4 weeks |
| Phase 5 | Frontend dashboard | 2-3 weeks |
