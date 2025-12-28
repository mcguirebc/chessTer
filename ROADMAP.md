# ChessTer Roadmap

## Current Status

- CNN model trained via behavior cloning + self-play RL
- Model plays legal chess but is not yet competitive with Stockfish
- Training data has been scaled up, but more iterations needed
- **ELO rating system implemented** (leaderboard, gating integration, API endpoint)
- **HuggingFace LLM policy implemented** (`policies/hf_llm.py`) with ELO-aware prompts
- **LoRA training policy implemented** (`policies/llm_lora.py`) for RL fine-tuning
- **REINFORCE trainer implemented** (`train/llm_rl.py`) with outcome rewards
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

**Phase 2a (Completed):** LoRA + REINFORCE training infrastructure

```
src/chesster/
├── policies/
│   └── llm_lora.py        # LoRA-wrapped LLM for training (DONE)
├── train/
│   └── llm_rl.py          # REINFORCE trainer with outcome rewards (DONE)
```

**Phase 2.5 (Next):** Model comparison (Qwen vs DeepSeek)

Run training on both models, compare results, decide which to scale.

### Dependencies

```bash
pip install chesster[llm]  # transformers, accelerate, bitsandbytes, peft, trl
```

### CLI Usage

```bash
# Run MVP training (100 games, 1 epoch)
python -m chesster.train.llm_rl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --games 100 \
    --epochs 1 \
    --output runs/llm/qwen_v1

# Run with DeepSeek for comparison
python -m chesster.train.llm_rl \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --games 100 \
    --epochs 1 \
    --output runs/llm/deepseek_v1
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

## Phase 2 Decision Tree

After running the MVP, evaluate results and follow this decision tree:

### Phase 2a: MVP Validation (Current)

| Metric | Success Criteria |
|--------|------------------|
| Training completes | No OOM on M4 16GB |
| Loss trend | Decreasing (not flat) |
| Win rate vs Random | > 50% |

### Phase 2.5: Model Comparison

Once Phase 2a succeeds, run training on both models:

| Model | Games | Epochs | Evaluation |
|-------|-------|--------|------------|
| Qwen2.5-1.5B-Instruct | 500 | 3 | 50 games vs Stockfish |
| DeepSeek-R1-Distill-Qwen-1.5B | 500 | 3 | 50 games vs Stockfish |

**Comparison Metrics:**
- Win rate vs Random (target: >70%)
- Win rate vs Stockfish depth 5
- Training stability (loss variance)
- Inference speed (ms/move)

**Decision Matrix:**

| Qwen Result | DeepSeek Result | Next Step |
|-------------|-----------------|-----------|
| Good (>60% vs Random) | Bad (<55%) | Scale Qwen only |
| Bad (<55%) | Good (>60%) | Scale DeepSeek only |
| Good | Good | Scale both, compare at 1000+ games |
| Bad | Bad | Phase 2b: Try alternative approaches |

### Phase 2b/2c/2d: Alternative Approaches (if needed)

| Phase | Condition | Action |
|-------|-----------|--------|
| 2b | REINFORCE too unstable | Try PPO via TRL |
| 2c | No learning signal | Add dense Stockfish rewards (cp_delta) |
| 2d | PPO also fails | Try DPO with preference pairs |

### Phase 3: Scale Up (after Phase 2.5 succeeds)

| Parameter | Value |
|-----------|-------|
| Model(s) | Winner(s) from Phase 2.5 |
| Games | 1000+ per epoch |
| Epochs | 10+ |
| Integration | Gating system, ELO tracking |
| Evaluation | Head-to-head vs CNN approach |

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
| P0 | LoRA training policy (`llm_lora.py`) | **Done** |
| P0 | REINFORCE trainer (`llm_rl.py`) | **Done** |
| P0 | Phase 2a: Run MVP training (100 games) | **Next** |
| P0 | Phase 2.5: Qwen vs DeepSeek comparison | Next |
| P1 | Phase 3: Scale winner(s) to 1000+ games | Todo |
| P1 | Compare CNN vs LLM approaches | Todo |
| P2 | PPO/A2C implementation (if REINFORCE unstable) | Contingent |
| P2 | Dense Stockfish rewards (if no learning) | Contingent |
| P2 | GCP setup with GPU/TPU | Todo |
| P3 | Frontend dashboard | Todo |
| P3 | Distributed training | Todo |

---

## Timeline Estimates

| Phase | Tasks | Duration |
|-------|-------|----------|
| ~~Phase 0~~ | ~~ELO rating system~~ | **Done** |
| ~~Phase 1a~~ | ~~HuggingFace LLM inference policy~~ | **Done** |
| ~~Phase 1b~~ | ~~Test inference with models~~ | **Done** |
| ~~Phase 2a~~ | ~~LoRA + REINFORCE infrastructure~~ | **Done** |
| ~~Phase 2a-run~~ | ~~MVP validation (pipeline works, 45% valid moves)~~ | **Done** |
| Phase 2.5-local | Play more games, tune prompts for higher valid rate | ~1 day |
| Phase 2.5-GCP | Full REINFORCE training (needs GPU memory) | 1 week |
| Phase 3 | Scale winner(s), CNN vs LLM comparison | 1-2 weeks |
| Phase 4 | GCP setup, larger scale training | 2-4 weeks |
| Phase 5 | Frontend dashboard | 2-3 weeks |

### MVP Results (Dec 27, 2024)

**Model Comparison (M4 MacBook):**

| Model | Valid Move Rate | Time (5 games) | Notes |
|-------|----------------|----------------|-------|
| **Qwen2.5-1.5B-Instruct** | **99.1%** | 40s | Best for direct move output |
| DeepSeek-R1-Distill-Qwen-1.5B | 0% | 123s | Outputs reasoning text, not moves |

**Winner: Qwen** - DeepSeek is a reasoning model that "thinks" before answering, making it unsuitable for direct UCI output.

**Key findings:**
- Simpler prompts work better ("Pick from list" vs complex FEN prompts)
- System prompt is essential for instruction following
- LoRA adapters save/load correctly
- REINFORCE backprop needs GPU (OOM on 16GB unified memory)
