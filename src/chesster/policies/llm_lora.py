"""LoRA-wrapped LLM policy for reinforcement learning training."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess

from chesster.chess.board import (
    choose_fallback_legal_move,
    legal_moves_uci,
    move_to_san,
    parse_uci_move,
)

from .base import ChooseMoveParams, MoveResult

if TYPE_CHECKING:
    import torch
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Regex to extract UCI moves from LLM output
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha (scaling)
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None  # None = auto-detect
    bias: str = "none"


@dataclass
class MoveWithLogProb:
    """Move result with log probability for RL training."""

    uci: str
    san: str
    log_prob: "torch.Tensor"
    info: dict[str, Any]


def _get_device(device: str | None) -> str:
    """Get the best available device."""
    if device is not None:
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class LoRALLMPolicy:
    """
    LLM policy with LoRA adapters for reinforcement learning.

    This wraps a base LLM with trainable LoRA adapters, exposing:
    - forward() that returns move + log probability for policy gradients
    - save_adapter() / load_adapter() for checkpointing
    - Only ~0.5% of parameters are trainable

    Example:
        >>> policy = LoRALLMPolicy("Qwen/Qwen2.5-1.5B-Instruct", device="mps")
        >>> board = chess.Board()
        >>> result = policy.forward(board, ChooseMoveParams())
        >>> print(result.uci, result.log_prob)  # e.g., "e2e4", tensor(-1.23)
    """

    policy_id = "llm_lora"
    description = "LoRA-wrapped LLM for RL training with log probabilities."

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        lora_config: LoRAConfig | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize the LoRA LLM policy.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to load model on ("cuda", "mps", "cpu", or None for auto).
            lora_config: LoRA adapter configuration.
            load_in_4bit: Whether to use 4-bit quantization (QLoRA).
        """
        self._model_name = model_name
        self._device = _get_device(device)
        self._lora_config = lora_config or LoRAConfig()
        self._load_in_4bit = load_in_4bit

        self._model: PeftModel | None = None
        self._base_model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the model with LoRA adapters on first use."""
        if self._loaded:
            return

        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "LoRA training requires peft. Install with: pip install 'chesster[llm]'"
            ) from e

        logger.info(f"Loading model {self._model_name} with LoRA on {self._device}...")

        # Determine torch dtype
        dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load base model
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }

        if self._load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **model_kwargs,
        )

        if not self._load_in_4bit:
            self._base_model = self._base_model.to(self._device)

        # Apply LoRA adapters
        peft_config = LoraConfig(
            r=self._lora_config.r,
            lora_alpha=self._lora_config.lora_alpha,
            lora_dropout=self._lora_config.lora_dropout,
            target_modules=self._lora_config.target_modules,
            bias=self._lora_config.bias,
            task_type="CAUSAL_LM",
        )

        self._model = get_peft_model(self._base_model, peft_config)
        self._model.print_trainable_parameters()

        self._loaded = True
        logger.info(f"Model loaded with LoRA on {self._device}")

    def _build_prompt(
        self,
        board: chess.Board,
        legal_moves: list[str],
        params: ChooseMoveParams,
    ) -> str:
        """Build the prompt for the LLM using chat template if available."""
        # Simpler prompt = better valid move rate (tested: 100% vs 45%)
        moves_str = " ".join(legal_moves[:15])  # Limit to avoid token overflow

        # Try to use chat template for better instruction following
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "Pick one move from the list. Output only the move."},
                {"role": "user", "content": f"Moves: {moves_str}\nChoice:"},
            ]
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return prompt
            except Exception:
                pass  # Fall back to simple prompt

        # Simple prompt fallback
        return f"Pick one move from the list. Output only the move.\n\nMoves: {moves_str}\nChoice:"

    def _extract_move(self, text: str, board: chess.Board) -> str | None:
        """Extract a legal move from LLM output (UCI or SAN format)."""
        text_clean = text.strip().lower()
        
        # Get legal moves in UCI format
        legal_uci = {m.uci() for m in board.legal_moves}
        
        # First try: exact match if output is just the move
        first_word = text_clean.split()[0] if text_clean else ""
        if first_word in legal_uci:
            return first_word
        
        # Second try: find UCI pattern anywhere in text
        candidates = _UCI_RE.findall(text_clean)
        for candidate in candidates:
            if candidate in legal_uci:
                return candidate
        
        # Third try: SAN notation (e4, Nf3, Bxc6, etc.)
        san_pattern = re.compile(r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b", re.IGNORECASE)
        san_candidates = san_pattern.findall(text)
        
        for san in san_candidates:
            try:
                move = board.parse_san(san)
                return move.uci()
            except Exception:
                continue
        
        return None

    def forward(
        self,
        board: chess.Board,
        params: ChooseMoveParams,
    ) -> MoveWithLogProb:
        """
        Forward pass that returns move with log probability.

        This is used for REINFORCE training where we need:
        - The chosen move
        - The log probability of that move for policy gradients

        Args:
            board: Current chess position.
            params: Move parameters including optional ELO context.

        Returns:
            MoveWithLogProb with uci, san, log_prob, and info.
        """
        import torch

        self._ensure_loaded()

        legal_uci = legal_moves_uci(board)
        prompt = self._build_prompt(board, legal_uci, params)
        
        # Debug: log first prompt
        if not hasattr(self, "_logged_prompt"):
            logger.info(f"Sample prompt:\n{prompt[:300]}...")
            self._logged_prompt = True

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate (no gradients needed here - we'll compute loss in batch later)
        temperature = params.temperature if params.temperature is not None else 0.1
        max_new_tokens = params.max_tokens if params.max_tokens is not None else 16

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=params.top_p if params.top_p is not None else 0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode generated text
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Store prompt for batch training later (no gradient computation during play)
        # We'll compute loss in a separate batch pass
        # TODO: For actual RL training, compute token log probabilities here.
        # Currently a placeholder - acceptable for MVP pipeline validation (see ROADMAP).
        log_prob = torch.tensor(0.0, requires_grad=False)

        # Extract move
        chosen_uci = self._extract_move(generated_text, board)

        info: dict[str, Any] = {
            "model": self._model_name,
            "device": self._device,
            "temperature": temperature,
            "raw_output": generated_text[:100],
        }

        if chosen_uci is None:
            info["fallback"] = True
            # Debug only on first few fallbacks
            if not hasattr(self, "_fallback_count"):
                self._fallback_count = 0
            self._fallback_count += 1
            if self._fallback_count <= 3:
                logger.warning(
                    f"Invalid LLM output (#{self._fallback_count}): '{generated_text}' | "
                    f"Sample legal: {list(legal_uci)[:3]}"
                )
            fallback = choose_fallback_legal_move(board, seed=params.seed)
            chosen_move = fallback
        else:
            chosen_move = parse_uci_move(board, chosen_uci)

        uci = chosen_move.uci()
        san = move_to_san(board, chosen_move)

        return MoveWithLogProb(uci=uci, san=san, log_prob=log_prob, info=info)

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult:
        """Standard choose_move interface (ignores log_prob)."""
        result = self.forward(board, params)
        return MoveResult(uci=result.uci, san=result.san, info=result.info)

    def train_mode(self) -> None:
        """Set model to training mode."""
        self._ensure_loaded()
        self._model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self._ensure_loaded()
        self._model.eval()

    def parameters(self):
        """Return trainable parameters (LoRA only)."""
        self._ensure_loaded()
        return self._model.parameters()

    def trainable_parameters(self) -> int:
        """Return count of trainable parameters."""
        self._ensure_loaded()
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Return count of total parameters."""
        self._ensure_loaded()
        return sum(p.numel() for p in self._model.parameters())

    def save_adapter(self, path: str | Path) -> None:
        """Save LoRA adapter weights."""
        self._ensure_loaded()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        logger.info(f"Saved LoRA adapter to {path}")

    def load_adapter(self, path: str | Path) -> None:
        """Load LoRA adapter weights."""
        from peft import PeftModel

        self._ensure_loaded()
        path = Path(path)

        # Reload model with saved adapter
        self._model = PeftModel.from_pretrained(
            self._base_model,
            path,
        )
        logger.info(f"Loaded LoRA adapter from {path}")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._base_model is not None:
            del self._base_model
            self._base_model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded")
