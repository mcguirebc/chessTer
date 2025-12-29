"""HuggingFace LLM-based chess policy with ELO-aware prompts."""
from __future__ import annotations

import logging
import re
import time
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
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Regex to extract UCI moves from LLM output
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


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


class HuggingFaceLLMPolicy:
    """
    Chess policy using HuggingFace transformer models.

    Supports any causal LM from HuggingFace Hub, with optional:
    - 4-bit quantization for memory efficiency
    - ELO-aware prompts for opponent context
    - MPS (Apple Silicon) acceleration

    Example:
        >>> policy = HuggingFaceLLMPolicy(device="mps")
        >>> board = chess.Board()
        >>> params = ChooseMoveParams(opponent_elo=1200.0, self_elo=1500.0)
        >>> result = policy.choose_move(board, params)
        >>> print(result.uci)  # e.g., "e2e4"
    """

    policy_id = "hf_llm"
    description = "Choose a move via HuggingFace LLM, with optional ELO context."

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: str | None = None,
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
    ) -> None:
        """
        Initialize the HuggingFace LLM policy.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to load model on ("cuda", "mps", "cpu", or None for auto).
            load_in_4bit: Whether to use 4-bit quantization (requires bitsandbytes).
            torch_dtype: Data type for model weights ("auto", "float16", "bfloat16").
        """
        self._model_name = model_name
        self._device = _get_device(device)
        self._load_in_4bit = load_in_4bit
        self._torch_dtype = torch_dtype

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HuggingFace transformers required. Install with: pip install 'chesster[llm]'"
            ) from e

        logger.info(f"Loading model {self._model_name} on {self._device}...")

        # Determine torch dtype
        if self._torch_dtype == "auto":
            dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
        elif self._torch_dtype == "float16":
            dtype = torch.float16
        elif self._torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with optional quantization
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
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")

        # Load model
        if self._load_in_4bit:
            # Quantized models handle device placement automatically
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                device_map="auto",
                **model_kwargs,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **model_kwargs,
            )
            self._model = self._model.to(self._device)

        self._model.eval()
        self._loaded = True
        logger.info(f"Model loaded successfully on {self._device}")

    def _build_prompt(
        self,
        board: chess.Board,
        legal_moves: list[str],
        opponent_elo: float | None = None,
        self_elo: float | None = None,
        opponent_is_bot: bool | None = None,
    ) -> str:
        """
        Build the prompt for the LLM.

        Args:
            board: Current chess position.
            legal_moves: List of legal moves in UCI format.
            opponent_elo: Opponent's ELO rating (optional).
            self_elo: Our ELO rating (optional).
            opponent_is_bot: Whether opponent is a bot (optional).

        Returns:
            Formatted prompt string.
        """
        fen = board.fen()
        side = "White" if board.turn == chess.WHITE else "Black"
        moves_str = " ".join(legal_moves)

        # Build context parts
        context_parts = []
        if opponent_elo is not None:
            opponent_type = "bot" if opponent_is_bot else "human"
            context_parts.append(f"Opponent: {opponent_elo:.0f} ELO {opponent_type}")
        if self_elo is not None:
            context_parts.append(f"Your rating: {self_elo:.0f} ELO")

        # Try to use chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            system_content = "You are a chess engine. Pick one move from the list. Output only the UCI move notation."
            user_content = f"FEN: {fen}\n"
            if context_parts:
                user_content += "\n".join(context_parts) + "\n"
            user_content += f"Legal moves: {moves_str}\nChoice:"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback to plain text if template fails
                pass

        # Plain text fallback
        if context_parts:
            context = "\n".join(context_parts) + "\n\n"
            prompt = (
                f"You are a chess engine playing as {side}.\n"
                f"{context}"
                f"Position (FEN): {fen}\n"
                f"Legal moves (UCI): {moves_str}\n\n"
                f"Choose the best move. Reply with only the UCI move notation."
            )
        else:
            prompt = (
                f"You are a chess engine. Choose the best move for {side}.\n\n"
                f"Position (FEN): {fen}\n"
                f"Legal moves (UCI): {moves_str}\n\n"
                f"Reply with only the UCI move notation."
            )

        return prompt

    def _extract_move(self, text: str, board: chess.Board) -> str | None:
        """
        Extract a legal UCI move from LLM output.

        Args:
            text: Raw LLM output text.
            board: Current board position for move validation.

        Returns:
            UCI move string if found and legal, else None.
        """
        # Find all potential UCI moves in the text
        candidates = _UCI_RE.findall(text.lower())

        for candidate in candidates:
            try:
                # Verify the move is legal
                parse_uci_move(board, candidate)
                return candidate
            except Exception:
                continue

        return None

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult:
        """
        Choose a move using the LLM.

        Args:
            board: Current chess position.
            params: Move parameters including optional ELO context.

        Returns:
            MoveResult with chosen move and metadata.
        """
        self._ensure_loaded()

        started = time.perf_counter()
        legal_uci = legal_moves_uci(board)

        # Build prompt with optional ELO context
        prompt = self._build_prompt(
            board=board,
            legal_moves=legal_uci,
            opponent_elo=params.opponent_elo,
            self_elo=params.self_elo,
            opponent_is_bot=params.opponent_is_bot,
        )

        # Tokenize
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        temperature = params.temperature if params.temperature is not None else 0.1
        max_new_tokens = params.max_tokens if params.max_tokens is not None else 32

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=params.top_p if params.top_p is not None else 0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_text = self._tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        # Extract move
        chosen_uci = self._extract_move(generated_text, board)

        info: dict[str, Any] = {
            "took_ms": int((time.perf_counter() - started) * 1000),
            "model": self._model_name,
            "device": self._device,
            "temperature": temperature,
            "raw_output": generated_text[:200],  # Truncate for logging
        }

        if params.opponent_elo is not None:
            info["opponent_elo"] = params.opponent_elo
        if params.self_elo is not None:
            info["self_elo"] = params.self_elo

        if chosen_uci is None:
            info["invalid_output"] = True
            info["fallback"] = True
            logger.warning(f"Invalid LLM output, falling back to random: {generated_text[:100]}")
            fallback = choose_fallback_legal_move(board, seed=params.seed)
            chosen_move = fallback
        else:
            chosen_move = parse_uci_move(board, chosen_uci)

        uci = chosen_move.uci()
        san = move_to_san(board, chosen_move)

        return MoveResult(uci=uci, san=san, info=info)

    def close(self) -> None:
        """Alias for unload() to match policy interface."""
        self.unload()

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        # Try to free GPU/MPS memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded")
