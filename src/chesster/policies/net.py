"""SmallNetPolicy: a trainable neural network policy for chess."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chesster.chess.board import legal_moves_uci, move_to_san

from .base import ChooseMoveParams, MoveResult

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Move vocabulary: AlphaZero-style encoding
# ---------------------------------------------------------------------------
# We encode moves as (from_square, to_square, promotion_type).
# from_square: 0-63, to_square: 0-63, promotion: 0 (none), 1-4 (q,r,b,n)
# Total: 64 * 64 * 5 = 20480 but many are invalid. We use a flat index.
# For simplicity, we use a smaller encoding: all legal UCI strings mapped to indices.
# This is built dynamically per position (legal move masking).

# For a fixed vocabulary, we use the 4672 move encoding from AlphaZero:
# - Queen moves from each square in 8 directions * 7 distances = 56 * 64 = 3584
# - Knight moves: 8 * 64 = 512
# - Underpromotions: 3 types * 3 directions * 8 files * 2 (for each side) = 144
# Total: ~4672

# For simplicity, we'll use UCI string hashing with a max vocab size.


def _uci_to_index(uci: str, vocab_size: int = 8192) -> int:
    """Hash a UCI move string to an index."""
    return hash(uci) % vocab_size


def _build_move_vocab() -> tuple[dict[str, int], list[str]]:
    """
    Build a fixed move vocabulary covering all possible UCI moves.

    Returns:
        (uci_to_idx dict, idx_to_uci list)
    """
    moves: set[str] = set()

    # Generate all possible moves from all positions
    # This is expensive, so we enumerate possible UCI patterns instead
    files = "abcdefgh"
    ranks = "12345678"
    promotions = ["", "q", "r", "b", "n"]

    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    from_sq = f1 + r1
                    to_sq = f2 + r2
                    if from_sq == to_sq:
                        continue

                    # Normal moves
                    moves.add(from_sq + to_sq)

                    # Promotions (only from rank 7 to 8 or rank 2 to 1)
                    if (r1 == "7" and r2 == "8") or (r1 == "2" and r2 == "1"):
                        for promo in ["q", "r", "b", "n"]:
                            moves.add(from_sq + to_sq + promo)

    # Sort for deterministic ordering
    move_list = sorted(moves)
    uci_to_idx = {uci: i for i, uci in enumerate(move_list)}

    return uci_to_idx, move_list


# Global move vocabulary (computed once)
_MOVE_VOCAB: tuple[dict[str, int], list[str]] | None = None


def get_move_vocab() -> tuple[dict[str, int], list[str]]:
    """Get the global move vocabulary."""
    global _MOVE_VOCAB
    if _MOVE_VOCAB is None:
        _MOVE_VOCAB = _build_move_vocab()
    return _MOVE_VOCAB


def vocab_size() -> int:
    """Return the move vocabulary size."""
    uci_to_idx, _ = get_move_vocab()
    return len(uci_to_idx)


# ---------------------------------------------------------------------------
# Board feature extraction
# ---------------------------------------------------------------------------


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board to a feature tensor.

    Returns:
        Tensor of shape (19, 8, 8) with:
        - 12 planes for pieces (P,N,B,R,Q,K for white, p,n,b,r,q,k for black)
        - 1 plane for side to move
        - 2 planes for castling rights (kingside, queenside) per side
        - 2 planes for en passant (file indicator)
    """
    planes = torch.zeros(19, 8, 8, dtype=torch.float32)

    # Piece planes (0-11)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = square // 8
        file = square % 8
        # White pieces: 0-5 (P,N,B,R,Q,K)
        # Black pieces: 6-11 (p,n,b,r,q,k)
        plane_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        planes[plane_idx, rank, file] = 1.0

    # Side to move (plane 12)
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights (planes 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # En passant (planes 17-18)
    if board.ep_square is not None:
        ep_file = board.ep_square % 8
        planes[17, :, ep_file] = 1.0
        planes[18, board.ep_square // 8, ep_file] = 1.0

    return planes


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """
    Create a mask over the move vocabulary for legal moves.

    Returns:
        Boolean tensor of shape (vocab_size,) where True = legal.
    """
    uci_to_idx, _ = get_move_vocab()
    mask = torch.zeros(len(uci_to_idx), dtype=torch.bool)

    for move in board.legal_moves:
        uci = move.uci()
        if uci in uci_to_idx:
            mask[uci_to_idx[uci]] = True

    return mask


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------


class SmallChessNet(nn.Module):
    """
    A small convolutional network for chess move prediction.

    Architecture:
    - Input: (batch, 19, 8, 8)
    - Conv layers with residual connections
    - Policy head: outputs logits over move vocabulary
    - Value head: outputs scalar value estimate
    """

    def __init__(
        self,
        num_channels: int = 128,
        num_blocks: int = 4,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()

        if vocab_size is None:
            vocab_size = len(get_move_vocab()[0])

        self.vocab_size = vocab_size

        # Initial conv
        self.conv_in = nn.Conv2d(19, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_channels),
                    nn.ReLU(),
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_channels),
                )
            )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, vocab_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 19, 8, 8).
            legal_mask: Optional boolean mask of shape (batch, vocab_size).

        Returns:
            (policy_logits, value) tuple.
            policy_logits: shape (batch, vocab_size)
            value: shape (batch, 1)
        """
        # Initial conv
        out = F.relu(self.bn_in(self.conv_in(x)))

        # Residual blocks
        for block in self.blocks:
            residual = out
            out = block(out)
            out = F.relu(out + residual)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Apply legal move mask (set illegal moves to -inf)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value


# ---------------------------------------------------------------------------
# SmallNetPolicy: wraps the network for use as a Policy
# ---------------------------------------------------------------------------


class SmallNetPolicy:
    """A trainable neural network policy."""

    policy_id = "smallnet"
    description = "Small convolutional neural network policy (trainable)."

    def __init__(
        self,
        model: SmallChessNet | None = None,
        device: str = "cpu",
        temperature: float = 0.0,
    ) -> None:
        """
        Initialize the policy.

        Args:
            model: Pre-trained model, or None to create a new one.
            device: Device to run inference on.
            temperature: Sampling temperature (0 = greedy).
        """
        if model is None:
            model = SmallChessNet()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.temperature = temperature

        self._uci_to_idx, self._idx_to_uci = get_move_vocab()

    def choose_move(self, board: chess.Board, params: ChooseMoveParams) -> MoveResult:
        """Choose a move using the neural network."""
        started = time.perf_counter()

        # Extract features
        features = board_to_tensor(board).unsqueeze(0).to(self.device)
        mask = legal_move_mask(board).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits, value = self.model(features, mask)

        # Get probabilities
        temperature = params.temperature if params.temperature is not None else self.temperature

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            # Sample
            if params.seed is not None:
                torch.manual_seed(params.seed)
            idx = torch.multinomial(probs[0], 1).item()
        else:
            # Greedy
            idx = logits[0].argmax().item()

        uci = self._idx_to_uci[idx]

        # Verify legality (should always be legal due to masking)
        legal_uci = legal_moves_uci(board)
        if uci not in legal_uci:
            # Fallback to first legal move (shouldn't happen)
            uci = legal_uci[0]

        move = chess.Move.from_uci(uci)
        san = move_to_san(board, move)

        took_ms = int((time.perf_counter() - started) * 1000)

        info: dict[str, Any] = {
            "took_ms": took_ms,
            "value": float(value[0, 0]),
            "temperature": temperature,
        }

        return MoveResult(uci=uci, san=san, info=info)

    def forward(
        self, board: chess.Board
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Returns:
            (logits, value, legal_mask) tuple.
        """
        features = board_to_tensor(board).unsqueeze(0).to(self.device)
        mask = legal_move_mask(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(features, mask)

        return logits, value, mask

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()

