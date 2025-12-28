"""Policy loading factory for evaluation.

Supports loading any policy type from a unified URI format:
    - hf:Qwen/Qwen2.5-1.5B-Instruct     # HuggingFace model
    - pt:runs/bc/v1/best_model.pt       # PyTorch checkpoint
    - reg:init                           # Registry snapshot
    - stockfish:5                        # Stockfish at depth 5
    - stockfish                          # Stockfish at default depth
    - random                             # Random policy

Example:
    >>> policy = load_policy("hf:Qwen/Qwen2.5-1.5B-Instruct")
    >>> policy = load_policy("stockfish:10")
    >>> policy = load_policy("random")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesster.policies.base import Policy

logger = logging.getLogger(__name__)


class PolicyLoadError(Exception):
    """Error loading a policy from a spec string."""

    pass


def load_policy(
    spec: str,
    *,
    registry_path: str | Path | None = None,
    device: str | None = None,
) -> Policy:
    """
    Load a policy from a spec string.

    Args:
        spec: Policy specification in format "type:arg" or just "type".
            Supported formats:
            - "hf:model_name" - HuggingFace model
            - "pt:checkpoint_path" - PyTorch checkpoint (.pt file)
            - "reg:snapshot_name" - Model from registry
            - "stockfish:depth" or "stockfish" - Stockfish engine
            - "random" - Random policy
        registry_path: Path to model registry (required for "reg:" specs).
        device: Device override for models (None = auto-detect).

    Returns:
        Loaded Policy instance.

    Raises:
        PolicyLoadError: If spec is invalid or loading fails.

    Example:
        >>> policy = load_policy("stockfish:5")
        >>> policy = load_policy("hf:Qwen/Qwen2.5-1.5B-Instruct", device="mps")
    """
    spec = spec.strip()

    # Parse spec into type and argument
    if ":" in spec:
        policy_type, arg = spec.split(":", 1)
    else:
        policy_type = spec
        arg = ""

    policy_type = policy_type.lower()

    try:
        if policy_type == "hf":
            return _load_hf_policy(arg, device=device)
        elif policy_type == "pt":
            return _load_pt_policy(arg, device=device)
        elif policy_type == "reg":
            return _load_registry_policy(arg, registry_path=registry_path, device=device)
        elif policy_type == "stockfish":
            return _load_stockfish_policy(arg)
        elif policy_type == "random":
            return _load_random_policy()
        else:
            raise PolicyLoadError(
                f"Unknown policy type: {policy_type!r}. "
                "Supported: hf, pt, reg, stockfish, random"
            )
    except PolicyLoadError:
        raise
    except Exception as e:
        raise PolicyLoadError(f"Failed to load policy {spec!r}: {e}") from e


def _load_hf_policy(model_name: str, *, device: str | None) -> Policy:
    """Load a HuggingFace LLM policy."""
    if not model_name:
        raise PolicyLoadError("HuggingFace spec requires model name: hf:model_name")

    try:
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy
    except ImportError as e:
        raise PolicyLoadError(
            "HuggingFace dependencies not installed. Run: pip install chesster[llm]"
        ) from e

    logger.info(f"Loading HuggingFace model: {model_name}")
    policy = HuggingFaceLLMPolicy(model_name=model_name, device=device)
    # Update policy_id to include model name
    policy.policy_id = f"hf:{model_name}"
    return policy


def _load_pt_policy(checkpoint_path: str, *, device: str | None) -> Policy:
    """Load a SmallNetPolicy from a PyTorch checkpoint."""
    if not checkpoint_path:
        raise PolicyLoadError("PyTorch spec requires path: pt:path/to/model.pt")

    path = Path(checkpoint_path)
    if not path.exists():
        raise PolicyLoadError(f"Checkpoint not found: {path}")

    try:
        import torch

        from chesster.policies.net import SmallChessNet, SmallNetPolicy
    except ImportError as e:
        raise PolicyLoadError(f"PyTorch dependencies not available: {e}") from e

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Loading PyTorch checkpoint: {path} on {device}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the dict is the state dict itself
            state_dict = checkpoint
    else:
        raise PolicyLoadError(f"Unexpected checkpoint format: {type(checkpoint)}")

    # Create model and load weights
    model = SmallChessNet()
    model.load_state_dict(state_dict)

    policy = SmallNetPolicy(model=model, device=device)
    policy.policy_id = f"pt:{path.name}"
    return policy


def _load_registry_policy(
    snapshot_name: str,
    *,
    registry_path: str | Path | None,
    device: str | None,
) -> Policy:
    """Load a policy from the model registry."""
    if not snapshot_name:
        raise PolicyLoadError("Registry spec requires snapshot name: reg:name")

    if registry_path is None:
        # Default registry path
        registry_path = Path("runs/registry")

    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise PolicyLoadError(f"Registry not found: {registry_path}")

    try:
        from chesster.league.registry import ModelRegistry
    except ImportError as e:
        raise PolicyLoadError(f"Registry dependencies not available: {e}") from e

    logger.info(f"Loading from registry: {snapshot_name}")
    registry = ModelRegistry(registry_path)

    if snapshot_name not in registry.list_snapshots():
        available = ", ".join(registry.list_snapshots()) or "(none)"
        raise PolicyLoadError(
            f"Snapshot {snapshot_name!r} not found in registry. Available: {available}"
        )

    # Get snapshot info
    info = registry.get_snapshot(snapshot_name)
    artifact_path = registry_path / info.artifact_path

    # Load as PyTorch checkpoint
    return _load_pt_policy(str(artifact_path), device=device)


def _load_stockfish_policy(depth_str: str) -> Policy:
    """Load Stockfish policy with optional depth."""
    from chesster.policies.stockfish import StockfishPolicy

    if depth_str:
        try:
            depth = int(depth_str)
        except ValueError:
            raise PolicyLoadError(f"Invalid Stockfish depth: {depth_str!r}")
    else:
        depth = 10  # Default depth

    logger.info(f"Loading Stockfish at depth {depth}")
    policy = StockfishPolicy(default_depth=depth)
    policy.policy_id = f"stockfish:{depth}"
    return policy


def _load_random_policy() -> Policy:
    """Load random policy."""
    from chesster.policies.random import RandomPolicy

    logger.info("Loading Random policy")
    return RandomPolicy()


def get_supported_types() -> list[str]:
    """Return list of supported policy types."""
    return ["hf", "pt", "reg", "stockfish", "random"]


def parse_opponent_specs(specs: str) -> list[str]:
    """
    Parse a comma-separated list of opponent specs.

    Args:
        specs: Comma-separated specs like "stockfish:1,stockfish:5,random"

    Returns:
        List of individual spec strings.
    """
    return [s.strip() for s in specs.split(",") if s.strip()]
