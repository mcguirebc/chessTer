"""Policy backends for choosing chess moves."""

from chesster.policies.base import ChooseMoveParams, MoveResult, Policy
from chesster.policies.random import RandomPolicy

__all__ = [
    "ChooseMoveParams",
    "MoveResult",
    "Policy",
    "RandomPolicy",
]

# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "HuggingFaceLLMPolicy":
        from chesster.policies.hf_llm import HuggingFaceLLMPolicy
        return HuggingFaceLLMPolicy
    if name == "LoRALLMPolicy":
        from chesster.policies.llm_lora import LoRALLMPolicy
        return LoRALLMPolicy
    if name == "OllamaPolicy":
        from chesster.policies.ollama import OllamaPolicy
        return OllamaPolicy
    if name == "StockfishPolicy":
        from chesster.policies.stockfish import StockfishPolicy
        return StockfishPolicy
    if name == "SmallNetPolicy":
        from chesster.policies.net import SmallNetPolicy
        return SmallNetPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")