from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from chesster.api.idempotency import IdempotencyStore
from chesster.api.settings import ApiSettings
from chesster.policies.ollama import OllamaPolicy
from chesster.policies.random import RandomPolicy
from chesster.policies.stockfish import StockfishPolicy


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()


@dataclass(frozen=True)
class PolicyRegistry:
    policies: dict[str, object]

    def list(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for pid, policy in self.policies.items():
            desc = getattr(policy, "description", "")
            out.append((pid, str(desc)))
        out.sort(key=lambda x: x[0])
        return out

    def get(self, policy_id: str) -> object | None:
        return self.policies.get(policy_id)


@lru_cache
def get_policy_registry() -> PolicyRegistry:
    s = get_settings()
    return PolicyRegistry(
        policies={
            "random": RandomPolicy(),
            "stockfish": StockfishPolicy(
                stockfish_path=s.stockfish_path, default_depth=s.stockfish_default_depth
            ),
            "ollama": OllamaPolicy(
                base_url=s.ollama_base_url,
                default_model=s.ollama_default_model,
                timeout_s=s.ollama_timeout_s,
            ),
        }
    )


@lru_cache
def get_idempotency_store() -> IdempotencyStore:
    s = get_settings()
    return IdempotencyStore(db_path=s.idempotency_db_path)




