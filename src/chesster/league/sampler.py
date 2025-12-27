"""Opponent sampling strategies for self-play training."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesster.league.registry import ModelRegistry, SnapshotInfo
    from chesster.policies.base import Policy


class OpponentSampler(ABC):
    """Base class for opponent sampling strategies."""

    @abstractmethod
    def sample(self, k: int) -> list[SnapshotInfo]:
        """
        Sample k opponents from the registry.

        Args:
            k: Number of opponents to sample.

        Returns:
            List of SnapshotInfo objects.
        """
        ...

    @abstractmethod
    def sample_policies(self, k: int, device: str = "cpu") -> list[Policy]:
        """
        Sample k opponent policies, loaded and ready for play.

        Args:
            k: Number of opponents to sample.
            device: Device to load policies on.

        Returns:
            List of Policy objects.
        """
        ...


class UniformSampler(OpponentSampler):
    """Sample opponents uniformly at random."""

    def __init__(self, registry: ModelRegistry, seed: int | None = None) -> None:
        self.registry = registry
        self.rng = random.Random(seed)

    def sample(self, k: int) -> list[SnapshotInfo]:
        snapshots = self.registry.list_snapshots()
        if not snapshots:
            return []
        k = min(k, len(snapshots))
        return self.rng.sample(snapshots, k)

    def sample_policies(self, k: int, device: str = "cpu") -> list[Policy]:
        snapshots = self.sample(k)
        return [self.registry.load_policy(s.name, device=device) for s in snapshots]


class RecentBiasSampler(OpponentSampler):
    """
    Sample opponents with bias toward recent snapshots.

    Uses exponential weighting: more recent = higher probability.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        decay: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            registry: Model registry.
            decay: Decay factor per position (0 < decay <= 1).
                   decay=0.5 means each older snapshot has half the weight.
            seed: Random seed.
        """
        self.registry = registry
        self.decay = decay
        self.rng = random.Random(seed)

    def sample(self, k: int) -> list[SnapshotInfo]:
        snapshots = self.registry.list_snapshots()  # Already sorted newest-first
        if not snapshots:
            return []

        k = min(k, len(snapshots))

        # Compute weights: weight[i] = decay^i
        weights = [self.decay**i for i in range(len(snapshots))]

        # Sample without replacement using weighted selection
        selected: list[SnapshotInfo] = []
        available = list(enumerate(snapshots))

        for _ in range(k):
            if not available:
                break

            # Compute probabilities
            available_weights = [weights[idx] for idx, _ in available]
            total = sum(available_weights)
            probs = [w / total for w in available_weights]

            # Sample one
            r = self.rng.random()
            cumsum = 0.0
            chosen_idx = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    chosen_idx = i
                    break

            _, snapshot = available.pop(chosen_idx)
            selected.append(snapshot)

        return selected

    def sample_policies(self, k: int, device: str = "cpu") -> list[Policy]:
        snapshots = self.sample(k)
        return [self.registry.load_policy(s.name, device=device) for s in snapshots]


class LatestSampler(OpponentSampler):
    """Always sample the most recent k snapshots."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def sample(self, k: int) -> list[SnapshotInfo]:
        snapshots = self.registry.list_snapshots()  # Already sorted newest-first
        return snapshots[:k]

    def sample_policies(self, k: int, device: str = "cpu") -> list[Policy]:
        snapshots = self.sample(k)
        return [self.registry.load_policy(s.name, device=device) for s in snapshots]


class MixedSampler(OpponentSampler):
    """
    Sample from a mix of registry opponents and built-in policies.

    Useful for including baseline opponents like random or Stockfish.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        builtin_policies: list[Policy] | None = None,
        registry_weight: float = 0.7,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            registry: Model registry.
            builtin_policies: List of built-in policies to include.
            registry_weight: Fraction of samples from registry (0-1).
            seed: Random seed.
        """
        self.registry = registry
        self.builtin_policies = builtin_policies or []
        self.registry_weight = registry_weight
        self.rng = random.Random(seed)
        self._registry_sampler = UniformSampler(registry, seed)

    def sample(self, k: int) -> list[SnapshotInfo]:
        # Only return registry snapshots from this method
        n_registry = int(k * self.registry_weight)
        return self._registry_sampler.sample(n_registry)

    def sample_policies(self, k: int, device: str = "cpu") -> list[Policy]:
        n_registry = int(k * self.registry_weight)
        n_builtin = k - n_registry

        policies: list[Policy] = []

        # Sample from registry
        if n_registry > 0:
            snapshots = self._registry_sampler.sample(n_registry)
            for s in snapshots:
                policies.append(self.registry.load_policy(s.name, device=device))

        # Sample from builtins
        if n_builtin > 0 and self.builtin_policies:
            n_builtin = min(n_builtin, len(self.builtin_policies))
            policies.extend(self.rng.sample(self.builtin_policies, n_builtin))

        return policies


@dataclass
class SamplerFactory:
    """Factory for creating opponent samplers."""

    @staticmethod
    def create(
        sampler_type: str,
        registry: ModelRegistry,
        seed: int | None = None,
        **kwargs,
    ) -> OpponentSampler:
        """
        Create an opponent sampler.

        Args:
            sampler_type: One of "uniform", "recent", "latest", "mixed".
            registry: Model registry.
            seed: Random seed.
            **kwargs: Additional arguments for specific samplers.

        Returns:
            OpponentSampler instance.
        """
        if sampler_type == "uniform":
            return UniformSampler(registry, seed=seed)
        elif sampler_type == "recent":
            decay = kwargs.get("decay", 0.5)
            return RecentBiasSampler(registry, decay=decay, seed=seed)
        elif sampler_type == "latest":
            return LatestSampler(registry)
        elif sampler_type == "mixed":
            builtin = kwargs.get("builtin_policies", [])
            weight = kwargs.get("registry_weight", 0.7)
            return MixedSampler(registry, builtin, weight, seed=seed)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

