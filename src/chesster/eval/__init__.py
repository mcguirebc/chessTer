"""Model evaluation framework for chesster.

Usage:
    from chesster.eval import evaluate_model, load_policy

    # Evaluate a model
    result = evaluate_model("hf:Qwen/Qwen2.5-1.5B-Instruct", suite="quick")
    print(f"ELO: {result.estimated_elo}")

    # Load any policy type
    policy = load_policy("stockfish:5")
"""

from chesster.eval.baselines import (
    FULL_SUITE,
    QUICK_SUITE,
    STANDARD_SUITE,
    SUITES,
    BaselineSuite,
    get_baseline_suite,
    get_reference_elo,
    list_suites,
)
from chesster.eval.evaluate import (
    EvaluationResult,
    OpponentResult,
    evaluate_model,
)
from chesster.eval.loader import (
    PolicyLoadError,
    get_supported_types,
    load_policy,
    parse_opponent_specs,
)
from chesster.eval.report import (
    format_compact,
    format_markdown,
    format_report,
)

__all__ = [
    # Main functions
    "evaluate_model",
    "load_policy",
    # Result types
    "EvaluationResult",
    "OpponentResult",
    # Baselines
    "BaselineSuite",
    "get_baseline_suite",
    "list_suites",
    "get_reference_elo",
    "SUITES",
    "QUICK_SUITE",
    "STANDARD_SUITE",
    "FULL_SUITE",
    # Loader utilities
    "PolicyLoadError",
    "get_supported_types",
    "parse_opponent_specs",
    # Report formatting
    "format_report",
    "format_compact",
    "format_markdown",
]
