"""Result formatting and export for evaluation reports.

Provides functions to format evaluation results for console output,
JSON export, and other formats.

Example:
    >>> from chesster.eval.evaluate import evaluate_model
    >>> from chesster.eval.report import format_report
    >>> result = evaluate_model("stockfish:1", suite="quick")
    >>> print(format_report(result))
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesster.eval.evaluate import EvaluationResult


def format_report(result: EvaluationResult, width: int = 80) -> str:
    """
    Format evaluation result as a human-readable report.

    Args:
        result: EvaluationResult to format.
        width: Line width for formatting.

    Returns:
        Formatted string report.
    """
    lines: list[str] = []
    sep = "=" * width
    thin_sep = "-" * width

    # Header
    lines.append(sep)
    lines.append("Model Evaluation Report".center(width))
    lines.append(sep)
    lines.append(f"Model: {result.model_id}")
    if result.suite_name:
        lines.append(f"Suite: {result.suite_name} ({result.total_games} games total)")
    else:
        lines.append(f"Games: {result.total_games}")
    lines.append(f"Date: {result.timestamp[:19].replace('T', ' ')}")
    lines.append("")

    # Results by opponent
    lines.append("RESULTS BY OPPONENT")
    lines.append(thin_sep)

    # Table header
    header = f"{'Opponent':<20} {'Games':>6} {'W':>4} {'D':>4} {'L':>4} {'Score':>8} {'Win Rate':>10}"
    lines.append(header)
    lines.append(thin_sep)

    # Sort opponents by reference ELO (weakest first)
    sorted_opponents = sorted(
        result.opponents.values(),
        key=lambda x: x.reference_elo,
    )

    for opp in sorted_opponents:
        row = (
            f"{opp.opponent_id:<20} "
            f"{opp.games:>6} "
            f"{opp.wins:>4} "
            f"{opp.draws:>4} "
            f"{opp.losses:>4} "
            f"{opp.score_rate:>7.1%} "
            f"{opp.win_rate:>9.1%}"
        )
        lines.append(row)

    # Total row
    lines.append(thin_sep)
    total_row = (
        f"{'TOTAL':<20} "
        f"{result.total_games:>6} "
        f"{result.total_wins:>4} "
        f"{result.total_draws:>4} "
        f"{result.total_losses:>4} "
        f"{result.score_rate:>7.1%} "
        f"{result.win_rate:>9.1%}"
    )
    lines.append(total_row)
    lines.append("")

    # Summary metrics
    lines.append("SUMMARY METRICS")
    lines.append(thin_sep)
    lines.append(f"Estimated ELO:        {result.estimated_elo:.0f}")

    # Key opponent scores
    if result.vs_random_score is not None:
        status = "PASS" if result.vs_random_score >= 0.7 else "FAIL"
        lines.append(f"vs Random:            {result.vs_random_score:.1%} (target: >70%) [{status}]")

    if result.vs_stockfish_1_score is not None:
        lines.append(f"vs Stockfish d=1:     {result.vs_stockfish_1_score:.1%}")

    if result.vs_stockfish_3_score is not None:
        lines.append(f"vs Stockfish d=3:     {result.vs_stockfish_3_score:.1%}")

    if result.vs_stockfish_5_score is not None:
        lines.append(f"vs Stockfish d=5:     {result.vs_stockfish_5_score:.1%}")

    lines.append(sep)

    return "\n".join(lines)


def format_compact(result: EvaluationResult) -> str:
    """
    Format evaluation result as a single-line summary.

    Useful for logging during training.

    Args:
        result: EvaluationResult to format.

    Returns:
        Single-line summary string.
    """
    parts = [
        f"ELO={result.estimated_elo:.0f}",
        f"W/D/L={result.total_wins}/{result.total_draws}/{result.total_losses}",
        f"score={result.score_rate:.1%}",
    ]

    if result.vs_random_score is not None:
        parts.append(f"vs_random={result.vs_random_score:.1%}")

    return " | ".join(parts)


def format_markdown(result: EvaluationResult) -> str:
    """
    Format evaluation result as Markdown.

    Args:
        result: EvaluationResult to format.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append(f"## Evaluation: {result.model_id}")
    lines.append("")
    if result.suite_name:
        lines.append(f"**Suite:** {result.suite_name}")
    lines.append(f"**Date:** {result.timestamp[:19].replace('T', ' ')}")
    lines.append(f"**Estimated ELO:** {result.estimated_elo:.0f}")
    lines.append("")

    # Results table
    lines.append("### Results by Opponent")
    lines.append("")
    lines.append("| Opponent | Games | W | D | L | Score | Win Rate |")
    lines.append("|----------|-------|---|---|---|-------|----------|")

    sorted_opponents = sorted(
        result.opponents.values(),
        key=lambda x: x.reference_elo,
    )

    for opp in sorted_opponents:
        lines.append(
            f"| {opp.opponent_id} | {opp.games} | {opp.wins} | {opp.draws} | "
            f"{opp.losses} | {opp.score_rate:.1%} | {opp.win_rate:.1%} |"
        )

    lines.append(
        f"| **TOTAL** | {result.total_games} | {result.total_wins} | "
        f"{result.total_draws} | {result.total_losses} | "
        f"{result.score_rate:.1%} | {result.win_rate:.1%} |"
    )
    lines.append("")

    # Key metrics
    lines.append("### Key Metrics")
    lines.append("")

    if result.vs_random_score is not None:
        status = "PASS" if result.vs_random_score >= 0.7 else "FAIL"
        lines.append(f"- vs Random: {result.vs_random_score:.1%} (target: >70%) **{status}**")

    if result.vs_stockfish_1_score is not None:
        lines.append(f"- vs Stockfish d=1: {result.vs_stockfish_1_score:.1%}")

    if result.vs_stockfish_5_score is not None:
        lines.append(f"- vs Stockfish d=5: {result.vs_stockfish_5_score:.1%}")

    return "\n".join(lines)
