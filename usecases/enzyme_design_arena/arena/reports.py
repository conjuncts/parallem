
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_markdown_report(path: Path, summary: dict[str, Any], top_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Enzyme Design Arena Report",
        "",
        "## Summary",
        f"- target: {summary.get('target')}",
        f"- strategy: {summary.get('strategy')}",
        f"- num_candidates: {summary.get('num_candidates')}",
        f"- num_scored: {summary.get('num_scored')}",
        "",
        "## Top Candidates",
        "| candidate_id | aggregate_score | function_score | activity_score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in top_rows:
        lines.append(
            f"| {row.get('candidate_id')} | {row.get('aggregate_score')} | {row.get('function_score')} | {row.get('activity_score')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
