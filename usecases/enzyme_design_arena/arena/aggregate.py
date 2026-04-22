
from __future__ import annotations

from typing import Any


def aggregate_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        f_score = float(row.get("function_score", 0.0))
        a_score = float(row.get("activity_score", 0.0))
        agg = 0.55 * f_score + 0.45 * a_score
        merged = dict(row)
        merged["aggregate_score"] = round(agg, 6)
        out.append(merged)
    return out


def select_top_k(rows: list[dict[str, Any]], k: int = 10) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: float(r.get("aggregate_score", 0.0)), reverse=True)
    return sorted_rows[: max(0, k)]
