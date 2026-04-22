
from __future__ import annotations

from typing import Any


def function_compatibility_score(candidate: dict[str, Any]) -> float:
    seq = str(candidate.get("sequence", ""))
    if not seq:
        return 0.0
    polar = set("NQST")
    score = sum(1 for c in seq if c in polar) / float(len(seq))
    return min(1.0, max(0.0, score + 0.15))
