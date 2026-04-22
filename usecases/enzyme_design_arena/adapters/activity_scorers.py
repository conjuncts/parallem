
from __future__ import annotations

from typing import Any


def activity_surrogate_score(candidate: dict[str, Any]) -> float:
    seq = str(candidate.get("sequence", ""))
    if not seq:
        return 0.0
    hydrophobic = set("AILMFWVY")
    return sum(1 for c in seq if c in hydrophobic) / float(len(seq))
