
from __future__ import annotations

from typing import Any


def cheap_structure_filter(candidate: dict[str, Any]) -> bool:
    seq = str(candidate.get("sequence", ""))
    if len(seq) < 6:
        return False
    return set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY"))
