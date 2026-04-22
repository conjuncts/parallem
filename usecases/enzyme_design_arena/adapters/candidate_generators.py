
from __future__ import annotations

from typing import Any


def generate_candidates(target: str, seed_sequences: list[str], max_candidates: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not seed_sequences:
        seed_sequences = ["MNNNQKK"]
    i = 0
    while len(out) < max_candidates:
        base = seed_sequences[i % len(seed_sequences)]
        mutated = base[:-1] + chr(65 + (i % 20))
        out.append(
            {"candidate_id": f"{target}-{i:05d}", "sequence": mutated, "origin": "seed_mutation"}
        )
        i += 1
    return out
