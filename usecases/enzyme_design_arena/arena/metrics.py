
from __future__ import annotations


def top_k_hit_rate(num_hits: int, k: int) -> float:
    if k <= 0:
        return 0.0
    return max(0.0, min(1.0, num_hits / float(k)))


def enrichment_over_random(hit_rate: float, baseline_rate: float) -> float:
    if baseline_rate <= 0:
        return 0.0
    return hit_rate / baseline_rate
