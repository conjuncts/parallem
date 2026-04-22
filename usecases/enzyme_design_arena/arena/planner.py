
from __future__ import annotations


def planner_prompt_shape(target: str, max_candidates: int) -> str:
    return (
        f"Plan arena for target={target}; "
        f"generate up to {max_candidates} candidates, "
        "fan out to scorer committee, aggregate consensus, and report top hits."
    )
