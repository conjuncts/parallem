
from __future__ import annotations

from typing import Callable

from usecases.enzyme_design_arena.adapters.activity_scorers import activity_surrogate_score
from usecases.enzyme_design_arena.adapters.function_scorers import function_compatibility_score


SCORER_REGISTRY: dict[str, Callable[[dict], float]] = {
    "function_compatibility": function_compatibility_score,
    "activity_surrogate": activity_surrogate_score,
}
