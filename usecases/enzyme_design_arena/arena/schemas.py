
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    candidate_id: str
    sequence: str
    origin: str


@dataclass
class ScoredCandidate:
    candidate_id: str
    sequence: str
    function_score: float
    activity_score: float
    aggregate_score: float
