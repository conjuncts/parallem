
from __future__ import annotations


def propose_sequence_variant(sequence: str, position: int, aa: str) -> str:
    if not sequence:
        return sequence
    position = max(0, min(position, len(sequence) - 1))
    return sequence[:position] + aa + sequence[position + 1 :]
