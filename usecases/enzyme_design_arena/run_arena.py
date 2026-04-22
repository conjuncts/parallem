
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import parallem as pllm

from usecases.enzyme_design_arena.adapters.activity_scorers import activity_surrogate_score
from usecases.enzyme_design_arena.adapters.candidate_generators import generate_candidates
from usecases.enzyme_design_arena.adapters.function_scorers import function_compatibility_score
from usecases.enzyme_design_arena.adapters.structure_filters import cheap_structure_filter
from usecases.enzyme_design_arena.arena.aggregate import aggregate_scores, select_top_k
from usecases.enzyme_design_arena.arena.reports import write_markdown_report


def _load_nucb_seed_sequences(repo_root: Path, limit: int) -> list[str]:
    csv_path = repo_root / "usecases/enzyme_design_arena/data/raw/nucb/landscape.csv"
    if not csv_path.exists():
        return ["MNNNQKK", "MNNNQKR", "MNNNQKA"]
    out: list[str] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("variant_aa_seq") or row.get("sequence") or row.get("seq")
            if seq:
                out.append(seq)
            if len(out) >= limit:
                break
    return out or ["MNNNQKK", "MNNNQKR", "MNNNQKA"]


def run_pipeline(strategy: str, target: str, max_candidates: int, llm: str) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / "runs" / "enzyme_design_arena"
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = _load_nucb_seed_sequences(repo_root, limit=max(8, min(max_candidates, 64)))
    candidates = generate_candidates(target=target, seed_sequences=seeds, max_candidates=max_candidates)
    candidates = [c for c in candidates if cheap_structure_filter(c)]

    with pllm.resume_directory(
        str(run_dir),
        provider="openai",
        strategy=strategy,
        llm=llm,
        dashboard=False,
        hash_by=["llm"],
    ) as orch:
        _ = orch
        scored_rows = []
        for cand in candidates:
            row = {
                "candidate_id": cand["candidate_id"],
                "sequence": cand["sequence"],
                "function_score": function_compatibility_score(cand),
                "activity_score": activity_surrogate_score(cand),
            }
            scored_rows.append(row)

    aggregated = aggregate_scores(scored_rows)
    top = select_top_k(aggregated, k=min(10, len(aggregated)))

    out_dir = repo_root / "usecases/enzyme_design_arena/data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "top_k_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "sequence", "aggregate_score", "function_score", "activity_score"],
        )
        writer.writeheader()
        for row in top:
            writer.writerow(row)

    summary = {
        "target": target,
        "strategy": strategy,
        "num_candidates": len(candidates),
        "num_scored": len(scored_rows),
        "num_top": len(top),
    }
    (out_dir / "arena_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(out_dir / "arena_report.md", summary=summary, top_rows=top)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enzyme design arena demo pipeline")
    parser.add_argument("--strategy", choices=["sync", "concurrent", "batch"], default="sync")
    parser.add_argument("--target", default="nucb")
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--llm", default="gpt-5-mini")
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(
                strategy=args.strategy,
                target=args.target,
                max_candidates=args.max_candidates,
                llm=args.llm,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
