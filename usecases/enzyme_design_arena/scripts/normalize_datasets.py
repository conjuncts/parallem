from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "sha256": "",
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def normalize_nucb(raw_csv: Path, out_csv: Path) -> dict[str, Any]:
    """Normalize NucB landscape into a simple candidate table."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not raw_csv.exists():
        out_csv.write_text("candidate_id,sequence\n", encoding="utf-8")
        return {"dataset": "nucb", "rows": 0, "missing_raw": True}

    with raw_csv.open("r", encoding="utf-8", errors="ignore") as src, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = [
            "candidate_id",
            "sequence",
            "fitness",
            "source_row",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        idx = 0
        for row in reader:
            seq = row.get("variant_aa_seq") or row.get("sequence") or row.get("seq")
            if not seq:
                continue
            fitness = (
                row.get("fitness")
                or row.get("activity")
                or row.get("score")
                or row.get("y")
                or ""
            )
            writer.writerow(
                {
                    "candidate_id": f"nucb-{idx:05d}",
                    "sequence": seq,
                    "fitness": fitness,
                    "source_row": idx,
                }
            )
            idx += 1
    return {"dataset": "nucb", "rows": idx, "missing_raw": False}


def normalize_proteingym(raw_csv: Path, out_csv: Path) -> dict[str, Any]:
    """
    Normalize optional ProteinGym substitutions metadata into a compact table.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not raw_csv.exists():
        out_csv.write_text("entry_id,protein\n", encoding="utf-8")
        return {"dataset": "proteingym_optional", "rows": 0, "missing_raw": True}

    with raw_csv.open("r", encoding="utf-8", errors="ignore") as src, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = [
            "entry_id",
            "protein",
            "dms_id",
            "num_mutations",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        idx = 0
        for row in reader:
            protein = (
                row.get("UniProt_ID")
                or row.get("protein")
                or row.get("gene")
                or row.get("DMS_id")
                or "unknown"
            )
            writer.writerow(
                {
                    "entry_id": f"pg-{idx:05d}",
                    "protein": protein,
                    "dms_id": row.get("DMS_id", ""),
                    "num_mutations": row.get("num_mutations", ""),
                }
            )
            idx += 1
    return {"dataset": "proteingym_optional", "rows": idx, "missing_raw": False}


def summarize_assets(
    out_csv: Path,
    dataset_name: str,
    file_paths: list[Path],
) -> dict[str, Any]:
    """
    Record non-CSV raw assets (pdf/xlsx/repo stats) into a processed inventory table.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for p in file_paths:
        rows.append(_file_summary(p))

    with out_csv.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["path", "exists", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "dataset": dataset_name,
        "rows": len(rows),
        "missing_raw": not all(row["exists"] for row in rows),
    }


def run_normalization(repo_root: Path, include_optional: bool = True) -> dict[str, Any]:
    usecase_root = repo_root / "usecases" / "enzyme_design_arena"
    raw_root = usecase_root / "data" / "raw"
    processed_root = usecase_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "usecase": "enzyme_design_arena",
        "processed_root": str(processed_root),
        "datasets": [],
    }

    report["datasets"].append(
        normalize_nucb(
            raw_root / "nucb" / "landscape.csv",
            processed_root / "nucb_candidates.csv",
        )
    )

    report["datasets"].append(
        summarize_assets(
            processed_root / "modify_rma_assets.csv",
            dataset_name="modify_rma",
            file_paths=[
                raw_root / "modify_rma" / "supplementary_data_1.xlsx",
                raw_root / "modify_rma" / "supplement.pdf",
            ],
        )
    )

    mcba_repo = raw_root / "mcba" / "accelerated_enzyme_engineering_repo"
    repo_stats_path = processed_root / "mcba_repo_stats.json"
    repo_stats = {
        "path": str(mcba_repo),
        "exists": mcba_repo.exists(),
        "files": 0,
    }
    if mcba_repo.exists():
        repo_stats["files"] = sum(1 for _ in mcba_repo.rglob("*") if _.is_file())
    repo_stats_path.write_text(json.dumps(repo_stats, indent=2), encoding="utf-8")

    report["datasets"].append(
        summarize_assets(
            processed_root / "mcba_assets.csv",
            dataset_name="mcba",
            file_paths=[
                raw_root / "mcba" / "source_data.xlsx",
                raw_root / "mcba" / "supplement.pdf",
                repo_stats_path,
            ],
        )
    )

    if include_optional:
        report["datasets"].append(
            normalize_proteingym(
                raw_root / "proteingym_optional" / "DMS_substitutions.csv",
                processed_root / "proteingym_substitutions.csv",
            )
        )

    report_path = processed_root / "normalization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw datasets into processed tables for enzyme_design_arena."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to auto-detected root from this file.",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional ProteinGym normalization.",
    )
    args = parser.parse_args()

    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[3]
    )
    report = run_normalization(repo_root, include_optional=not args.skip_optional)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
