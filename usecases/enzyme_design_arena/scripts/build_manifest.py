from __future__ import annotations

import json
from pathlib import Path


def build_manifest_text(repo_root: Path) -> str:
    def exists(rel_path: str) -> bool:
        return (repo_root / rel_path).exists()

    nucb_enabled = exists("usecases/enzyme_design_arena/data/raw/nucb/landscape.csv")
    modify_enabled = exists(
        "usecases/enzyme_design_arena/data/raw/modify_rma/supplementary_data_1.xlsx"
    )
    mcba_enabled = exists("usecases/enzyme_design_arena/data/raw/mcba/source_data.xlsx")
    pg_enabled = exists(
        "usecases/enzyme_design_arena/data/raw/proteingym_optional/DMS_substitutions.csv"
    )

    return f"""nucb:
  enabled: {"true" if nucb_enabled else "false"}
  raw_path: usecases/enzyme_design_arena/data/raw/nucb/landscape.csv
  task_type: sequence_activity_landscape

modify_rma:
  enabled: {"true" if modify_enabled else "false"}
  raw_path: usecases/enzyme_design_arena/data/raw/modify_rma/supplementary_data_1.xlsx
  task_type: library_screening
  subtasks: [c_b, c_si]

mcba:
  enabled: {"true" if mcba_enabled else "false"}
  raw_path: usecases/enzyme_design_arena/data/raw/mcba/source_data.xlsx
  task_type: reaction_conditioned_engineering

proteingym_optional:
  enabled: {"true" if pg_enabled else "false"}
  raw_path: usecases/enzyme_design_arena/data/raw/proteingym_optional/DMS_substitutions.csv
  task_type: mutation_effect_extension
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    manifest = repo_root / "usecases/enzyme_design_arena/data/manifests/datasets.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(build_manifest_text(repo_root), encoding="utf-8")
    print(f"Wrote manifest: {manifest}")

    summary = {
        "manifest": str(manifest),
        "datasets": {
            "nucb": (repo_root / "usecases/enzyme_design_arena/data/raw/nucb/landscape.csv").exists(),
            "modify_rma": (repo_root / "usecases/enzyme_design_arena/data/raw/modify_rma/supplementary_data_1.xlsx").exists(),
            "mcba": (repo_root / "usecases/enzyme_design_arena/data/raw/mcba/source_data.xlsx").exists(),
            "proteingym_optional": (repo_root / "usecases/enzyme_design_arena/data/raw/proteingym_optional/DMS_substitutions.csv").exists(),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
