from __future__ import annotations

import os
from pathlib import Path

import pytest


REQUIRED_USECASE_DIRS = [
    "usecases/enzyme_design_arena",
    "usecases/enzyme_design_arena/configs",
    "usecases/enzyme_design_arena/data",
    "usecases/enzyme_design_arena/data/raw",
    "usecases/enzyme_design_arena/data/processed",
    "usecases/enzyme_design_arena/data/manifests",
    "usecases/enzyme_design_arena/adapters",
    "usecases/enzyme_design_arena/arena",
    "usecases/enzyme_design_arena/prompts",
    "usecases/enzyme_design_arena/scripts",
]

REQUIRED_USECASE_FILES = [
    "usecases/enzyme_design_arena/README.md",
    "usecases/enzyme_design_arena/run_arena.py",
    "usecases/enzyme_design_arena/configs/targets.yaml",
    "usecases/enzyme_design_arena/configs/scorers.yaml",
    "usecases/enzyme_design_arena/configs/policies.yaml",
    "usecases/enzyme_design_arena/adapters/candidate_generators.py",
    "usecases/enzyme_design_arena/adapters/sequence_design.py",
    "usecases/enzyme_design_arena/adapters/structure_filters.py",
    "usecases/enzyme_design_arena/adapters/activity_scorers.py",
    "usecases/enzyme_design_arena/adapters/function_scorers.py",
    "usecases/enzyme_design_arena/arena/schemas.py",
    "usecases/enzyme_design_arena/arena/registry.py",
    "usecases/enzyme_design_arena/arena/planner.py",
    "usecases/enzyme_design_arena/arena/aggregate.py",
    "usecases/enzyme_design_arena/arena/metrics.py",
    "usecases/enzyme_design_arena/arena/reports.py",
    "usecases/enzyme_design_arena/prompts/planner.txt",
    "usecases/enzyme_design_arena/prompts/reviewer.txt",
    "usecases/enzyme_design_arena/prompts/failure_analysis.txt",
    "usecases/enzyme_design_arena/scripts/download_data.sh",
    "usecases/enzyme_design_arena/scripts/build_manifest.py",
    "usecases/enzyme_design_arena/scripts/normalize_datasets.py",
    "usecases/enzyme_design_arena/data/manifests/datasets.yaml",
]

CORE_DATASET_EXPECTATIONS = {
    "nucb": [
        "usecases/enzyme_design_arena/data/raw/nucb/landscape.csv",
        "usecases/enzyme_design_arena/data/raw/nucb/README_source.txt",
    ],
    "modify_rma": [
        "usecases/enzyme_design_arena/data/raw/modify_rma/supplement.pdf",
        "usecases/enzyme_design_arena/data/raw/modify_rma/supplementary_data_1.xlsx",
        "usecases/enzyme_design_arena/data/raw/modify_rma/README_source.txt",
    ],
    "mcba": [
        "usecases/enzyme_design_arena/data/raw/mcba/source_data.xlsx",
        "usecases/enzyme_design_arena/data/raw/mcba/supplement.pdf",
        "usecases/enzyme_design_arena/data/raw/mcba/README_source.txt",
    ],
}

OPTIONAL_DATASET_EXPECTATIONS = {
    "proteingym_optional": [
        "usecases/enzyme_design_arena/data/raw/proteingym_optional/DMS_substitutions.csv",
        "usecases/enzyme_design_arena/data/raw/proteingym_optional/README_source.txt",
    ]
}


def _discover_repo_root() -> Path:
    override = os.environ.get("USECASE_REPO_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"USECASE_REPO_ROOT does not exist: {root}")
        return root
    return Path.cwd().resolve()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _discover_repo_root()


@pytest.fixture(scope="session")
def usecase_root(repo_root: Path) -> Path:
    return repo_root / "usecases" / "enzyme_design_arena"


@pytest.fixture(scope="session")
def core_dataset_expectations() -> dict[str, list[str]]:
    return CORE_DATASET_EXPECTATIONS


@pytest.fixture(scope="session")
def optional_dataset_expectations() -> dict[str, list[str]]:
    return OPTIONAL_DATASET_EXPECTATIONS
