from __future__ import annotations

from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_config_files_exist_and_are_nonempty(repo_root: Path) -> None:
    config_paths = [
        repo_root / "usecases/enzyme_design_arena/configs/targets.yaml",
        repo_root / "usecases/enzyme_design_arena/configs/scorers.yaml",
        repo_root / "usecases/enzyme_design_arena/configs/policies.yaml",
    ]
    missing_or_empty = []
    for path in config_paths:
        if not path.exists() or path.stat().st_size == 0:
            missing_or_empty.append(str(path.relative_to(repo_root)))
    assert not missing_or_empty, f"Missing or empty config files: {missing_or_empty}"


def test_dataset_manifest_exists_and_mentions_core_datasets(repo_root: Path) -> None:
    manifest = repo_root / "usecases/enzyme_design_arena/data/manifests/datasets.yaml"
    assert manifest.exists(), f"Missing dataset manifest: {manifest}"
    text = _read_text(manifest)
    for token in ["nucb:", "modify_rma:", "mcba:"]:
        assert token in text, f"Manifest missing dataset key: {token}"
    assert "raw_path:" in text, "Manifest should declare raw_path entries"
    assert "task_type:" in text, "Manifest should declare task_type entries"


def test_manifest_points_into_usecase_data_tree(repo_root: Path) -> None:
    manifest = repo_root / "usecases/enzyme_design_arena/data/manifests/datasets.yaml"
    text = _read_text(manifest)
    expected_paths = [
        "usecases/enzyme_design_arena/data/raw/nucb/landscape.csv",
        "usecases/enzyme_design_arena/data/raw/modify_rma/supplementary_data_1.xlsx",
        "usecases/enzyme_design_arena/data/raw/mcba/source_data.xlsx",
    ]
    missing = [p for p in expected_paths if p not in text]
    assert not missing, f"Manifest missing expected raw paths: {missing}"


def test_targets_config_mentions_primary_tasks(repo_root: Path) -> None:
    targets = repo_root / "usecases/enzyme_design_arena/configs/targets.yaml"
    text = _read_text(targets)
    expected_tokens = ["nucb", "modify_rma", "mcba"]
    missing = [t for t in expected_tokens if t not in text]
    assert not missing, f"targets.yaml should mention all primary tasks: {missing}"


def test_scorers_config_and_policies_have_expected_sections(repo_root: Path) -> None:
    scorers_text = _read_text(repo_root / "usecases/enzyme_design_arena/configs/scorers.yaml")
    policies_text = _read_text(repo_root / "usecases/enzyme_design_arena/configs/policies.yaml")

    scorer_tokens = ["scor", "function", "activity"]
    assert any(token in scorers_text.lower() for token in scorer_tokens), (
        "scorers.yaml should contain scorer-related definitions"
    )

    policy_tokens = ["top_k", "consensus", "aggregate", "diversity", "uncertainty"]
    assert any(token in policies_text.lower() for token in policy_tokens), (
        "policies.yaml should contain aggregation or ranking policy definitions"
    )
