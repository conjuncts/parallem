from __future__ import annotations

import ast
from pathlib import Path


PY_MODULES_TO_PARSE = [
    "usecases/enzyme_design_arena/run_arena.py",
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
    "usecases/enzyme_design_arena/scripts/build_manifest.py",
    "usecases/enzyme_design_arena/scripts/normalize_datasets.py",
]


def _parse_python(path: Path) -> ast.AST:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def test_python_files_have_valid_syntax(repo_root: Path) -> None:
    broken = []
    for rel in PY_MODULES_TO_PARSE:
        path = repo_root / rel
        try:
            _parse_python(path)
        except SyntaxError as exc:
            broken.append(f"{rel}: {exc}")
    assert not broken, "Python syntax errors detected:\n" + "\n".join(broken)


def test_run_arena_mentions_strategy_switch(repo_root: Path) -> None:
    path = repo_root / "usecases/enzyme_design_arena/run_arena.py"
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "--strategy" in lowered or "strategy" in lowered, "run_arena.py should expose a strategy switch"
    assert "sync" in lowered, "run_arena.py should mention sync mode"
    assert ("batch" in lowered) or ("concurrent" in lowered), (
        "run_arena.py should mention batch or concurrent mode"
    )


def test_run_arena_uses_resume_directory_or_equivalent(repo_root: Path) -> None:
    path = repo_root / "usecases/enzyme_design_arena/run_arena.py"
    text = path.read_text(encoding="utf-8")
    assert "resume_directory" in text, (
        "run_arena.py should use Parallem's resume_directory entrypoint to demonstrate the usecase"
    )


def test_run_arena_has_cli_shape(repo_root: Path) -> None:
    path = repo_root / "usecases/enzyme_design_arena/run_arena.py"
    text = path.read_text(encoding="utf-8").lower()
    expected_tokens = ["target", "max-candidates"]
    missing = [token for token in expected_tokens if token not in text]
    assert not missing, f"run_arena.py CLI looks incomplete; missing tokens: {missing}"


def test_scripts_reference_manifest_or_normalization(repo_root: Path) -> None:
    build_manifest = (repo_root / "usecases/enzyme_design_arena/scripts/build_manifest.py").read_text(encoding="utf-8").lower()
    normalize = (repo_root / "usecases/enzyme_design_arena/scripts/normalize_datasets.py").read_text(encoding="utf-8").lower()

    assert "manifest" in build_manifest, "build_manifest.py should talk about the dataset manifest"
    assert any(token in normalize for token in ["normalize", "processed", "raw"]), (
        "normalize_datasets.py should reference dataset normalization behavior"
    )
