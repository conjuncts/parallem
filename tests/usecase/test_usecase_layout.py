from __future__ import annotations

from pathlib import Path

from conftest import REQUIRED_USECASE_DIRS, REQUIRED_USECASE_FILES


def test_usecase_root_exists(usecase_root: Path) -> None:
    assert usecase_root.exists(), f"Missing usecase root: {usecase_root}"
    assert usecase_root.is_dir(), f"Usecase root is not a directory: {usecase_root}"


def test_required_directories_exist(repo_root: Path) -> None:
    missing = []
    for rel in REQUIRED_USECASE_DIRS:
        path = repo_root / rel
        if not path.exists() or not path.is_dir():
            missing.append(rel)
    assert not missing, f"Missing required directories: {missing}"


def test_required_files_exist(repo_root: Path) -> None:
    missing = []
    for rel in REQUIRED_USECASE_FILES:
        path = repo_root / rel
        if not path.exists() or not path.is_file():
            missing.append(rel)
    assert not missing, f"Missing required files: {missing}"


def test_python_sources_are_nonempty(repo_root: Path) -> None:
    py_files = [repo_root / rel for rel in REQUIRED_USECASE_FILES if rel.endswith('.py')]
    tiny = []
    for path in py_files:
        if path.stat().st_size < 20:
            tiny.append(str(path.relative_to(repo_root)))
    assert not tiny, f"Unexpectedly tiny Python source files: {tiny}"


def test_prompt_files_are_nonempty(repo_root: Path) -> None:
    prompt_files = [
        repo_root / "usecases/enzyme_design_arena/prompts/planner.txt",
        repo_root / "usecases/enzyme_design_arena/prompts/reviewer.txt",
        repo_root / "usecases/enzyme_design_arena/prompts/failure_analysis.txt",
    ]
    empty = []
    for path in prompt_files:
        if path.stat().st_size == 0:
            empty.append(str(path.relative_to(repo_root)))
    assert not empty, f"Prompt files should not be empty: {empty}"
