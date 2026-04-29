from __future__ import annotations

import os
from pathlib import Path

import pytest

from usecases.enzyme_design_arena import run_arena


def test_direct_openai_path_is_forbidden() -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        run_arena.run_direct_openai_judge_forbidden()


def test_run_arena_judge_uses_parallem(repo_root: Path) -> None:
    text = (repo_root / "usecases/enzyme_design_arena/run_arena.py").read_text(
        encoding="utf-8"
    )
    assert "resume_directory" in text
    assert "conv.ask_llm(" in text
    assert "llm-judge" in text


def test_run_arena_does_not_import_openai_directly(repo_root: Path) -> None:
    text = (repo_root / "usecases/enzyme_design_arena/run_arena.py").read_text(
        encoding="utf-8"
    ).lower()
    assert "from openai import" not in text
    assert "openai.openai(" not in text


def test_run_arena_exposes_parallem_knobs(repo_root: Path) -> None:
    text = (repo_root / "usecases/enzyme_design_arena/run_arena.py").read_text(
        encoding="utf-8"
    ).lower()
    for token in ["--provider", "--strategy", "--llm", "--dashboard", "--hash-by"]:
        assert token in text, f"missing cli option: {token}"
    assert "async" in text, "strategy switch should expose async alias for concurrent mode"
    for token in ["--total-job-budget", "--disable-llm-extra"]:
        assert token in text, f"missing cli option: {token}"


def test_config_api_key_can_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    run_arena._setup_api_key(
        {
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "api_key": "config-key",
            "prefer_config_key": True,
        }
    )
    assert os.environ["OPENAI_API_KEY"] == "config-key"
