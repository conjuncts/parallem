
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import parallem as pllm
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {c: i for i, c in enumerate(AA)}


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _seq_features(seq: str) -> list[float]:
    """
    Very small sequence-only featureizer:
    - 20 amino-acid composition features
    - sequence length feature (scaled)
    """
    if not seq:
        return [0.0] * 21
    seq = seq.strip().upper()
    counts = [0.0] * 20
    valid = 0
    for c in seq:
        idx = AA_INDEX.get(c)
        if idx is not None:
            counts[idx] += 1.0
            valid += 1
    if valid > 0:
        counts = [v / valid for v in counts]
    length_feature = min(len(seq), 2048) / 256.0
    return counts + [length_feature]


def _sequence_id_from_text(text: str, fallback_idx: int) -> str:
    if text:
        return f"seq-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
    return f"row-{fallback_idx:08d}"


def _split_indices_by_sequence_id(sequence_ids: list[str]) -> tuple[list[int], list[int], list[int]]:
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for i, sid in enumerate(sequence_ids):
        bucket = int(hashlib.sha1(sid.encode("utf-8")).hexdigest(), 16) % 10
        if bucket < 7:
            train.append(i)
        elif bucket == 7:
            val.append(i)
        else:
            test.append(i)
    if not train or not val or not test:
        raise ValueError(
            f"Bad split sizes from sequence_id split: train={len(train)} val={len(val)} test={len(test)}"
        )
    return train, val, test


def _load_nucb_rows(raw_csv: Path, max_candidates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with raw_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            seq = row.get("sequence", "")
            if not seq:
                continue
            y_cls = row.get("is_functional", "").strip().lower()
            y_num_mut = _safe_float(row.get("num_mutations"))
            if y_num_mut is None:
                continue
            rows.append(
                {
                    "sequence_id": _sequence_id_from_text(seq, idx),
                    "sequence": seq,
                    "x": _seq_features(seq),
                    "y_cls": 1 if y_cls in {"true", "1", "yes"} else 0,
                    "y_reg": y_num_mut,
                }
            )
            if len(rows) >= max_candidates:
                break
    return rows


def _load_proteingym_rows(raw_csv: Path, max_candidates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with raw_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            seq = row.get("target_seq", "")
            y = _safe_float(row.get("DMS_total_number_mutants"))
            if not seq or y is None:
                continue
            rows.append(
                {
                    "sequence_id": _sequence_id_from_text(seq, idx),
                    "sequence": seq,
                    "x": _seq_features(seq),
                    "y_reg": y,
                }
            )
            if len(rows) >= max_candidates:
                break
    return rows


def _build_benchmark_tasks(repo_root: Path, max_candidates: int) -> list[dict[str, Any]]:
    raw_root = repo_root / "usecases/enzyme_design_arena/data/raw"
    nucb_rows = _load_nucb_rows(raw_root / "nucb/landscape.csv", max_candidates=max_candidates)
    pg_rows = _load_proteingym_rows(
        raw_root / "proteingym_optional/DMS_substitutions.csv",
        max_candidates=max_candidates,
    )
    if len(nucb_rows) < 30 or len(pg_rows) < 30:
        raise ValueError(
            f"Insufficient rows for benchmark tasks: nucb={len(nucb_rows)} proteingym={len(pg_rows)}"
        )

    return [
        {
            "dataset": "nucb_functional_cls",
            "task_type": "classification",
            "X": [r["x"] for r in nucb_rows],
            "y": [r["y_cls"] for r in nucb_rows],
            "sequence_ids": [r["sequence_id"] for r in nucb_rows],
        },
        {
            "dataset": "nucb_num_mut_reg",
            "task_type": "regression",
            "X": [r["x"] for r in nucb_rows],
            "y": [r["y_reg"] for r in nucb_rows],
            "sequence_ids": [r["sequence_id"] for r in nucb_rows],
        },
        {
            "dataset": "proteingym_total_mut_reg",
            "task_type": "regression",
            "X": [r["x"] for r in pg_rows],
            "y": [r["y_reg"] for r in pg_rows],
            "sequence_ids": [r["sequence_id"] for r in pg_rows],
        },
    ]


def _classification_model_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "logistic": {"model__C": [0.2, 1.0]},
        "svc_rbf": {"model__C": [0.5, 1.5], "model__gamma": ["scale"]},
        "rf_cls": {"model__n_estimators": [120], "model__max_depth": [None, 12]},
        "gb_cls": {"model__n_estimators": [120], "model__learning_rate": [0.05, 0.1]},
        "knn_cls": {"model__n_neighbors": [5, 11]},
        "logistic_strong_reg": {"model__C": [0.05, 0.1]},
        "svc_linear": {"model__C": [0.3, 1.0], "model__kernel": ["linear"]},
        "rf_shallow": {"model__n_estimators": [100], "model__max_depth": [6, 10]},
        "gb_fast": {"model__n_estimators": [80], "model__learning_rate": [0.1, 0.2]},
        "knn_distance": {"model__n_neighbors": [7, 13], "model__weights": ["distance"]},
    }


def _regression_model_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "ridge": {"model__alpha": [0.1, 1.0, 5.0]},
        "elasticnet": {"model__alpha": [0.01, 0.1], "model__l1_ratio": [0.2, 0.8]},
        "svr_rbf": {"model__C": [0.5, 1.5], "model__epsilon": [0.05, 0.1]},
        "rf_reg": {"model__n_estimators": [120], "model__max_depth": [None, 12]},
        "gb_reg": {"model__n_estimators": [120], "model__learning_rate": [0.05, 0.1]},
        "knn_reg": {"model__n_neighbors": [5, 11]},
        "ridge_high": {"model__alpha": [10.0, 30.0]},
        "svr_linear": {"model__C": [0.5, 1.5], "model__kernel": ["linear"], "model__epsilon": [0.1]},
        "rf_reg_shallow": {"model__n_estimators": [100], "model__max_depth": [6, 10]},
        "knn_reg_distance": {"model__n_neighbors": [7, 13], "model__weights": ["distance"]},
    }


def _make_estimator(task_type: str, model_name: str):
    if task_type == "classification":
        mapping = {
            "logistic": LogisticRegression(max_iter=500),
            "svc_rbf": SVC(),
            "rf_cls": RandomForestClassifier(random_state=42),
            "gb_cls": GradientBoostingClassifier(random_state=42),
            "knn_cls": KNeighborsClassifier(),
            "logistic_strong_reg": LogisticRegression(max_iter=500),
            "svc_linear": SVC(),
            "rf_shallow": RandomForestClassifier(random_state=42),
            "gb_fast": GradientBoostingClassifier(random_state=42),
            "knn_distance": KNeighborsClassifier(),
        }
    else:
        mapping = {
            "ridge": Ridge(),
            "elasticnet": ElasticNet(max_iter=3000, random_state=42),
            "svr_rbf": SVR(),
            "rf_reg": RandomForestRegressor(random_state=42),
            "gb_reg": GradientBoostingRegressor(random_state=42),
            "knn_reg": KNeighborsRegressor(),
            "ridge_high": Ridge(),
            "svr_linear": SVR(),
            "rf_reg_shallow": RandomForestRegressor(random_state=42),
            "knn_reg_distance": KNeighborsRegressor(),
        }
    return mapping[model_name]


def _build_pipeline(task_type: str, model_name: str) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("model", _make_estimator(task_type, model_name)),
        ]
    )


def _safe_fit_pipeline_params(
    task_type: str,
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    pipe = _build_pipeline(task_type, model_name)
    allowed = set(pipe.get_params().keys())
    out: dict[str, Any] = {}
    for k, v in params.items():
        key = k if "__" in k else f"model__{k}"
        if key in allowed:
            out[key] = v
    return out


def _fit_model_job(
    _agt: pllm.AgentContext,
    *,
    dataset: str,
    task_type: str,
    model_name: str,
    model_grid: dict[str, list[Any]],
    rows_x: list[list[float]],
    rows_y: list[float | int],
    sequence_ids: list[str],
) -> dict[str, Any]:
    x_np = np.asarray(rows_x, dtype=float)
    y_np = np.asarray(rows_y)
    train_idx, val_idx, test_idx = _split_indices_by_sequence_id(sequence_ids)

    x_trainval = np.concatenate([x_np[train_idx], x_np[val_idx]], axis=0)
    y_trainval = np.concatenate([y_np[train_idx], y_np[val_idx]], axis=0)
    x_test = x_np[test_idx]
    y_test = y_np[test_idx]

    n_features = x_np.shape[1]
    max_pca = min(n_features, max(2, len(train_idx) - 1))
    pca_options = [None] + [v for v in [8, 12, 16] if v <= max_pca]

    pipe = _build_pipeline(task_type, model_name)
    param_grid = {"pca__n_components": pca_options}
    param_grid.update(model_grid)

    split = PredefinedSplit(test_fold=[-1] * len(train_idx) + [0] * len(val_idx))
    scoring = "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=split,
        n_jobs=1,
        refit=True,
    )
    gs.fit(x_trainval, y_trainval)
    y_pred = gs.predict(x_test)

    if task_type == "classification":
        test_primary = float(accuracy_score(y_test, y_pred))
        secondary = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
        better = "higher"
        metric_name = "test_accuracy"
        secondary_name = "test_f1"
    else:
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        test_primary = rmse
        secondary = r2
        better = "lower"
        metric_name = "test_rmse"
        secondary_name = "test_r2"

    return {
        "dataset": dataset,
        "task_type": task_type,
        "model_name": model_name,
        "split_rule": "sha1(sequence_id) % 10 -> train<7, val==7, test>=8",
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "best_params": gs.best_params_,
        "val_score": float(gs.best_score_),
        "metric_name": metric_name,
        "test_primary": test_primary,
        "secondary_name": secondary_name,
        "test_secondary": secondary,
        "better_is": better,
        "test_mae": mae if task_type == "regression" else None,
        "search_method": "grid",
    }


def _fit_single_config_job(
    _agt: pllm.AgentContext,
    *,
    dataset: str,
    task_type: str,
    model_name: str,
    candidate_params: dict[str, Any],
    rows_x: list[list[float]],
    rows_y: list[float | int],
    sequence_ids: list[str],
) -> dict[str, Any]:
    x_np = np.asarray(rows_x, dtype=float)
    y_np = np.asarray(rows_y)
    train_idx, val_idx, test_idx = _split_indices_by_sequence_id(sequence_ids)
    try:
        x_train = x_np[train_idx]
        y_train = y_np[train_idx]
        x_val = x_np[val_idx]
        y_val = y_np[val_idx]
        x_trainval = np.concatenate([x_train, x_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)
        x_test = x_np[test_idx]
        y_test = y_np[test_idx]

        pipe = _build_pipeline(task_type, model_name)
        final_params = _safe_fit_pipeline_params(task_type, model_name, candidate_params)
        if "pca__n_components" in final_params:
            n_features = x_np.shape[1]
            max_pca = min(n_features, max(2, len(train_idx) - 1))
            pca_value = final_params["pca__n_components"]
            if isinstance(pca_value, int):
                final_params["pca__n_components"] = max(2, min(max_pca, pca_value))
        pipe.set_params(**final_params)

        pipe.fit(x_train, y_train)
        y_val_pred = pipe.predict(x_val)
        pipe.fit(x_trainval, y_trainval)
        y_test_pred = pipe.predict(x_test)

        if task_type == "classification":
            val_score = float(accuracy_score(y_val, y_val_pred))
            test_primary = float(accuracy_score(y_test, y_test_pred))
            secondary = float(f1_score(y_test, y_test_pred, average="binary", zero_division=0))
            better = "higher"
            metric_name = "test_accuracy"
            secondary_name = "test_f1"
            test_mae = None
        else:
            val_score = -float(mean_squared_error(y_val, y_val_pred) ** 0.5)
            test_primary = float(mean_squared_error(y_test, y_test_pred) ** 0.5)
            secondary = float(r2_score(y_test, y_test_pred))
            better = "lower"
            metric_name = "test_rmse"
            secondary_name = "test_r2"
            test_mae = float(mean_absolute_error(y_test, y_test_pred))

        return {
            "dataset": dataset,
            "task_type": task_type,
            "model_name": model_name,
            "split_rule": "sha1(sequence_id) % 10 -> train<7, val==7, test>=8",
            "num_train": len(train_idx),
            "num_val": len(val_idx),
            "num_test": len(test_idx),
            "best_params": final_params,
            "val_score": val_score,
            "metric_name": metric_name,
            "test_primary": test_primary,
            "secondary_name": secondary_name,
            "test_secondary": secondary,
            "better_is": better,
            "test_mae": test_mae,
            "search_method": "llm_extra",
            "failed": False,
        }
    except Exception as exc:
        if task_type == "classification":
            return {
                "dataset": dataset,
                "task_type": task_type,
                "model_name": model_name,
                "split_rule": "sha1(sequence_id) % 10 -> train<7, val==7, test>=8",
                "num_train": len(train_idx),
                "num_val": len(val_idx),
                "num_test": len(test_idx),
                "best_params": {},
                "val_score": -1.0,
                "metric_name": "test_accuracy",
                "test_primary": -1.0,
                "secondary_name": "test_f1",
                "test_secondary": 0.0,
                "better_is": "higher",
                "test_mae": None,
                "search_method": "llm_extra",
                "failed": True,
                "error": str(exc),
            }
        return {
            "dataset": dataset,
            "task_type": task_type,
            "model_name": model_name,
            "split_rule": "sha1(sequence_id) % 10 -> train<7, val==7, test>=8",
            "num_train": len(train_idx),
            "num_val": len(val_idx),
            "num_test": len(test_idx),
            "best_params": {},
            "val_score": -1e9,
            "metric_name": "test_rmse",
            "test_primary": 1e12,
            "secondary_name": "test_r2",
            "test_secondary": -1e9,
            "better_is": "lower",
            "test_mae": None,
            "search_method": "llm_extra",
            "failed": True,
            "error": str(exc),
        }


def run_direct_openai_judge_forbidden(*_: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "Direct sequential OpenAI API calls are forbidden in this usecase. "
        "Use parallem (resume_directory + agent.ask_llm) for all LLM judging."
    )


def _run_llm_judge_with_parallem(orch: pllm.AgentOrchestrator, benchmark_rows: list[dict[str, Any]]) -> str:
    compact_rows = [
        {
            "dataset": r["dataset"],
            "task_type": r["task_type"],
            "model_name": r["model_name"],
            "metric_name": r["metric_name"],
            "test_primary": r["test_primary"],
            "secondary_name": r["secondary_name"],
            "test_secondary": r["test_secondary"],
            "better_is": r["better_is"],
        }
        for r in benchmark_rows
    ]
    with orch.agent("llm-judge") as agt:
        conv = agt.get_msg_state()
        conv.ask_llm(
            "You are model-selection judge. Use ONLY test set metrics. "
            "For each dataset, rank top-3 models and choose a winner. "
            "Output concise markdown sections: dataset, ranking, selected model, justification.",
            json.dumps(compact_rows, ensure_ascii=True),
        )
        return conv[-1].final_answer


def _build_fallback_extra_candidates(tasks: list[dict[str, Any]], num_extra_jobs: int) -> list[dict[str, Any]]:
    rng = random.Random(42)
    model_pool = {
        "classification": ["logistic", "svc_rbf", "rf_cls", "gb_cls", "knn_cls", "svc_linear"],
        "regression": ["ridge", "elasticnet", "svr_rbf", "rf_reg", "gb_reg", "knn_reg"],
    }
    extras: list[dict[str, Any]] = []
    task_cycle = list(tasks)
    while len(extras) < num_extra_jobs:
        t = task_cycle[len(extras) % len(task_cycle)]
        model_name = rng.choice(model_pool[t["task_type"]])
        candidate = {
            "dataset": t["dataset"],
            "task_type": t["task_type"],
            "model_name": model_name,
            "candidate_params": {
                "pca__n_components": rng.choice([None, 8, 12, 16]),
            },
        }
        if model_name in {"logistic", "svc_rbf", "svc_linear", "svr_rbf"}:
            candidate["candidate_params"]["C"] = rng.choice([0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
        if model_name in {"rf_cls", "rf_reg"}:
            candidate["candidate_params"]["n_estimators"] = rng.choice([80, 120, 160, 240])
            candidate["candidate_params"]["max_depth"] = rng.choice([None, 6, 10, 14])
        if model_name in {"gb_cls", "gb_reg"}:
            candidate["candidate_params"]["n_estimators"] = rng.choice([80, 120, 160])
            candidate["candidate_params"]["learning_rate"] = rng.choice([0.03, 0.05, 0.1, 0.2])
        if model_name in {"knn_cls", "knn_reg"}:
            candidate["candidate_params"]["n_neighbors"] = rng.choice([3, 5, 7, 11, 15, 21])
            candidate["candidate_params"]["weights"] = rng.choice(["uniform", "distance"])
        if model_name in {"ridge"}:
            candidate["candidate_params"]["alpha"] = rng.choice([0.05, 0.1, 0.5, 1.0, 5.0, 20.0])
        if model_name in {"elasticnet"}:
            candidate["candidate_params"]["alpha"] = rng.choice([0.001, 0.01, 0.1, 0.5])
            candidate["candidate_params"]["l1_ratio"] = rng.choice([0.1, 0.3, 0.5, 0.8])
        extras.append(candidate)
    return extras


def _propose_extra_candidates_with_llm(
    orch: pllm.AgentOrchestrator,
    tasks: list[dict[str, Any]],
    num_extra_jobs: int,
) -> list[dict[str, Any]]:
    prompt = {
        "request": (
            "Propose extra ML hyperparameter candidates for fast ranking. "
            "Keep grid search as baseline; these are additive candidates only."
        ),
        "num_extra_jobs": num_extra_jobs,
        "tasks": [
            {"dataset": t["dataset"], "task_type": t["task_type"]}
            for t in tasks
        ],
        "allowed_models": {
            "classification": ["logistic", "svc_rbf", "rf_cls", "gb_cls", "knn_cls", "svc_linear"],
            "regression": ["ridge", "elasticnet", "svr_rbf", "rf_reg", "gb_reg", "knn_reg"],
        },
        "output_schema": [
            {
                "dataset": "task dataset name",
                "task_type": "classification|regression",
                "model_name": "one allowed model name",
                "candidate_params": {"param_name_or_pipeline_name": "value"},
            }
        ],
    }
    with orch.agent("llm-hp-proposer") as agt:
        conv = agt.get_msg_state()
        conv.ask_llm(
            "Return JSON only. No markdown. "
            "Generate diverse candidates with reasonable ranges. "
            "Output a JSON array.",
            json.dumps(prompt, ensure_ascii=True),
        )
        raw = conv[-1].final_answer.strip()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("not list")
        out = [x for x in parsed if isinstance(x, dict)]
        if not out:
            raise ValueError("empty")
        return out[:num_extra_jobs]
    except Exception:
        return _build_fallback_extra_candidates(tasks, num_extra_jobs)


def _load_user_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _resolve_runtime_config(
    repo_root: Path,
    *,
    config_path: Path | None,
    strategy: str | None,
    target: str | None,
    max_candidates: int | None,
    llm: str | None,
    provider: str | None,
) -> dict[str, Any]:
    default_config_path = repo_root / "usecases/enzyme_design_arena/user/config.yaml"
    cfg_path = config_path or default_config_path
    user_cfg = _load_user_config(cfg_path)

    par_cfg = user_cfg.get("parallem", {}) if isinstance(user_cfg, dict) else {}
    pipe_cfg = user_cfg.get("pipeline", {}) if isinstance(user_cfg, dict) else {}
    oa_cfg = user_cfg.get("openai", {}) if isinstance(user_cfg, dict) else {}

    raw_strategy = strategy or par_cfg.get("strategy") or "sync"
    if raw_strategy == "async":
        raw_strategy = "concurrent"

    resolved = {
        "config_path": str(cfg_path),
        "strategy": raw_strategy,
        "target": target or pipe_cfg.get("target") or "nucb",
        "max_candidates": max_candidates or pipe_cfg.get("max_candidates") or 4000,
        "total_job_budget": int(pipe_cfg.get("total_job_budget", 100)),
        "enable_llm_extra": bool(pipe_cfg.get("enable_llm_extra", True)),
        "llm": llm or par_cfg.get("llm") or "gpt-5-mini",
        "provider": provider or par_cfg.get("provider") or "openai",
        "run_dir": par_cfg.get("run_dir") or "runs/enzyme_design_arena_ml",
        "dashboard": bool(par_cfg.get("dashboard", False)),
        "hash_by": par_cfg.get("hash_by") or ["llm"],
        "api_key_env": oa_cfg.get("api_key_env") or "OPENAI_API_KEY",
        "api_key": oa_cfg.get("api_key") or "",
        "prefer_config_key": bool(oa_cfg.get("prefer_config_key", True)),
    }
    return resolved


def _setup_api_key(resolved: dict[str, Any]) -> None:
    key_env = str(resolved["api_key_env"])
    key_inline = str(resolved["api_key"])
    prefer_config_key = bool(resolved.get("prefer_config_key", True))
    if key_inline and prefer_config_key:
        os.environ[key_env] = key_inline
    elif not os.getenv(key_env) and key_inline:
        os.environ[key_env] = key_inline

    if resolved["provider"] == "openai" and not os.getenv(key_env):
        raise RuntimeError(
            "Missing OpenAI API key. Set it in environment variable "
            f"'{key_env}' or in user config openai.api_key."
        )


def run_pipeline(
    *,
    repo_root: Path,
    strategy: str,
    target: str,
    max_candidates: int,
    total_job_budget: int,
    enable_llm_extra: bool,
    llm: str,
    provider: str,
    run_dir: str,
    dashboard: bool,
    hash_by: list[str],
    config_path: str,
) -> dict[str, Any]:
    """
    End-to-end benchmark pipeline:
    1) build 3 dataset tasks
    2) run grid search per task with sequence_id split
    3) run extra LLM-proposed hyperparameter candidates (additive)
    3) evaluate on held-out test set
    4) use LLM as a judge via parallem agent.ask_llm
    """
    run_dir_path = (repo_root / run_dir).resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)
    tasks = _build_benchmark_tasks(repo_root, max_candidates=max_candidates)

    job_specs: list[dict[str, Any]] = []
    for t in tasks:
        grids = _classification_model_grids() if t["task_type"] == "classification" else _regression_model_grids()
        for model_name, model_grid in grids.items():
            job_specs.append(
                {
                    "dataset": t["dataset"],
                    "task_type": t["task_type"],
                    "model_name": model_name,
                    "model_grid": model_grid,
                    "rows_x": t["X"],
                    "rows_y": t["y"],
                    "sequence_ids": t["sequence_ids"],
                }
            )

    base_grid_jobs = len(job_specs)
    num_extra_jobs = max(0, total_job_budget - base_grid_jobs) if enable_llm_extra else 0

    with pllm.resume_directory(
        str(run_dir_path),
        provider=provider,
        strategy=strategy,
        llm=llm,
        dashboard=dashboard,
        hash_by=hash_by,
    ) as orch:
        extra_specs: list[dict[str, Any]] = []
        if num_extra_jobs > 0:
            llm_candidates = _propose_extra_candidates_with_llm(orch, tasks, num_extra_jobs)
            task_lookup = {t["dataset"]: t for t in tasks}
            allowed_models = {
                "classification": {"logistic", "svc_rbf", "rf_cls", "gb_cls", "knn_cls", "svc_linear"},
                "regression": {"ridge", "elasticnet", "svr_rbf", "rf_reg", "gb_reg", "knn_reg"},
            }
            for c in llm_candidates[:num_extra_jobs]:
                dataset = str(c.get("dataset", ""))
                if dataset not in task_lookup:
                    continue
                t = task_lookup[dataset]
                raw_model = str(c.get("model_name", ""))
                if raw_model not in allowed_models[t["task_type"]]:
                    raw_model = "logistic" if t["task_type"] == "classification" else "ridge"
                extra_specs.append(
                    {
                        "dataset": dataset,
                        "task_type": t["task_type"],
                        "model_name": raw_model,
                        "candidate_params": c.get("candidate_params", {}) if isinstance(c.get("candidate_params", {}), dict) else {},
                        "rows_x": t["X"],
                        "rows_y": t["y"],
                        "sequence_ids": t["sequence_ids"],
                    }
                )
            if len(extra_specs) < num_extra_jobs:
                fallback = _build_fallback_extra_candidates(tasks, num_extra_jobs - len(extra_specs))
                for c in fallback:
                    t = next(x for x in tasks if x["dataset"] == c["dataset"])
                    extra_specs.append(
                        {
                            "dataset": c["dataset"],
                            "task_type": c["task_type"],
                            "model_name": c["model_name"],
                            "candidate_params": c["candidate_params"],
                            "rows_x": t["X"],
                            "rows_y": t["y"],
                            "sequence_ids": t["sequence_ids"],
                        }
                    )
        if len(extra_specs) > num_extra_jobs:
            extra_specs = extra_specs[:num_extra_jobs]

        handles = []
        for spec in job_specs:
            handles.append(
                orch.create_agent(
                    _fit_model_job,
                    dataset=spec["dataset"],
                    task_type=spec["task_type"],
                    model_name=spec["model_name"],
                    model_grid=spec["model_grid"],
                    rows_x=spec["rows_x"],
                    rows_y=spec["rows_y"],
                    sequence_ids=spec["sequence_ids"],
                    agent_name=f"fit-{spec['dataset']}-{spec['model_name']}",
                )
            )
        for i, spec in enumerate(extra_specs):
            handles.append(
                orch.create_agent(
                    _fit_single_config_job,
                    dataset=spec["dataset"],
                    task_type=spec["task_type"],
                    model_name=spec["model_name"],
                    candidate_params=spec["candidate_params"],
                    rows_x=spec["rows_x"],
                    rows_y=spec["rows_y"],
                    sequence_ids=spec["sequence_ids"],
                    agent_name=f"extra-{spec['dataset']}-{spec['model_name']}-{i}",
                )
            )

        outcomes = orch.run_agents(handles, return_exceptions=True)
        valid = [o for o in outcomes if isinstance(o, dict) and "test_primary" in o]
        judge_md = _run_llm_judge_with_parallem(orch, valid)

    if not valid:
        raise RuntimeError("No successful hyperparameter jobs returned.")
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        by_dataset.setdefault(row["dataset"], []).append(row)

    best_by_dataset = {}
    for ds, rows in by_dataset.items():
        better_is = rows[0]["better_is"]
        if better_is == "higher":
            best = max(rows, key=lambda x: x["test_primary"])
        else:
            best = min(rows, key=lambda x: x["test_primary"])
        best_by_dataset[ds] = best

    out_dir = repo_root / "usecases/enzyme_design_arena/data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "benchmark_results.json"
    results_csv = out_dir / "benchmark_results.csv"
    judge_md_path = out_dir / "benchmark_judge.md"

    results_json.write_text(
        json.dumps(
            {
                "config_path": config_path,
                "provider": provider,
                "strategy": strategy,
                "llm": llm,
                "num_jobs": len(job_specs),
                "num_grid_jobs": base_grid_jobs,
                "num_llm_extra_jobs": len(extra_specs),
                "results": valid,
                "best_by_dataset": best_by_dataset,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "dataset",
            "task_type",
            "model_name",
            "metric_name",
            "test_primary",
            "secondary_name",
            "test_secondary",
            "val_score",
            "better_is",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in valid:
            writer.writerow({k: r.get(k) for k in cols})

    judge_md_path.write_text(judge_md, encoding="utf-8")

    summary = {
        "config_path": config_path,
        "target": target,
        "provider": provider,
        "strategy": strategy,
        "llm": llm,
        "num_datasets": len(tasks),
        "num_grid_jobs": base_grid_jobs,
        "num_llm_extra_jobs": len([r for r in valid if r.get("search_method") == "llm_extra"]),
        "num_jobs": len(valid),
        "split_rule": "sha1(sequence_id) % 10 -> train<7, val==7, test>=8",
        "results_json": str(results_json),
        "results_csv": str(results_csv),
        "judge_markdown": str(judge_md_path),
        "best_by_dataset": {
            ds: {
                "model_name": v["model_name"],
                "metric_name": v["metric_name"],
                "test_primary": v["test_primary"],
            }
            for ds, v in best_by_dataset.items()
        },
    }
    summary_path = out_dir / "ml_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enzyme design arena ML demo")
    parser.add_argument("--config", default=None, help="Path to user YAML config.")
    parser.add_argument("--provider", choices=["openai", "google", "anthropic", "multi"], default=None)
    parser.add_argument("--strategy", choices=["sync", "async", "concurrent", "batch"], default=None)
    parser.add_argument("--dashboard", action="store_true", help="Enable Parallem dashboard.")
    parser.add_argument("--hash-by", default=None, help="Comma-separated hash_by fields, e.g. llm,documents.")
    parser.add_argument("--target", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--total-job-budget", type=int, default=None, help="Total evaluation jobs (grid + llm_extra).")
    parser.add_argument("--disable-llm-extra", action="store_true", help="Disable additive LLM extra hyperparameter proposals.")
    parser.add_argument("--llm", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    resolved = _resolve_runtime_config(
        repo_root,
        config_path=Path(args.config).expanduser().resolve() if args.config else None,
        strategy=args.strategy,
        target=args.target,
        max_candidates=args.max_candidates,
        llm=args.llm,
        provider=args.provider,
    )
    _setup_api_key(resolved)
    if args.dashboard:
        resolved["dashboard"] = True
    if args.hash_by:
        resolved["hash_by"] = [x.strip() for x in args.hash_by.split(",") if x.strip()]
    if args.total_job_budget is not None:
        resolved["total_job_budget"] = int(args.total_job_budget)
    if args.disable_llm_extra:
        resolved["enable_llm_extra"] = False

    print(
        json.dumps(
            run_pipeline(
                repo_root=repo_root,
                strategy=resolved["strategy"],
                target=resolved["target"],
                max_candidates=int(resolved["max_candidates"]),
                total_job_budget=int(resolved["total_job_budget"]),
                enable_llm_extra=bool(resolved["enable_llm_extra"]),
                llm=resolved["llm"],
                provider=resolved["provider"],
                run_dir=resolved["run_dir"],
                dashboard=bool(resolved["dashboard"]),
                hash_by=list(resolved["hash_by"]),
                config_path=resolved["config_path"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
