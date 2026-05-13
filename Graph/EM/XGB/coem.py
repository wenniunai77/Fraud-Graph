"""Calibrated table-GNN Co-EM training for AML edge classification."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from .calibration import (
    CalibrationMethod,
    ProbabilityCalibrator,
    fit_calibrator,
)
from .config import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_XGB_PARAMS
from .graph_experts import PnaConfig, PnaEdgeExpert
from .modeling import (
    _best_iteration_limit,
    _best_threshold_from_validation,
    _evaluate,
    _feature_columns,
    _import_xgboost,
    _make_preprocessor,
    _prepare_xgb_matrix,
    _train_xgb_booster,
    _xgb_params,
)


EmOrder = Literal["gnn-first", "table-first"]
GraphExpertName = Literal["multi-pna-eu"]
GnnPseudoSource = Literal["train", "valid", "both"]
TableGnnFeatureMode = Literal["none", "scores", "embeddings", "all"]


@dataclass(frozen=True)
class CoemConfig:
    rounds: int = 2
    patience: int = 1
    em_order: EmOrder = "table-first"
    graph_expert: GraphExpertName = "multi-pna-eu"
    table_pl_weight: float = 0.0
    gnn_pl_weight: float = 0.1
    teacher_min_confidence: float = 0.0
    teacher_max_edges_per_class: int | None = 20000
    gnn_pseudo_source: GnnPseudoSource = "both"
    table_gnn_feature_mode: TableGnnFeatureMode = "all"
    calibration_method: CalibrationMethod = "platt"
    oof_folds: int = 5
    fusion_weight_start: float = 0.0
    fusion_weight_stop: float = 1.0
    fusion_weight_step: float = 0.05
    table_n_estimators: int | None = None
    table_early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS
    gnn_warmup_epochs: int = 2
    gnn_epochs: int = 1
    max_rows: int | None = None
    xgb_device: str = "gpu"
    n_jobs: int = 12
    verbose: bool = False
    pna: PnaConfig = PnaConfig()


@dataclass(frozen=True)
class CoemResult:
    output_dir: Path
    metrics_path: Path
    best_seed: int
    best_round: int
    best_head: str
    best_validation_auprc: float
    best_test_auprc: float | None


@dataclass
class TableState:
    model: Any
    preprocessor: Any
    numeric_columns: list[str]
    categorical_columns: list[str]
    calibrator: ProbabilityCalibrator
    raw_logits: np.ndarray
    raw_prob: np.ndarray
    calibrated_logits: np.ndarray
    calibrated_prob: np.ndarray
    teacher_prob: np.ndarray
    teacher_mask: np.ndarray
    teacher_diagnostics: dict[str, Any]
    oof_coverage: float
    train_seconds: float
    oof_seconds: float
    model_path: Path
    calibrator_path: Path


@dataclass
class GnnState:
    raw_logits: np.ndarray
    raw_prob: np.ndarray
    calibrated_logits: np.ndarray
    calibrated_prob: np.ndarray
    embeddings: np.ndarray
    calibrator: ProbabilityCalibrator
    train_stats: dict[str, float]
    train_seconds: float
    predict_seconds: float
    checkpoint_path: Path
    calibrator_path: Path
    pseudo_label_diagnostics: dict[str, Any] | None = None


def train_coem(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    seeds: list[int],
    config: CoemConfig,
) -> CoemResult:
    if not seeds:
        raise ValueError("seeds must contain at least one value")
    _validate_config(config)
    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(features_path, max_rows=config.max_rows)
    _validate_graph_ready_frame(frame)
    masks = _split_masks(frame)

    all_seed_metrics: list[dict[str, Any]] = []
    best_global: dict[str, Any] | None = None
    for seed in seeds:
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_metrics = _run_seed(frame.copy(), masks, seed=seed, seed_dir=seed_dir, config=config)
        all_seed_metrics.append(seed_metrics)
        candidate = seed_metrics["best"]
        if best_global is None or candidate["validation_auprc"] > best_global["validation_auprc"]:
            best_global = {**candidate, "seed": int(seed)}

    if best_global is None:
        raise RuntimeError("Co-EM produced no round metrics")
    metrics_out = {
        "features_path": str(features_path),
        "config": _config_to_dict(config),
        "seeds": [int(seed) for seed in seeds],
        "best": best_global,
        "runs": all_seed_metrics,
    }
    metrics_path = output_dir / "coem_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    return CoemResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        best_seed=int(best_global["seed"]),
        best_round=int(best_global["round"]),
        best_head=str(best_global["head"]),
        best_validation_auprc=float(best_global["validation_auprc"]),
        best_test_auprc=best_global["test"].get("auprc"),
    )


def _run_seed(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    seed: int,
    seed_dir: Path,
    config: CoemConfig,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.float32, copy=True)
    table_frame = _table_training_frame(frame)
    table_state = _train_table_state(
        table_frame,
        masks,
        seed=seed,
        output_dir=seed_dir / "warmup_table",
        config=config,
        previous_gnn_prob=None,
    )
    _write_table_predictions(seed_dir / "warmup_table" / "table_predictions.parquet", frame, table_state)

    graph_expert = PnaEdgeExpert(
        frame,
        config=config.pna,
        seed=seed,
        architecture=config.graph_expert,
    )
    gold_edge_ids = np.flatnonzero(masks["train"])
    warmup_start = time.perf_counter()
    warmup_stats = graph_expert.fit(
        edge_ids=gold_edge_ids,
        labels=labels,
        gold_mask=masks["train"],
        teacher_prob=None,
        teacher_mask=None,
        pseudo_weight=0.0,
        epochs=config.gnn_warmup_epochs,
    )
    warmup_seconds = time.perf_counter() - warmup_start
    gnn_state = _predict_and_calibrate_gnn(
        graph_expert,
        frame,
        masks,
        output_dir=seed_dir / "warmup_gnn",
        round_name="warmup",
        config=config,
        train_stats={**warmup_stats, "warmup_seconds": float(warmup_seconds)},
    )
    _write_gnn_predictions(seed_dir / "warmup_gnn" / "gnn_predictions.parquet", frame, gnn_state)
    _write_edge_embeddings(seed_dir / "warmup_gnn" / "edge_embeddings.parquet", frame, gnn_state.embeddings)
    warmup_metrics = _round_metrics(
        frame,
        masks,
        table_state=table_state,
        gnn_state=gnn_state,
        round_id=0,
        output_dir=seed_dir / "warmup_metrics",
        config=config,
        table_name="warmup_table",
        gnn_name="warmup_gnn",
        fusion_name="warmup_fusion",
    )

    rounds: list[dict[str, Any]] = []
    best_round: dict[str, Any] | None = warmup_metrics["best"]
    no_improve = 0
    warmup_table = table_state
    current_table = table_state
    current_gnn = gnn_state

    for round_id in range(1, config.rounds + 1):
        round_dir = seed_dir / f"round_{round_id:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if config.em_order == "gnn-first":
            current_gnn = _train_gnn_round(
                graph_expert,
                frame,
                masks,
                table_state=current_table,
                output_dir=round_dir / "gnn",
                round_name=f"round_{round_id}",
                config=config,
            )
            current_table = _train_table_with_gnn(
                frame,
                masks,
                gnn_state=current_gnn,
                seed=seed + round_id,
                output_dir=round_dir / "table",
                config=config,
            )
        elif config.em_order == "table-first":
            current_table = _train_table_with_gnn(
                frame,
                masks,
                gnn_state=current_gnn,
                seed=seed + round_id,
                output_dir=round_dir / "table",
                config=config,
            )
            current_gnn = _train_gnn_round(
                graph_expert,
                frame,
                masks,
                table_state=current_table,
                output_dir=round_dir / "gnn",
                round_name=f"round_{round_id}",
                config=config,
            )
        else:
            raise ValueError(f"unknown em_order: {config.em_order}")

        _write_table_predictions(round_dir / "table_predictions.parquet", frame, current_table)
        _write_gnn_predictions(round_dir / "gnn_predictions.parquet", frame, current_gnn)
        _write_edge_embeddings(round_dir / "edge_embeddings.parquet", frame, current_gnn.embeddings)
        round_metrics = _round_metrics(
            frame,
            masks,
            table_state=current_table,
            gnn_state=current_gnn,
            round_id=round_id,
            output_dir=round_dir,
            config=config,
            anchor_table_state=warmup_table,
        )
        rounds.append(round_metrics)

        round_best = round_metrics["best"]
        if best_round is None or round_best["validation_auprc"] > best_round["validation_auprc"]:
            best_round = round_best
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= config.patience:
            break

    return {
        "seed": int(seed),
        "warmup": {
            "table_oof_coverage": float(table_state.oof_coverage),
            "gnn_train_stats": gnn_state.train_stats,
            "metrics": warmup_metrics,
        },
        "rounds": rounds,
        "best": best_round,
    }


def _train_gnn_round(
    graph_expert: PnaEdgeExpert,
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    table_state: TableState,
    output_dir: Path,
    round_name: str,
    config: CoemConfig,
) -> GnnState:
    pseudo_edges, pseudo_diagnostics = _select_gnn_teacher_mask(
        table_state.teacher_prob,
        table_state.teacher_mask,
        masks,
        config=config,
    )
    if config.gnn_pl_weight <= 0.0:
        pseudo_edges = np.zeros_like(pseudo_edges, dtype=bool)
        pseudo_diagnostics = _teacher_label_diagnostics(
            table_state.teacher_prob,
            pseudo_edges,
            masks,
        )
    train_edges = np.flatnonzero(masks["train"] | pseudo_edges)
    train_start = time.perf_counter()
    stats = graph_expert.fit(
        edge_ids=train_edges,
        labels=frame["label"].to_numpy(dtype=np.float32, copy=True),
        gold_mask=masks["train"],
        teacher_prob=table_state.teacher_prob,
        teacher_mask=table_state.teacher_mask,
        pseudo_weight=config.gnn_pl_weight,
        epochs=config.gnn_epochs,
    )
    train_seconds = time.perf_counter() - train_start
    return _predict_and_calibrate_gnn(
        graph_expert,
        frame,
        masks,
        output_dir=output_dir,
        round_name=round_name,
        config=config,
        train_stats={**stats, "train_seconds": float(train_seconds)},
        pseudo_label_diagnostics=pseudo_diagnostics,
    )


def _predict_and_calibrate_gnn(
    graph_expert: PnaEdgeExpert,
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    output_dir: Path,
    round_name: str,
    config: CoemConfig,
    train_stats: dict[str, float],
    pseudo_label_diagnostics: dict[str, Any] | None = None,
) -> GnnState:
    output_dir.mkdir(parents=True, exist_ok=True)
    predict_start = time.perf_counter()
    raw_logits, embeddings = graph_expert.predict(np.arange(len(frame), dtype=np.int64))
    predict_seconds = time.perf_counter() - predict_start
    raw_prob = expit(raw_logits).astype(np.float64, copy=False)
    calibrator = fit_calibrator(
        raw_logits[masks["valid"]],
        frame.loc[masks["valid"], "label"].to_numpy(dtype=np.int8, copy=True),
        method=config.calibration_method,
    )
    calibrated_logits = calibrator.transform_logits(raw_logits)
    calibrated_prob = calibrator.predict_proba(raw_logits)
    checkpoint_path = output_dir / f"{config.graph_expert}_{round_name}.pt"
    graph_expert.save_checkpoint(checkpoint_path)
    calibrator_path = output_dir / "gnn_calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)
    return GnnState(
        raw_logits=raw_logits.astype(np.float64, copy=False),
        raw_prob=raw_prob,
        calibrated_logits=calibrated_logits,
        calibrated_prob=calibrated_prob,
        embeddings=embeddings,
        calibrator=calibrator,
        train_stats=train_stats,
        train_seconds=float(train_stats.get("train_seconds", 0.0)),
        predict_seconds=float(predict_seconds),
        checkpoint_path=checkpoint_path,
        calibrator_path=calibrator_path,
        pseudo_label_diagnostics=pseudo_label_diagnostics,
    )


def _train_table_with_gnn(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    gnn_state: GnnState,
    seed: int,
    output_dir: Path,
    config: CoemConfig,
) -> TableState:
    table_frame = _table_training_frame(frame)
    _add_gnn_table_features(
        table_frame,
        logits=gnn_state.calibrated_logits,
        probabilities=gnn_state.calibrated_prob,
        embeddings=gnn_state.embeddings,
        mode=config.table_gnn_feature_mode,
    )
    return _train_table_state(
        table_frame,
        masks,
        seed=seed,
        output_dir=output_dir,
        config=config,
        previous_gnn_prob=gnn_state.calibrated_prob,
    )


def _add_gnn_table_features(
    frame: pd.DataFrame,
    *,
    logits: np.ndarray,
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    mode: TableGnnFeatureMode,
) -> None:
    if mode not in {"none", "scores", "embeddings", "all"}:
        raise ValueError("table_gnn_feature_mode must be one of none/scores/embeddings/all")
    if mode in {"scores", "all"}:
        frame["gnn_logit"] = logits.astype(np.float32, copy=False)
        frame["gnn_prob"] = probabilities.astype(np.float32, copy=False)
    if mode in {"embeddings", "all"}:
        for idx in range(embeddings.shape[1]):
            frame[f"edge_emb_{idx:04d}"] = embeddings[:, idx].astype(np.float32, copy=False)


def _train_table_state(
    table_frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    seed: int,
    output_dir: Path,
    config: CoemConfig,
    previous_gnn_prob: np.ndarray | None,
) -> TableState:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_start = time.perf_counter()
    model_bundle = _fit_table_model(
        table_frame,
        masks["train"],
        masks["valid"],
        seed=seed,
        config=config,
        previous_gnn_prob=previous_gnn_prob,
    )
    raw_logits = _predict_table_logits(model_bundle, table_frame, config=config)
    raw_prob = expit(raw_logits).astype(np.float64, copy=False)
    calibrator = fit_calibrator(
        raw_logits[masks["valid"]],
        table_frame.loc[masks["valid"], "label"].to_numpy(dtype=np.int8, copy=True),
        method=config.calibration_method,
    )
    calibrated_logits = calibrator.transform_logits(raw_logits)
    calibrated_prob = calibrator.predict_proba(raw_logits)
    train_seconds = time.perf_counter() - train_start

    oof_start = time.perf_counter()
    oof_logits, oof_mask = _temporal_oof_table_logits(
        table_frame,
        masks,
        seed=seed,
        config=config,
        previous_gnn_prob=previous_gnn_prob,
    )
    teacher_prob = np.zeros(len(table_frame), dtype=np.float64)
    teacher_mask = np.zeros(len(table_frame), dtype=bool)
    teacher_prob[oof_mask] = calibrator.predict_proba(oof_logits[oof_mask])
    teacher_mask[oof_mask] = True
    teacher_prob[masks["valid"]] = calibrated_prob[masks["valid"]]
    teacher_mask[masks["valid"]] = True
    teacher_diagnostics = _teacher_label_diagnostics(teacher_prob, teacher_mask, masks)
    oof_seconds = time.perf_counter() - oof_start
    oof_coverage = float(oof_mask[masks["train"]].sum() / max(1, masks["train"].sum()))

    model_path = output_dir / "table_model.joblib"
    calibrator_path = output_dir / "table_calibrator.joblib"
    joblib.dump(model_bundle, model_path)
    joblib.dump(calibrator, calibrator_path)
    return TableState(
        model=model_bundle["model"],
        preprocessor=model_bundle["preprocessor"],
        numeric_columns=model_bundle["numeric_columns"],
        categorical_columns=model_bundle["categorical_columns"],
        calibrator=calibrator,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        calibrated_logits=calibrated_logits,
        calibrated_prob=calibrated_prob,
        teacher_prob=teacher_prob,
        teacher_mask=teacher_mask,
        teacher_diagnostics=teacher_diagnostics,
        oof_coverage=oof_coverage,
        train_seconds=float(train_seconds),
        oof_seconds=float(oof_seconds),
        model_path=model_path,
        calibrator_path=calibrator_path,
    )


def _fit_table_model(
    table_frame: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    seed: int,
    config: CoemConfig,
    previous_gnn_prob: np.ndarray | None,
) -> dict[str, Any]:
    xgb = _import_xgboost()
    numeric_columns, categorical_columns = _feature_columns(table_frame)
    preprocessor = _make_preprocessor(numeric_columns, categorical_columns)
    x_train = table_frame.loc[train_mask].drop(columns=["label"])
    x_valid = table_frame.loc[valid_mask].drop(columns=["label"])
    y_train = table_frame.loc[train_mask, "label"].to_numpy(dtype=np.float32, copy=True)
    if previous_gnn_prob is not None and config.table_pl_weight > 0.0:
        y_train = (
            (1.0 - config.table_pl_weight) * y_train
            + config.table_pl_weight * previous_gnn_prob[train_mask].astype(np.float32)
        )
    y_valid = table_frame.loc[valid_mask, "label"].to_numpy(dtype=np.float32, copy=True)
    x_train_t, train_kind = _prepare_xgb_matrix(preprocessor.fit_transform(x_train))
    x_valid_t, _ = _prepare_xgb_matrix(preprocessor.transform(x_valid))
    dtrain = xgb.DMatrix(x_train_t, label=y_train, nthread=int(config.n_jobs))
    dvalid = xgb.DMatrix(x_valid_t, label=y_valid, nthread=int(config.n_jobs))
    params = _coem_xgb_params(seed, config)
    model = _train_xgb_booster(
        dtrain,
        dvalid,
        params=params,
        early_stopping_rounds=config.table_early_stopping_rounds,
        verbose=config.verbose,
    )
    return {
        "model": model,
        "preprocessor": preprocessor,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "xgb_params": params,
        "xgb_matrix_kind": train_kind,
    }


def _predict_table_logits(
    model_bundle: dict[str, Any],
    table_frame: pd.DataFrame,
    *,
    config: CoemConfig,
) -> np.ndarray:
    xgb = _import_xgboost()
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]
    x_all = table_frame.drop(columns=["label"])
    transformed, _ = _prepare_xgb_matrix(preprocessor.transform(x_all))
    dmatrix = xgb.DMatrix(transformed, nthread=int(config.n_jobs))
    best_limit = _best_iteration_limit(model)
    kwargs: dict[str, Any] = {"output_margin": True}
    if best_limit is not None:
        kwargs["iteration_range"] = (0, best_limit)
    return np.asarray(model.predict(dmatrix, **kwargs), dtype=np.float64)


def _temporal_oof_table_logits(
    table_frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    seed: int,
    config: CoemConfig,
    previous_gnn_prob: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if config.oof_folds < 2:
        raise ValueError("oof_folds must be at least 2")
    train_indices = np.flatnonzero(masks["train"])
    order = np.argsort(
        table_frame.iloc[train_indices]["timestamp_seconds"].to_numpy(dtype=np.int64),
        kind="mergesort",
    )
    ordered_train = train_indices[order]
    folds = np.array_split(ordered_train, int(config.oof_folds))
    logits = np.full(len(table_frame), np.nan, dtype=np.float64)
    mask = np.zeros(len(table_frame), dtype=bool)
    oof_config = replace(config, table_early_stopping_rounds=0)
    for fold_id in range(1, len(folds)):
        history = np.concatenate(folds[:fold_id])
        target = folds[fold_id]
        if history.size == 0 or target.size == 0:
            raise RuntimeError("temporal OOF produced an empty history or target fold")
        history_mask = np.zeros(len(table_frame), dtype=bool)
        target_mask = np.zeros(len(table_frame), dtype=bool)
        history_mask[history] = True
        target_mask[target] = True
        if len(np.unique(table_frame.loc[history_mask, "label"].to_numpy())) < 2:
            raise ValueError(
                f"temporal OOF fold {fold_id} history lacks both classes; "
                "increase data size or reduce oof_folds"
            )
        model_bundle = _fit_table_model(
            table_frame,
            history_mask,
            target_mask,
            seed=seed + 1000 + fold_id,
            config=oof_config,
            previous_gnn_prob=previous_gnn_prob,
        )
        fold_logits = _predict_table_logits(
            model_bundle,
            table_frame.loc[target_mask],
            config=oof_config,
        )
        logits[target_mask] = fold_logits
        mask[target_mask] = True
    return logits, mask


def _select_gnn_teacher_mask(
    teacher_prob: np.ndarray,
    teacher_mask: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    config: CoemConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    probabilities = np.asarray(teacher_prob, dtype=np.float64)
    selected = np.asarray(teacher_mask, dtype=bool).copy()
    if config.gnn_pseudo_source == "train":
        selected &= masks["train"]
    elif config.gnn_pseudo_source == "valid":
        selected &= masks["valid"]
    elif config.gnn_pseudo_source == "both":
        selected &= masks["train"] | masks["valid"]
    else:
        raise ValueError("gnn_pseudo_source must be train, valid, or both")
    selected &= ~masks["test"]

    confidence = np.maximum(probabilities, 1.0 - probabilities)
    if config.teacher_min_confidence > 0.0:
        selected &= confidence >= float(config.teacher_min_confidence)

    if config.teacher_max_edges_per_class is not None:
        cap = int(config.teacher_max_edges_per_class)
        if cap <= 0:
            selected[:] = False
        else:
            capped = np.zeros_like(selected, dtype=bool)
            for positive_class in (False, True):
                class_mask = selected & ((probabilities >= 0.5) == positive_class)
                class_indices = np.flatnonzero(class_mask)
                if class_indices.size == 0:
                    continue
                order = np.argsort(-confidence[class_indices], kind="mergesort")
                capped[class_indices[order[:cap]]] = True
            selected = capped
    return selected, _teacher_label_diagnostics(probabilities, selected, masks)


def _teacher_label_diagnostics(
    probabilities: np.ndarray,
    mask: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    selected_prob = probabilities[selected]
    confidence = np.maximum(selected_prob, 1.0 - selected_prob)
    return {
        "total": int(selected.sum()),
        "train": int((selected & masks["train"]).sum()),
        "valid": int((selected & masks["valid"]).sum()),
        "test": int((selected & masks["test"]).sum()),
        "positive": int((selected & (probabilities >= 0.5)).sum()),
        "negative": int((selected & (probabilities < 0.5)).sum()),
        "coverage": float(selected.sum() / max(1, selected.shape[0])),
        "confidence_mean": float(confidence.mean()) if confidence.size else 0.0,
        "confidence_min": float(confidence.min()) if confidence.size else 0.0,
        "confidence_max": float(confidence.max()) if confidence.size else 0.0,
        "prob_mean": float(selected_prob.mean()) if selected_prob.size else 0.0,
    }


def _round_metrics(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    table_state: TableState,
    gnn_state: GnnState,
    round_id: int,
    output_dir: Path,
    config: CoemConfig,
    table_name: str = "table",
    gnn_name: str = "gnn",
    fusion_name: str = "fusion",
    anchor_table_state: TableState | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_head = _head_metrics(
        frame,
        masks,
        name=table_name,
        raw_prob=table_state.raw_prob,
        calibrated_prob=table_state.calibrated_prob,
    )
    gnn_head = _head_metrics(
        frame,
        masks,
        name=gnn_name,
        raw_prob=gnn_state.raw_prob,
        calibrated_prob=gnn_state.calibrated_prob,
    )
    fusion_head = _fusion_metrics(
        frame,
        masks,
        table_state,
        gnn_state,
        config,
        name=fusion_name,
    )
    heads = [table_head, gnn_head, fusion_head]
    if anchor_table_state is not None:
        heads.append(
            _fusion_metrics(
                frame,
                masks,
                anchor_table_state,
                gnn_state,
                config,
                name="warmup_table_fusion",
            )
        )
    best = max(heads, key=lambda item: item["validation_auprc"])
    best = {**best, "round": int(round_id)}
    fusion_heads = [
        {
            "head": head["head"],
            "weight_table": head["weight_table"],
            "weight_gnn": head["weight_gnn"],
            "validation_auprc": head["validation_auprc"],
        }
        for head in heads
        if "weight_table" in head
    ]
    fusion_path = output_dir / "fusion.json"
    fusion_path.write_text(
        json.dumps(
            fusion_heads[0] if len(fusion_heads) == 1 else {"heads": fusion_heads},
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = {
        "round": int(round_id),
        "table_oof_coverage": float(table_state.oof_coverage),
        "table_train_seconds": float(table_state.train_seconds),
        "table_oof_seconds": float(table_state.oof_seconds),
        "gnn_train_seconds": float(gnn_state.train_seconds),
        "gnn_predict_seconds": float(gnn_state.predict_seconds),
        "table_teacher_labels": table_state.teacher_diagnostics,
        "table_calibrator": table_state.calibrator.to_dict(),
        "gnn_calibrator": gnn_state.calibrator.to_dict(),
        "heads": heads,
        "best": best,
    }
    if gnn_state.pseudo_label_diagnostics is not None:
        metrics["gnn_pseudo_labels"] = gnn_state.pseudo_label_diagnostics
    (output_dir / "round_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def _head_metrics(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    name: str,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
) -> dict[str, Any]:
    y_valid = frame.loc[masks["valid"], "label"].astype("int8")
    y_test = frame.loc[masks["test"], "label"].astype("int8")
    valid_scores = calibrated_prob[masks["valid"]]
    test_scores = calibrated_prob[masks["test"]]
    threshold, valid_best_f1 = _best_threshold_from_validation(y_valid, valid_scores)
    valid_metric = _evaluate(y_valid, valid_scores, threshold)
    test_metric = _evaluate(y_test, test_scores, threshold)
    raw_valid_metric = _evaluate(y_valid, raw_prob[masks["valid"]], threshold)
    return {
        "head": name,
        "validation_auprc": valid_metric["auprc"],
        "validation_best_f1": float(valid_best_f1),
        "threshold": float(threshold),
        "validation": valid_metric,
        "test": test_metric,
        "raw_validation": raw_valid_metric,
    }


def _fusion_metrics(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    table_state: TableState,
    gnn_state: GnnState,
    config: CoemConfig,
    *,
    name: str = "fusion",
) -> dict[str, Any]:
    y_valid = frame.loc[masks["valid"], "label"].astype("int8")
    y_test = frame.loc[masks["test"], "label"].astype("int8")
    best_weight = None
    best_auprc = -math.inf
    best_valid_prob = None
    for weight in _fusion_weight_grid(config):
        valid_logits = (
            weight * table_state.calibrated_logits[masks["valid"]]
            + (1.0 - weight) * gnn_state.calibrated_logits[masks["valid"]]
        )
        valid_prob = expit(valid_logits)
        metric = _evaluate(y_valid, valid_prob, 0.5)
        auprc = -math.inf if metric["auprc"] is None else float(metric["auprc"])
        if auprc > best_auprc:
            best_auprc = auprc
            best_weight = float(weight)
            best_valid_prob = valid_prob
    if best_weight is None or best_valid_prob is None:
        raise RuntimeError("fusion grid produced no candidate weights")
    test_prob = expit(
        best_weight * table_state.calibrated_logits[masks["test"]]
        + (1.0 - best_weight) * gnn_state.calibrated_logits[masks["test"]]
    )
    threshold, valid_best_f1 = _best_threshold_from_validation(y_valid, best_valid_prob)
    valid_metric = _evaluate(y_valid, best_valid_prob, threshold)
    test_metric = _evaluate(y_test, test_prob, threshold)
    return {
        "head": name,
        "weight_table": float(best_weight),
        "weight_gnn": float(1.0 - best_weight),
        "validation_auprc": valid_metric["auprc"],
        "validation_best_f1": float(valid_best_f1),
        "threshold": float(threshold),
        "validation": valid_metric,
        "test": test_metric,
        "raw_validation": valid_metric,
    }


def _write_table_predictions(path: Path, frame: pd.DataFrame, state: TableState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(),
            "table_logit_raw": state.raw_logits,
            "table_prob_raw": state.raw_prob,
            "table_logit": state.calibrated_logits,
            "table_prob": state.calibrated_prob,
            "teacher_mask": state.teacher_mask,
            "teacher_prob": state.teacher_prob,
        }
    ).to_parquet(path, index=False)


def _write_gnn_predictions(path: Path, frame: pd.DataFrame, state: GnnState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(),
            "gnn_logit_raw": state.raw_logits,
            "gnn_prob_raw": state.raw_prob,
            "gnn_logit": state.calibrated_logits,
            "gnn_prob": state.calibrated_prob,
        }
    ).to_parquet(path, index=False)


def _write_edge_embeddings(path: Path, frame: pd.DataFrame, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"row_id": frame["row_id"].to_numpy()}
    for idx in range(embeddings.shape[1]):
        data[f"edge_emb_{idx:04d}"] = embeddings[:, idx]
    pd.DataFrame(data).to_parquet(path, index=False)


def _load_frame(features_path: Path, *, max_rows: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(features_path)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        pieces = []
        total = len(frame)
        for split_name in ("train", "valid", "test"):
            split_frame = frame.loc[frame["split"] == split_name]
            take = max(1, int(round(max_rows * len(split_frame) / total)))
            pieces.append(split_frame.iloc[:take])
        frame = pd.concat(pieces, axis=0).sort_values(
            ["timestamp_seconds", "row_id"],
            kind="mergesort",
        )
    return frame.reset_index(drop=True)


def _table_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["edge_id", "src_node_id", "dst_node_id"], errors="ignore").copy()


def _split_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {
        "train": (frame["split"] == "train").to_numpy(dtype=bool),
        "valid": (frame["split"] == "valid").to_numpy(dtype=bool),
        "test": (frame["split"] == "test").to_numpy(dtype=bool),
    }
    missing = [name for name, mask in masks.items() if not bool(mask.any())]
    if missing:
        raise ValueError(f"feature table is missing required splits: {missing}")
    return masks


def _validate_graph_ready_frame(frame: pd.DataFrame) -> None:
    required = {"row_id", "split", "label", "edge_id", "src_node_id", "dst_node_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Co-EM requires a graph-ready feature table; missing columns: "
            f"{missing}. Regenerate with `features --graph-ready`."
        )
    for split_name in ("train", "valid", "test"):
        y = frame.loc[frame["split"] == split_name, "label"].to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError(f"{split_name} split must contain both classes")


def _coem_xgb_params(seed: int, config: CoemConfig) -> dict[str, Any]:
    overrides = None
    if config.table_n_estimators is not None:
        overrides = {"n_estimators": int(config.table_n_estimators)}
    else:
        overrides = {"n_estimators": int(DEFAULT_XGB_PARAMS["n_estimators"])}
    return _xgb_params(
        seed,
        overrides,
        config.n_jobs,
        xgb_device=config.xgb_device,
    )


def _fusion_weight_grid(config: CoemConfig) -> np.ndarray:
    if config.fusion_weight_step <= 0:
        raise ValueError("fusion_weight_step must be positive")
    if config.fusion_weight_stop < config.fusion_weight_start:
        raise ValueError("fusion_weight_stop must be >= fusion_weight_start")
    count = int(
        math.floor(
            (config.fusion_weight_stop - config.fusion_weight_start)
            / config.fusion_weight_step
        )
    )
    values = config.fusion_weight_start + np.arange(count + 1) * config.fusion_weight_step
    if values[-1] < config.fusion_weight_stop - 1e-12:
        values = np.append(values, config.fusion_weight_stop)
    return np.clip(values, 0.0, 1.0)


def _validate_config(config: CoemConfig) -> None:
    if config.rounds <= 0:
        raise ValueError("rounds must be positive")
    if config.rounds > 3:
        raise ValueError("rounds > 3 is intentionally blocked for Co-EM stability")
    if config.patience <= 0:
        raise ValueError("patience must be positive")
    for name, value in (
        ("table_pl_weight", config.table_pl_weight),
        ("gnn_pl_weight", config.gnn_pl_weight),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if config.em_order not in {"gnn-first", "table-first"}:
        raise ValueError("em_order must be 'gnn-first' or 'table-first'")
    if config.graph_expert != "multi-pna-eu":
        raise ValueError("graph_expert must be 'multi-pna-eu'")
    if not 0.0 <= config.teacher_min_confidence <= 1.0:
        raise ValueError("teacher_min_confidence must be in [0, 1]")
    if (
        config.teacher_max_edges_per_class is not None
        and config.teacher_max_edges_per_class < 0
    ):
        raise ValueError("teacher_max_edges_per_class must be non-negative or None")
    if config.gnn_pseudo_source not in {"train", "valid", "both"}:
        raise ValueError("gnn_pseudo_source must be train, valid, or both")
    if config.table_gnn_feature_mode not in {"none", "scores", "embeddings", "all"}:
        raise ValueError("table_gnn_feature_mode must be none/scores/embeddings/all")
    if not config.pna.num_neighbors:
        raise ValueError("pna.num_neighbors must contain at least one fanout")
    if config.pna.towers <= 0:
        raise ValueError("pna.towers must be positive")
    if config.pna.hidden_dim < config.pna.towers:
        raise ValueError("pna.hidden_dim must be at least pna.towers")
    if config.pna.batch_size <= 0:
        raise ValueError("pna.batch_size must be positive")


def _config_to_dict(config: CoemConfig) -> dict[str, Any]:
    out = asdict(config)
    out["pna"] = asdict(config.pna)
    return out
