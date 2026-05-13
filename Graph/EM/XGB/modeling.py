"""Training and evaluation for engineered AML features with XGBoost."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

from .config import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_XGB_PARAMS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


CATEGORICAL_COLUMNS = [
    "receiving_currency",
    "payment_currency",
    "payment_format",
]
NON_FEATURE_COLUMNS = {"row_id", "timestamp", "split", "label"}
GRAPH_ID_COLUMNS = {"edge_id", "src_node_id", "dst_node_id"}
NON_FEATURE_COLUMNS = NON_FEATURE_COLUMNS | GRAPH_ID_COLUMNS
ENGINEERED_FEATURE_PREFIXES = ("gfp_", "gad_", "gnn_", "edge_emb_")


@dataclass(frozen=True)
class TrainResult:
    model_name: str
    output_dir: Path
    metrics_path: Path
    model_paths: list[Path]
    test_auprc_mean: float


@dataclass(frozen=True)
class XgbMatrixSet:
    train: Any
    valid: Any
    test: Any
    matrix_kind: str
    n_features: int
    transform_seconds: float
    dmatrix_seconds: float


def _import_xgboost():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing strict dependency 'xgboost'. Install xgboost before training."
        ) from exc

    return xgb


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    engineered = [
        column
        for column in df.columns
        if str(column).startswith(ENGINEERED_FEATURE_PREFIXES)
    ]
    if engineered:
        return engineered, []

    categorical = [column for column in CATEGORICAL_COLUMNS if column in df.columns]
    numeric = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and column not in categorical
    ]
    return numeric, categorical


def _make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=0.3,
    )


def _split_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = df["split"] == "train"
    valid = df["split"] == "valid"
    test = df["split"] == "test"
    if not train.any() or not valid.any() or not test.any():
        raise ValueError("feature file must contain train, valid, and test splits")
    feature_df = df.drop(columns=["label"])
    y = df["label"].astype("int8")
    return (
        feature_df.loc[train],
        y.loc[train],
        feature_df.loc[valid],
        y.loc[valid],
        feature_df.loc[test],
        y.loc[test],
    )


def _prepare_xgb_matrix(matrix: Any) -> tuple[Any, str]:
    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=False)
        if csr.dtype != np.float32:
            csr = csr.astype(np.float32)
        csr.sort_indices()
        csr.eliminate_zeros()
        return csr, "csr"

    dense = np.asarray(matrix, dtype=np.float32, order="C")
    return dense, "dense"


def _build_xgb_matrices(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_jobs: int,
) -> XgbMatrixSet:
    xgb = _import_xgboost()
    transform_start = time.perf_counter()
    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)
    x_test_t = preprocessor.transform(x_test)
    x_train_t, train_kind = _prepare_xgb_matrix(x_train_t)
    x_valid_t, _ = _prepare_xgb_matrix(x_valid_t)
    x_test_t, _ = _prepare_xgb_matrix(x_test_t)
    transform_seconds = time.perf_counter() - transform_start

    dmatrix_start = time.perf_counter()
    dtrain = xgb.DMatrix(
        x_train_t,
        label=y_train.to_numpy(dtype=np.float32, copy=False),
        nthread=int(n_jobs),
    )
    dvalid = xgb.DMatrix(
        x_valid_t,
        label=y_valid.to_numpy(dtype=np.float32, copy=False),
        nthread=int(n_jobs),
    )
    dtest = xgb.DMatrix(
        x_test_t,
        label=y_test.to_numpy(dtype=np.float32, copy=False),
        nthread=int(n_jobs),
    )
    dmatrix_seconds = time.perf_counter() - dmatrix_start

    return XgbMatrixSet(
        train=dtrain,
        valid=dvalid,
        test=dtest,
        matrix_kind=train_kind,
        n_features=int(x_train_t.shape[1]),
        transform_seconds=float(transform_seconds),
        dmatrix_seconds=float(dmatrix_seconds),
    )


def _evaluate(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_array = np.asarray(y_true)
    pred = (scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    positive_count = int(np.sum(y_array == 1))
    unique_label_count = int(len(np.unique(y_array)))
    if unique_label_count < 2:
        average_precision = None
        roc_auc = None
    else:
        average_precision = float(average_precision_score(y_true, scores))
        roc_auc = float(roc_auc_score(y_true, scores))
    return {
        "threshold": float(threshold),
        "minority_precision": float(precision),
        "minority_recall": float(recall),
        "minority_f1": float(f1),
        "minority_support": positive_count,
        "average_precision": average_precision,
        "auprc": average_precision,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _best_iteration_limit(model: Any) -> int | None:
    try:
        best_iteration = model.best_iteration
    except Exception:
        return None
    if best_iteration is None:
        return None
    return int(best_iteration) + 1


def _best_score(model: Any) -> float | None:
    try:
        return float(model.best_score)
    except Exception:
        return None


def _predict_scores(model: Any, dmatrix: Any) -> np.ndarray:
    best_limit = _best_iteration_limit(model)
    if best_limit is None:
        scores = model.predict(dmatrix)
    else:
        scores = model.predict(dmatrix, iteration_range=(0, best_limit))
    return np.asarray(scores, dtype=np.float64)


def _best_threshold_from_validation(
    y_valid: pd.Series,
    scores: np.ndarray,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_valid, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    index = int(np.nanargmax(f1_values))
    return float(thresholds[index]), float(f1_values[index])


def _xgb_params(
    seed: int,
    overrides: dict[str, Any] | None,
    n_jobs: int,
    *,
    xgb_device: str,
) -> dict[str, Any]:
    if xgb_device not in {"gpu", "cpu"}:
        raise ValueError("xgb_device must be either 'gpu' or 'cpu'")

    params = dict(DEFAULT_XGB_PARAMS)
    if overrides:
        params.update(overrides)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "gpu_hist" if xgb_device == "gpu" else "hist",
            "predictor": "gpu_predictor" if xgb_device == "gpu" else "cpu_predictor",
            "nthread": int(n_jobs),
            "seed": int(seed),
        }
    )
    return params


def _split_train_params(params: dict[str, Any]) -> tuple[dict[str, Any], int]:
    train_params = dict(params)
    num_boost_round = int(train_params.pop("n_estimators"))
    return train_params, num_boost_round


def _train_xgb_booster(
    dtrain: Any,
    dvalid: Any,
    *,
    params: dict[str, Any],
    early_stopping_rounds: int,
    verbose: bool,
) -> Any:
    xgb = _import_xgboost()
    train_params, num_boost_round = _split_train_params(params)
    return xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=(
            int(early_stopping_rounds) if early_stopping_rounds > 0 else None
        ),
        verbose_eval=verbose,
    )


def _sample_xgb_configs(n_configs: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    configs: list[dict[str, Any]] = []
    for _ in range(n_configs):
        configs.append(
            {
                "n_estimators": int(rng.integers(10, 1001)),
                "max_depth": int(rng.integers(1, 16)),
                "learning_rate": float(10 ** rng.uniform(-2.5, -1.0)),
                "reg_lambda": float(10 ** rng.uniform(-2.0, 2.0)),
                "scale_pos_weight": float(rng.uniform(1.0, 10.0)),
                "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "subsample": float(rng.uniform(0.5, 1.0)),
            }
        )
    return configs


def _successive_halving_xgb(
    dtrain: Any,
    n_train_rows: int,
    dvalid: Any,
    y_valid: pd.Series,
    *,
    x0: int,
    eta: float,
    r0: float,
    seed: int,
    n_jobs: int,
    threshold: float,
    xgb_device: str,
    early_stopping_rounds: int,
    verbose: bool,
) -> dict[str, Any]:
    configs = _sample_xgb_configs(x0, seed)
    rng = np.random.default_rng(seed)
    train_order = rng.permutation(n_train_rows).astype(np.int32, copy=False)
    frac = r0
    round_id = 0
    history: list[dict[str, Any]] = []

    while configs and frac <= 1.0000001:
        n_subset = max(1, int(n_train_rows * min(frac, 1.0)))
        subset_idx = np.ascontiguousarray(train_order[:n_subset], dtype=np.int32)
        subset_train = dtrain.slice(subset_idx)
        scored: list[tuple[float, dict[str, Any]]] = []
        for config in configs:
            model = _train_xgb_booster(
                subset_train,
                dvalid,
                params=_xgb_params(
                    seed + round_id,
                    config,
                    n_jobs,
                    xgb_device=xgb_device,
                ),
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
            valid_scores = _predict_scores(model, dvalid)
            metric = _evaluate(y_valid, valid_scores, threshold)
            auprc = metric["auprc"]
            scored.append((-math.inf if auprc is None else float(auprc), config))
            history.append(
                {
                    "round": int(round_id),
                    "train_fraction": float(min(frac, 1.0)),
                    "validation_auprc": auprc,
                    "minority_f1": metric["minority_f1"],
                    "params": config,
                }
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        keep = max(1, int(math.ceil(len(scored) / eta)))
        configs = [config for _, config in scored[:keep]]
        if frac >= 1.0:
            break
        frac = min(1.0, frac * eta)
        round_id += 1

    best = max(
        history,
        key=lambda item: (
            -math.inf
            if item["validation_auprc"] is None
            else float(item["validation_auprc"])
        ),
    )
    return {"best_params": best["params"], "history": history}


def _drop_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered_columns = [
        column
        for column in df.columns
        if str(column).startswith(ENGINEERED_FEATURE_PREFIXES)
    ]
    if not engineered_columns:
        return df
    return df.drop(columns=engineered_columns)


def _plot_metrics_bar_chart(
    all_metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    model_name: str,
) -> Path | None:
    """Generate bar charts for AUPRC, AUROC, Minority F1, Precision, and Recall.

    Produces two charts:
    1. Per-seed grouped bar chart comparing the five metrics across seeds.
    2. Summary bar chart with mean ± std error bars.
    """
    if not _HAS_MATPLOTLIB:
        return None

    metric_keys = [
        ("auprc", "AUPRC"),
        ("roc_auc", "AUROC"),
        ("minority_f1", "Minority F1"),
        ("minority_precision", "Precision"),
        ("minority_recall", "Recall"),
    ]
    seeds = [run["seed"] for run in all_metrics]

    # ------------------------------------------------------------------
    # Chart 1: per-seed grouped bar chart
    # ------------------------------------------------------------------
    n_metrics = len(metric_keys)
    n_seeds = len(seeds)
    x = np.arange(n_metrics)
    width = 0.8 / n_seeds

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, n_seeds))

    for i, run in enumerate(all_metrics):
        test = run["test"]
        values = [test[key] for key, _ in metric_keys]
        bars = ax1.bar(
            x + (i - (n_seeds - 1) / 2) * width,
            values,
            width,
            label=f"seed={run['seed']}",
            color=colors[i],
            edgecolor="white",
            linewidth=0.6,
        )
        # Annotate each bar with its value
        for bar, val in zip(bars, values):
            if val is not None:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([label for _, label in metric_keys], fontsize=11)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title(
        f"Test Set Metrics per Seed ({model_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()

    per_seed_path = output_dir / f"{model_name}_metrics_per_seed.png"
    fig1.savefig(per_seed_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Chart 2: summary bar chart with mean ± std
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    summary_keys = [
        ("test_auprc_mean", "test_auprc_std", "AUPRC"),
        ("test_auroc_mean", "test_auroc_std", "AUROC"),
        ("test_minority_f1_mean", "test_minority_f1_std", "Minority F1"),
        ("test_precision_mean", "test_precision_std", "Precision"),
        ("test_recall_mean", "test_recall_std", "Recall"),
    ]

    means = []
    stds = []
    labels = []
    for mean_key, std_key, label in summary_keys:
        means.append(summary[mean_key])
        stds.append(summary[std_key])
        labels.append(label)

    bar_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
    bars = ax2.bar(labels, means, yerr=stds, capsize=6, color=bar_colors, edgecolor="white", linewidth=1.2)

    for bar, mean_val, std_val in zip(bars, means, stds):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_val + 0.012,
            f"{mean_val:.4f} ± {std_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title(
        f"Test Set Metrics Summary — Mean ± Std ({model_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_ylim(0, max(means) + max(stds) + 0.18)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()

    summary_path = output_dir / f"{model_name}_metrics_summary.png"
    fig2.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return per_seed_path


def train_models(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "xgboost",
    seeds: list[int],
    threshold: float | None = None,
    n_jobs: int = 12,
    tune_xgb: bool = False,
    xgb_x0: int = 1000,
    xgb_eta: float = 2.0,
    xgb_r0: float = 0.1,
    xgb_device: str = "gpu",
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    drop_engineered_features: bool = False,
    verbose: bool = False,
) -> TrainResult:
    if model_name != "xgboost":
        raise ValueError("Only 'xgboost' is supported.")
    if not seeds:
        raise ValueError("seeds must contain at least one value")

    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    if drop_engineered_features:
        df = _drop_engineered_features(df)
    numeric_columns, categorical_columns = _feature_columns(df)
    selected_threshold = 0.5 if threshold is None else float(threshold)
    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    ) = _split_frame(df)

    preprocessor = _make_preprocessor(numeric_columns, categorical_columns)
    matrices = _build_xgb_matrices(
        preprocessor,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        n_jobs=n_jobs,
    )

    tuning_result: dict[str, Any] | None = None
    xgb_overrides: dict[str, Any] | None = None
    if tune_xgb:
        tuning_result = _successive_halving_xgb(
            matrices.train,
            len(y_train),
            matrices.valid,
            y_valid.reset_index(drop=True),
            x0=xgb_x0,
            eta=xgb_eta,
            r0=xgb_r0,
            seed=seeds[0],
            n_jobs=n_jobs,
            threshold=selected_threshold,
            xgb_device=xgb_device,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        xgb_overrides = tuning_result["best_params"]

    all_metrics: list[dict[str, Any]] = []
    model_paths: list[Path] = []

    for seed in seeds:
        train_start = time.perf_counter()
        params = _xgb_params(seed, xgb_overrides, n_jobs, xgb_device=xgb_device)
        model = _train_xgb_booster(
            matrices.train,
            matrices.valid,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        train_seconds = time.perf_counter() - train_start
        valid_scores = _predict_scores(model, matrices.valid)
        test_scores = _predict_scores(model, matrices.test)
        best_threshold, validation_best_f1 = _best_threshold_from_validation(
            y_valid,
            valid_scores,
        )
        metrics = {
            "seed": int(seed),
            "selected_threshold": float(selected_threshold),
            "auto_threshold": float(best_threshold),
            "train_seconds": float(train_seconds),
            "xgb_params": params,
            "best_iteration": None
            if _best_iteration_limit(model) is None
            else _best_iteration_limit(model) - 1,
            "best_score": _best_score(model),
            "validation": _evaluate(y_valid, valid_scores, selected_threshold),
            "validation_at_best_threshold": {
                **_evaluate(y_valid, valid_scores, best_threshold),
                "best_validation_f1": float(validation_best_f1),
            },
            "test": _evaluate(y_test, test_scores, best_threshold),
            "test_at_fixed_threshold": _evaluate(
                y_test,
                test_scores,
                selected_threshold,
            ),
        }
        all_metrics.append(metrics)

        model_path = output_dir / f"{model_name}_seed{seed}.joblib"
        joblib.dump(
            {
                "preprocessor": preprocessor,
                "model": model,
                "booster": model,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "threshold": float(best_threshold),
                "model_name": model_name,
                "xgb_device": xgb_device,
            },
            model_path,
        )
        model_paths.append(model_path)

    test_auprc_values = [
        metrics["test"]["auprc"]
        for metrics in all_metrics
        if metrics["test"]["auprc"] is not None
    ]
    test_auroc_values = [
        metrics["test"]["roc_auc"]
        for metrics in all_metrics
        if metrics["test"]["roc_auc"] is not None
    ]
    test_f1_values = [metrics["test"]["minority_f1"] for metrics in all_metrics]
    test_precision_values = [
        metrics["test"]["minority_precision"] for metrics in all_metrics
    ]
    test_recall_values = [
        metrics["test"]["minority_recall"] for metrics in all_metrics
    ]
    auto_thresholds = [metrics["auto_threshold"] for metrics in all_metrics]
    if not test_auprc_values:
        raise ValueError("test split must contain both classes to summarize AUPRC")
    summary = {
        "test_auprc_mean": float(np.mean(test_auprc_values)),
        "test_auprc_std": float(np.std(test_auprc_values)),
        "test_auprc_percent_mean": float(np.mean(test_auprc_values) * 100.0),
        "test_auprc_percent_std": float(np.std(test_auprc_values) * 100.0),
        "test_auroc_mean": float(np.mean(test_auroc_values)),
        "test_auroc_std": float(np.std(test_auroc_values)),
        "test_auroc_percent_mean": float(np.mean(test_auroc_values) * 100.0),
        "test_auroc_percent_std": float(np.std(test_auroc_values) * 100.0),
        "test_minority_f1_mean": float(np.mean(test_f1_values)),
        "test_minority_f1_std": float(np.std(test_f1_values)),
        "test_minority_f1_percent_mean": float(np.mean(test_f1_values) * 100.0),
        "test_minority_f1_percent_std": float(np.std(test_f1_values) * 100.0),
        "test_precision_mean": float(np.mean(test_precision_values)),
        "test_precision_std": float(np.std(test_precision_values)),
        "test_precision_percent_mean": float(np.mean(test_precision_values) * 100.0),
        "test_precision_percent_std": float(np.std(test_precision_values) * 100.0),
        "test_recall_mean": float(np.mean(test_recall_values)),
        "test_recall_std": float(np.std(test_recall_values)),
        "test_recall_percent_mean": float(np.mean(test_recall_values) * 100.0),
        "test_recall_percent_std": float(np.std(test_recall_values) * 100.0),
    }
    metrics_out = {
        "features_path": str(features_path),
        "model_name": model_name,
        "primary_metric": "auprc",
        "threshold": selected_threshold,
        "auto_threshold": float(np.mean(auto_thresholds)),
        "threshold_strategy": "auto_from_validation",
        "early_stopping_rounds": int(early_stopping_rounds),
        "seeds": seeds,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "tuning": tuning_result,
        "xgb_device": xgb_device,
        "xgb_matrix_kind": matrices.matrix_kind,
        "xgb_feature_count": matrices.n_features,
        "preprocessing_seconds": matrices.transform_seconds,
        "dmatrix_seconds": matrices.dmatrix_seconds,
        "runs": all_metrics,
        "summary": summary,
    }
    metrics_path = output_dir / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    # Generate bar charts
    chart_path = _plot_metrics_bar_chart(
        all_metrics, summary, output_dir, model_name
    )

    return TrainResult(
        model_name=model_name,
        output_dir=output_dir,
        metrics_path=metrics_path,
        model_paths=model_paths,
        test_auprc_mean=metrics_out["summary"]["test_auprc_mean"],
    )
