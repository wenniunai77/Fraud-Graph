"""Constants for AML feature engineering and XGBoost experiments."""

from __future__ import annotations

from pathlib import Path

# ---- Default I/O paths ----
DEFAULT_DATA_CSV = Path("data/raw/HI-Small_Trans.csv")
DEFAULT_GFP_OUTPUT = Path("artifacts/hi_small_gfp.parquet")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_FEATURE_ENGINEERING = "gfp"

SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR

PAPER_BATCH_SIZE = 128
DEFAULT_EARLY_STOPPING_ROUNDS = 0

PAPER_GFP_PARAMS = {
    "num_threads": 12,
    "time_window": -1,
    "max_no_edges": -1,
    "fan": False,
    "degree": False,
    "scatter-gather": True,
    "scatter-gather_tw": 6 * SECONDS_PER_HOUR,
    "temp-cycle": True,
    "temp-cycle_tw": SECONDS_PER_DAY,
    "lc-cycle": True,
    "lc-cycle_tw": SECONDS_PER_DAY,
    "lc-cycle_len": 10,
    "vertex_stats": True,
    "vertex_stats_tw": SECONDS_PER_DAY,
    # Fraud-style edge-list columns begin with
    # [edge_id, src, dst, timestamp_seconds, amount_paid].
    "vertex_stats_cols": [3, 4],
}

PAPER_TEMPORAL_SPLIT = (0.60, 0.20, 0.20)

PAPER_XGBOOST_VERSION = "1.7.6"

PAPER_XGB_SUCCESSIVE_HALVING = {
    "small": {"x0": 1000, "eta": 2.0, "r0": 0.1},
    "medium": {"x0": 100, "eta": 2.0, "r0": 0.1},
    "large": {"x0": 16, "eta": 2.0, "r0": 0.2},
}

PAPER_HI_SMALL_XGB_F1 = 0.6323

DEFAULT_XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 8,
    "learning_rate": 0.03,
    "reg_lambda": 10.0,
    "scale_pos_weight": 6.0,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
}
