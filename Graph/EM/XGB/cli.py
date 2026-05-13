"""Command-line entry point for the AML feature and XGBoost workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_CSV,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_FEATURE_ENGINEERING,
    DEFAULT_GFP_OUTPUT,
    PAPER_BATCH_SIZE,
    PAPER_XGB_SUCCESSIVE_HALVING,
)
from .features import FEATURE_ENGINEER_NAMES, build_feature_table
from .modeling import train_models
from .coem import CoemConfig, PnaConfig, train_coem


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aml_gfp_repro",
        description="Build AML feature tables and train an efficient XGBoost model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    features = subparsers.add_parser(
        "features", help="Generate a model-ready feature table from the AML CSV."
    )
    features.add_argument("--csv", type=Path, default=DEFAULT_DATA_CSV)
    features.add_argument("--out", type=Path, default=DEFAULT_GFP_OUTPUT)
    features.add_argument(
        "--feature-engineering",
        choices=FEATURE_ENGINEER_NAMES,
        default=DEFAULT_FEATURE_ENGINEERING,
        help="Feature engineering backend to apply before training.",
    )
    features.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE)
    features.add_argument("--write-chunk-size", type=int, default=65536)
    features.add_argument("--num-threads", type=int, default=None)
    features.add_argument(
        "--graph-ready",
        action="store_true",
        help="Retain edge_id/src_node_id/dst_node_id for graph expert training.",
    )
    features.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows after chronological sorting for smoke feature builds.",
    )

    train = subparsers.add_parser(
        "train", help="Train the GPU-oriented XGBoost baseline."
    )
    train.add_argument("--features", type=Path, default=DEFAULT_GFP_OUTPUT)
    train.add_argument("--out-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR / "xgboost")
    train.add_argument(
        "--model",
        choices=["xgboost"],
        default="xgboost",
        help="Only xgboost is supported.",
    )
    train.add_argument("--seeds", type=_parse_seeds, default=[0])
    train.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed classification threshold. If omitted, XGBoost uses 0.5.",
    )
    train.add_argument("--n-jobs", type=int, default=12)
    train.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help="Validation early-stopping patience. Use 0 to disable.",
    )
    train.add_argument(
        "--xgb-device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Use gpu_hist/gpu_predictor or the CPU hist path.",
    )
    train.add_argument("--verbose", action="store_true")

    train.add_argument("--tune-xgb", action="store_true")
    train.add_argument(
        "--paper-size",
        choices=["small", "medium", "large"],
        default="small",
        help="Paper successive-halving budget preset used with --tune-xgb.",
    )
    train.add_argument("--xgb-x0", type=int, default=None)
    train.add_argument("--xgb-eta", type=float, default=None)
    train.add_argument("--xgb-r0", type=float, default=None)
    train.add_argument(
        "--drop-engineered-features",
        action="store_true",
        help="Ignore gfp_* and gad_* columns in the feature table for training.",
    )
    train.add_argument(
        "--drop-gfp",
        action="store_true",
        dest="drop_engineered_features",
        help=argparse.SUPPRESS,
    )

    coem = subparsers.add_parser(
        "coem",
        help="Train calibrated table-GNN Co-EM with XGBoost and Multi-PNA-EU experts.",
    )
    coem.add_argument("--features", type=Path, required=True)
    coem.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "coem_multi_pna_eu",
    )
    coem.add_argument("--seeds", type=_parse_seeds, default=[0])
    coem.add_argument("--rounds", type=int, default=2)
    coem.add_argument("--patience", type=int, default=1)
    coem.add_argument(
        "--em-order",
        choices=["gnn-first", "table-first"],
        default="table-first",
    )
    coem.add_argument(
        "--graph-expert",
        choices=["multi-pna-eu"],
        default="multi-pna-eu",
    )
    coem.add_argument("--table-pl-weight", type=float, default=0.0)
    coem.add_argument("--gnn-pl-weight", type=float, default=0.1)
    coem.add_argument("--teacher-min-confidence", type=float, default=0.0)
    coem.add_argument("--teacher-max-edges-per-class", type=int, default=20000)
    coem.add_argument(
        "--gnn-pseudo-source",
        choices=["train", "valid", "both"],
        default="both",
    )
    coem.add_argument(
        "--table-gnn-feature-mode",
        choices=["none", "scores", "embeddings", "all"],
        default="all",
    )
    coem.add_argument(
        "--calibration-method",
        choices=["platt", "temperature", "none"],
        default="platt",
    )
    coem.add_argument("--oof-folds", type=int, default=5)
    coem.add_argument("--fusion-weight-start", type=float, default=0.0)
    coem.add_argument("--fusion-weight-stop", type=float, default=1.0)
    coem.add_argument("--fusion-weight-step", type=float, default=0.05)
    coem.add_argument("--table-n-estimators", type=int, default=None)
    coem.add_argument(
        "--table-early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
    )
    coem.add_argument("--xgb-device", choices=["gpu", "cpu"], default="gpu")
    coem.add_argument("--n-jobs", type=int, default=12)
    coem.add_argument("--max-rows", type=int, default=None)
    coem.add_argument("--gnn-warmup-epochs", type=int, default=2)
    coem.add_argument("--gnn-epochs", type=int, default=1)
    coem.add_argument("--pna-hidden-dim", type=int, default=32)
    coem.add_argument("--pna-node-embedding-dim", type=int, default=32)
    coem.add_argument("--pna-edge-embedding-dim", type=int, default=32)
    coem.add_argument("--pna-layers", type=int, default=2)
    coem.add_argument("--pna-towers", type=int, default=5)
    coem.add_argument("--pna-batch-size", type=int, default=1024)
    coem.add_argument(
        "--pna-num-neighbors",
        type=str,
        default="15,10",
        help="Comma-separated neighbor fanouts, e.g. 15,10.",
    )
    coem.add_argument("--pna-learning-rate", type=float, default=0.001)
    coem.add_argument("--pna-weight-decay", type=float, default=0.0001)
    coem.add_argument("--pna-dropout", type=float, default=0.1)
    coem.add_argument("--pna-loader-num-workers", type=int, default=0)
    coem.add_argument("--pna-device", choices=["cuda", "cpu"], default="cuda")
    coem.add_argument("--disable-amp", action="store_true")
    coem.add_argument("--grad-clip-norm", type=float, default=1.0)
    coem.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "features":
        try:
            result = build_feature_table(
                args.csv,
                args.out,
                feature_engineering=args.feature_engineering,
                batch_size=args.batch_size,
                write_chunk_size=args.write_chunk_size,
                num_threads=args.num_threads,
                graph_ready=args.graph_ready,
                max_rows=args.max_rows,
            )
        except NotImplementedError as exc:
            parser.error(str(exc))
        print(f"Wrote {result.rows:,} rows to {result.output_path}")
        print(f"Feature engineering: {result.feature_engineering}")
        print(f"Generated feature count: {result.generated_feature_count}")
        print(f"Manifest: {result.manifest_path}")
        return 0

    if args.command == "train":
        preset = PAPER_XGB_SUCCESSIVE_HALVING[args.paper_size]
        xgb_x0 = args.xgb_x0 if args.xgb_x0 is not None else preset["x0"]
        xgb_eta = args.xgb_eta if args.xgb_eta is not None else preset["eta"]
        xgb_r0 = args.xgb_r0 if args.xgb_r0 is not None else preset["r0"]
        result = train_models(
            args.features,
            args.out_dir,
            model_name=args.model,
            seeds=args.seeds,
            threshold=args.threshold,
            n_jobs=args.n_jobs,
            tune_xgb=args.tune_xgb,
            xgb_x0=xgb_x0,
            xgb_eta=xgb_eta,
            xgb_r0=xgb_r0,
            xgb_device=args.xgb_device,
            early_stopping_rounds=args.early_stopping_rounds,
            drop_engineered_features=args.drop_engineered_features,
            verbose=args.verbose,
        )
        print(f"Metrics: {result.metrics_path}")
        print(f"Mean test AUPRC: {result.test_auprc_mean:.6f}")
        try:
            metrics_data = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            summary = metrics_data.get("summary", {})
            print(f"  AUPRC  : {summary.get('test_auprc_mean', 0):.4f} ± {summary.get('test_auprc_std', 0):.4f}")
            print(f"  AUROC  : {summary.get('test_auroc_mean', 0):.4f} ± {summary.get('test_auroc_std', 0):.4f}")
            print(f"  F1     : {summary.get('test_minority_f1_mean', 0):.4f} ± {summary.get('test_minority_f1_std', 0):.4f}")
            print(f"  Precision: {summary.get('test_precision_mean', 0):.4f} ± {summary.get('test_precision_std', 0):.4f}")
            print(f"  Recall : {summary.get('test_recall_mean', 0):.4f} ± {summary.get('test_recall_std', 0):.4f}")
        except Exception:
            pass
        for model_path in result.model_paths:
            print(f"Model: {model_path}")
        # Print chart paths if they exist
        chart_per_seed = result.output_dir / f"{result.model_name}_metrics_per_seed.png"
        chart_summary = result.output_dir / f"{result.model_name}_metrics_summary.png"
        if chart_per_seed.exists():
            print(f"Chart (per seed): {chart_per_seed}")
        if chart_summary.exists():
            print(f"Chart (summary) : {chart_summary}")
        return 0

    if args.command == "coem":
        pna_config = PnaConfig(
            hidden_dim=args.pna_hidden_dim,
            node_embedding_dim=args.pna_node_embedding_dim,
            edge_embedding_dim=args.pna_edge_embedding_dim,
            num_layers=args.pna_layers,
            towers=args.pna_towers,
            batch_size=args.pna_batch_size,
            num_neighbors=tuple(
                int(part.strip())
                for part in args.pna_num_neighbors.split(",")
                if part.strip()
            ),
            learning_rate=args.pna_learning_rate,
            weight_decay=args.pna_weight_decay,
            dropout=args.pna_dropout,
            loader_num_workers=args.pna_loader_num_workers,
            amp=not args.disable_amp,
            grad_clip_norm=args.grad_clip_norm,
            device=args.pna_device,
        )
        result = train_coem(
            args.features,
            args.out_dir,
            seeds=args.seeds,
            config=CoemConfig(
                rounds=args.rounds,
                patience=args.patience,
                em_order=args.em_order,
                graph_expert=args.graph_expert,
                table_pl_weight=args.table_pl_weight,
                gnn_pl_weight=args.gnn_pl_weight,
                teacher_min_confidence=args.teacher_min_confidence,
                teacher_max_edges_per_class=args.teacher_max_edges_per_class,
                gnn_pseudo_source=args.gnn_pseudo_source,
                table_gnn_feature_mode=args.table_gnn_feature_mode,
                calibration_method=args.calibration_method,
                oof_folds=args.oof_folds,
                fusion_weight_start=args.fusion_weight_start,
                fusion_weight_stop=args.fusion_weight_stop,
                fusion_weight_step=args.fusion_weight_step,
                table_n_estimators=args.table_n_estimators,
                table_early_stopping_rounds=args.table_early_stopping_rounds,
                gnn_warmup_epochs=args.gnn_warmup_epochs,
                gnn_epochs=args.gnn_epochs,
                max_rows=args.max_rows,
                xgb_device=args.xgb_device,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
                pna=pna_config,
            ),
        )
        print(f"Metrics: {result.metrics_path}")
        print(
            "Best: "
            f"seed={result.best_seed} round={result.best_round} "
            f"head={result.best_head} valid_auprc={result.best_validation_auprc:.6f}"
        )
        if result.best_test_auprc is not None:
            print(f"Best test AUPRC: {result.best_test_auprc:.6f}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
