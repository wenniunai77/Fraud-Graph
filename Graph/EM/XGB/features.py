"""Feature table generation for AML transaction CSV files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_FEATURE_ENGINEERING,
    PAPER_BATCH_SIZE,
    PAPER_GFP_PARAMS,
    SECONDS_PER_DAY,
)


CSV_RENAME = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "label",
}

FEATURE_BASE_COLUMNS = [
    "row_id",
    "timestamp",
    "timestamp_seconds",
    "hour",
    "day_of_week",
    "from_bank",
    "to_bank",
    "amount_received",
    "amount_paid",
    "receiving_currency",
    "payment_currency",
    "payment_format",
    "split",
    "label",
]

GRAPH_ID_COLUMNS = [
    "edge_id",
    "src_node_id",
    "dst_node_id",
]

GFP_RAW_INPUT_COLUMNS = [
    "EdgeID",
    "Source",
    "Target",
    "Timestamp",
    "Amount Paid",
    "Amount Received",
    "log_amount_paid",
    "log_amount_received",
    "amount_abs_diff",
    "currency_mismatch",
    "same_bank",
    "self_loop",
    "Payment Currency ID",
    "Receiving Currency ID",
    "Payment Format ID",
    "From Bank ID",
    "To Bank ID",
    "hour",
    "minute",
    "day_index",
]

GFP_MODEL_INPUT_NAMES = [
    "gfp_input_timestamp_seconds",
    "gfp_input_amount_paid",
    "gfp_input_amount_received",
    "gfp_input_log_amount_paid",
    "gfp_input_log_amount_received",
    "gfp_input_amount_abs_diff",
    "gfp_input_currency_mismatch",
    "gfp_input_same_bank",
    "gfp_input_self_loop",
    "gfp_input_payment_currency_id",
    "gfp_input_receiving_currency_id",
    "gfp_input_payment_format_id",
    "gfp_input_from_bank_id",
    "gfp_input_to_bank_id",
    "gfp_input_hour",
    "gfp_input_minute",
    "gfp_input_day_index",
]


@dataclass(frozen=True)
class FeatureBuildResult:
    output_path: Path
    manifest_path: Path
    rows: int
    feature_engineering: str
    generated_feature_count: int
    label_counts: dict[str, int]

    @property
    def gfp_feature_count(self) -> int:
        if self.feature_engineering != "gfp":
            return 0
        return self.generated_feature_count


@dataclass(frozen=True)
class FeatureBuildContext:
    transactions: pd.DataFrame
    base_frame: pd.DataFrame
    edge_array: np.ndarray | None
    edge_model_input_names: list[str]
    vertex_count: int | None
    batch_size: int
    write_chunk_size: int
    num_threads: int | None


@dataclass(frozen=True)
class FeatureBatch:
    start: int
    end: int
    frame: pd.DataFrame


@dataclass
class FeatureEngineer:
    name: str
    feature_prefix: str | None = None
    requires_graph: bool = False
    implemented: bool = True
    manifest_values: dict[str, Any] = field(default_factory=dict)

    def iter_batches(self, context: FeatureBuildContext) -> Iterable[FeatureBatch]:
        raise NotImplementedError

    def manifest(self) -> dict[str, Any]:
        return dict(self.manifest_values)


class NoFeatureEngineer(FeatureEngineer):
    def __init__(self) -> None:
        super().__init__(name="none")

    def iter_batches(self, context: FeatureBuildContext) -> Iterable[FeatureBatch]:
        for start, end in _iter_batches(
            len(context.base_frame),
            context.write_chunk_size,
        ):
            empty = pd.DataFrame(index=np.arange(end - start))
            yield FeatureBatch(start=start, end=end, frame=empty)


class GfpFeatureEngineer(FeatureEngineer):
    def __init__(self) -> None:
        super().__init__(name="gfp", feature_prefix="gfp_", requires_graph=True)

    def iter_batches(self, context: FeatureBuildContext) -> Iterable[FeatureBatch]:
        if context.edge_array is None:
            raise RuntimeError("GFP feature engineering requires a graph edge array.")
        if not context.edge_model_input_names:
            raise RuntimeError("GFP feature engineering requires model input names.")

        GraphFeaturePreprocessor = _import_snapml_gfp()
        params = dict(PAPER_GFP_PARAMS)
        if context.num_threads is not None:
            params["num_threads"] = int(context.num_threads)
        self.manifest_values["gfp_params"] = params
        self.manifest_values["raw_input_cols"] = GFP_RAW_INPUT_COLUMNS
        self.manifest_values["model_drops"] = ["EdgeID", "Source", "Target"]

        gfp = GraphFeaturePreprocessor()
        gfp.set_params(params)

        model_column_names: list[str] | None = None
        model_column_count = 0
        n_rows = len(context.transactions)
        for start, end in _iter_batches(n_rows, context.batch_size):
            transformed = gfp.transform(context.edge_array[start:end])
            model_values = transformed[:, 3:].astype(
                np.float32,
                copy=False,
            )
            if model_column_names is None:
                generated_count = model_values.shape[1] - len(context.edge_model_input_names)
                if generated_count < 0:
                    raise RuntimeError(
                        "GraphFeaturePreprocessor returned fewer columns than "
                        "the Fraud-style model inputs."
                    )
                model_column_names = list(context.edge_model_input_names) + [
                    f"gfp_{idx:04d}" for idx in range(generated_count)
                ]
                model_column_count = model_values.shape[1]
                self.manifest_values["gfp_raw_model_input_count"] = int(
                    len(context.edge_model_input_names)
                )
                self.manifest_values["gfp_generated_feature_count"] = int(generated_count)
                self.manifest_values["gfp_model_feature_count"] = int(model_column_count)
            if model_values.shape[1] != model_column_count:
                raise RuntimeError(
                    "GraphFeaturePreprocessor returned a different number of "
                    f"features: expected {model_column_count}, got {model_values.shape[1]}"
                )
            if end == n_rows or end % 500000 < context.batch_size:
                print(f"GFP processed {end:,}/{n_rows:,} rows")
            yield FeatureBatch(
                start=start,
                end=end,
                frame=pd.DataFrame(model_values, columns=model_column_names),
            )


class GadFeatureEngineer(FeatureEngineer):
    def __init__(self) -> None:
        super().__init__(
            name="gad",
            feature_prefix="gad_",
            requires_graph=True,
            implemented=False,
        )

    def iter_batches(self, context: FeatureBuildContext) -> Iterable[FeatureBatch]:
        del context
        raise NotImplementedError(
            "GAD feature engineering is registered as an adapter slot but is not "
            "implemented yet. Add the GAD transformation in GadFeatureEngineer."
        )


FEATURE_ENGINEERS: dict[str, type[FeatureEngineer]] = {
    "none": NoFeatureEngineer,
    "gfp": GfpFeatureEngineer,
    "gad": GadFeatureEngineer,
}
FEATURE_ENGINEER_NAMES = tuple(FEATURE_ENGINEERS)


def _import_snapml_gfp():
    try:
        from snapml import GraphFeaturePreprocessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing strict dependency 'snapml'. The GFP path requires "
            "snapml.GraphFeaturePreprocessor. Use --feature-engineering none "
            "to build a base feature table without Snap ML."
        ) from exc

    return GraphFeaturePreprocessor


def _read_transactions(csv_path: Path) -> pd.DataFrame:
    dtype = {
        "From Bank": "string",
        "Account": "string",
        "To Bank": "string",
        "Account.1": "string",
        "Amount Received": "float32",
        "Receiving Currency": "string",
        "Amount Paid": "float32",
        "Payment Currency": "string",
        "Payment Format": "string",
        "Is Laundering": "int8",
    }
    df = pd.read_csv(csv_path, dtype=dtype)
    missing = sorted(set(CSV_RENAME) - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.rename(columns=CSV_RENAME)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%Y/%m/%d %H:%M", errors="raise"
    )
    df = df.sort_values(["timestamp", "row_id"], kind="mergesort").reset_index(drop=True)
    min_ts = df["timestamp"].iloc[0]
    df["timestamp_seconds"] = (
        (df["timestamp"] - min_ts).dt.total_seconds().astype("int64")
    )
    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["minute"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).astype(
        "int16"
    )
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["day_index"] = np.floor(
        df["timestamp_seconds"].to_numpy(dtype=np.float64) / SECONDS_PER_DAY
    ).astype(np.int16)
    return df


def _account_codes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    src = df["from_bank"].astype("string") + ":" + df["from_account"].astype("string")
    dst = df["to_bank"].astype("string") + ":" + df["to_account"].astype("string")
    codes, uniques = pd.factorize(pd.concat([src, dst], ignore_index=True), sort=True)
    n_rows = len(df)
    return codes[:n_rows].astype(np.float64), codes[n_rows:].astype(np.float64), len(uniques)


def _chronological_split_points(day_index: np.ndarray) -> tuple[int, int, list[list[int]]]:
    days, counts = np.unique(day_index.astype(np.int64), return_counts=True)
    if len(days) < 3:
        n_rows = int(counts.sum())
        return int(n_rows * 0.60), int(n_rows * 0.80), [days.tolist(), [], []]

    targets = np.asarray([0.6, 0.2, 0.2], dtype=np.float64)
    total = float(counts.sum())
    best: tuple[float, int, int] | None = None
    for i in range(1, len(counts)):
        for j in range(i + 1, len(counts) + 1):
            split_counts = np.asarray(
                [counts[:i].sum(), counts[i:j].sum(), counts[j:].sum()],
                dtype=np.float64,
            )
            score = float(np.max(np.abs((split_counts / total) - targets) / targets))
            if best is None or score < best[0]:
                best = (score, i, j)
    if best is None:
        n_rows = int(counts.sum())
        return int(n_rows * 0.60), int(n_rows * 0.80), [days.tolist(), [], []]

    _, i, j = best
    train_end = int(counts[:i].sum())
    valid_end = int(counts[:j].sum())
    split_days = [days[:i].tolist(), days[i:j].tolist(), days[j:].tolist()]
    return train_end, valid_end, split_days


def _split_labels(day_index: np.ndarray) -> np.ndarray:
    train_end, valid_end, _ = _chronological_split_points(day_index)
    n_rows = len(day_index)
    split = np.empty(n_rows, dtype=object)
    split[:train_end] = "train"
    split[train_end:valid_end] = "valid"
    split[valid_end:] = "test"
    return split


def _build_edge_array(df: pd.DataFrame) -> tuple[np.ndarray, int, list[str]]:
    src_codes, dst_codes, vertex_count = _account_codes(df)
    edge_id = np.arange(len(df), dtype=np.float64)
    timestamp_seconds = df["timestamp_seconds"].to_numpy(dtype=np.float64)
    amount_paid = df["amount_paid"].to_numpy(dtype=np.float64)
    amount_received = df["amount_received"].to_numpy(dtype=np.float64)
    from_bank = df["from_bank"].astype("string")
    to_bank = df["to_bank"].astype("string")
    src_account = from_bank + ":" + df["from_account"].astype("string")
    dst_account = to_bank + ":" + df["to_account"].astype("string")
    payment_currency_id = pd.factorize(df["payment_currency"].astype("string"), sort=True)[
        0
    ].astype(np.float64)
    receiving_currency_id = pd.factorize(
        df["receiving_currency"].astype("string"), sort=True
    )[0].astype(np.float64)
    payment_format_id = pd.factorize(df["payment_format"].astype("string"), sort=True)[
        0
    ].astype(np.float64)
    from_bank_id = pd.factorize(from_bank, sort=True)[0].astype(np.float64)
    to_bank_id = pd.factorize(to_bank, sort=True)[0].astype(np.float64)
    edge_array = np.column_stack(
        [
            edge_id,
            src_codes,
            dst_codes,
            timestamp_seconds,
            amount_paid,
            amount_received,
            np.log1p(np.maximum(amount_paid, 0)),
            np.log1p(np.maximum(amount_received, 0)),
            np.abs(amount_paid - amount_received),
            (payment_currency_id != receiving_currency_id).astype(np.float64),
            (from_bank.to_numpy(dtype=object) == to_bank.to_numpy(dtype=object)).astype(
                np.float64
            ),
            (src_account.to_numpy(dtype=object) == dst_account.to_numpy(dtype=object)).astype(
                np.float64
            ),
            payment_currency_id,
            receiving_currency_id,
            payment_format_id,
            from_bank_id,
            to_bank_id,
            df["hour"].to_numpy(dtype=np.float64),
            df["minute"].to_numpy(dtype=np.float64),
            df["day_index"].to_numpy(dtype=np.float64),
        ]
    )
    return edge_array, vertex_count, GFP_MODEL_INPUT_NAMES


def _base_feature_frame(
    df: pd.DataFrame,
    *,
    graph_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    values: dict[str, Any] = {
        "row_id": df["row_id"].astype("int64"),
        "timestamp": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_seconds": df["timestamp_seconds"].astype("int64"),
        "hour": df["hour"].astype("int8"),
        "day_of_week": df["day_of_week"].astype("int8"),
        "from_bank": pd.to_numeric(df["from_bank"], errors="raise").astype("int32"),
        "to_bank": pd.to_numeric(df["to_bank"], errors="raise").astype("int32"),
        "amount_received": df["amount_received"].astype("float32"),
        "amount_paid": df["amount_paid"].astype("float32"),
        "receiving_currency": df["receiving_currency"].astype("string"),
        "payment_currency": df["payment_currency"].astype("string"),
        "payment_format": df["payment_format"].astype("string"),
        "split": _split_labels(df["day_index"].to_numpy()),
        "label": df["label"].astype("int8"),
    }
    columns = list(FEATURE_BASE_COLUMNS)
    if graph_ids is not None:
        if graph_ids.shape != (len(df), 3):
            raise ValueError(
                "graph_ids must have shape (n_rows, 3) for edge/src/dst identifiers"
            )
        values.update(
            {
                "edge_id": graph_ids[:, 0].astype(np.int64, copy=False),
                "src_node_id": graph_ids[:, 1].astype(np.int64, copy=False),
                "dst_node_id": graph_ids[:, 2].astype(np.int64, copy=False),
            }
        )
        label_index = columns.index("label")
        columns = columns[:label_index] + GRAPH_ID_COLUMNS + columns[label_index:]
    out = pd.DataFrame(values)
    return out[columns]


def _iter_batches(n_rows: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def _feature_engineer(name: str) -> FeatureEngineer:
    try:
        engineer_cls = FEATURE_ENGINEERS[name]
    except KeyError as exc:
        valid = ", ".join(FEATURE_ENGINEER_NAMES)
        raise ValueError(f"unknown feature engineering '{name}'. Choose one of: {valid}") from exc
    return engineer_cls()


def build_feature_table(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    feature_engineering: str = DEFAULT_FEATURE_ENGINEERING,
    batch_size: int = PAPER_BATCH_SIZE,
    write_chunk_size: int = 65536,
    num_threads: int | None = None,
    graph_ready: bool = False,
    max_rows: int | None = None,
) -> FeatureBuildResult:
    """Build a model-ready feature table and write it as a Parquet file."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    engineer = _feature_engineer(feature_engineering)
    if not engineer.implemented:
        raise NotImplementedError(
            f"{engineer.name} feature engineering is registered as an adapter slot "
            "but is not implemented yet."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if write_chunk_size <= 0:
        raise ValueError("write_chunk_size must be positive")
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    edge_array: np.ndarray | None = None
    edge_model_input_names: list[str] = []
    vertex_count: int | None = None
    df = _read_transactions(csv_path)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        df = df.iloc[: int(max_rows)].reset_index(drop=True)
    if engineer.requires_graph or graph_ready:
        edge_array, vertex_count, edge_model_input_names = _build_edge_array(df)
    graph_ids = None if not graph_ready else edge_array[:, :3]
    base_df = _base_feature_frame(df, graph_ids=graph_ids)

    context = FeatureBuildContext(
        transactions=df,
        base_frame=base_df,
        edge_array=edge_array,
        edge_model_input_names=edge_model_input_names,
        vertex_count=vertex_count,
        batch_size=int(batch_size),
        write_chunk_size=int(write_chunk_size),
        num_threads=num_threads,
    )

    writer: pq.ParquetWriter | None = None
    pending_frames: list[pd.DataFrame] = []
    pending_rows = 0
    generated_feature_count: int | None = None

    def flush_pending() -> None:
        nonlocal writer, pending_rows
        if not pending_frames:
            return
        chunk = pd.concat(pending_frames, ignore_index=True)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        pending_frames.clear()
        pending_rows = 0

    try:
        for feature_batch in engineer.iter_batches(context):
            start, end = feature_batch.start, feature_batch.end
            if start < 0 or end > len(base_df) or start >= end:
                raise RuntimeError(f"invalid feature batch boundaries: {start}:{end}")
            if len(feature_batch.frame) != end - start:
                raise RuntimeError(
                    "feature engineering returned a batch with "
                    f"{len(feature_batch.frame)} rows for slice {start}:{end}"
                )
            feature_count = int(feature_batch.frame.shape[1])
            if generated_feature_count is None:
                generated_feature_count = feature_count
            elif generated_feature_count != feature_count:
                raise RuntimeError(
                    "feature engineering returned a different number of features: "
                    f"expected {generated_feature_count}, got {feature_count}"
                )
            base_chunk = base_df.iloc[start:end].reset_index(drop=True)
            extra_chunk = feature_batch.frame.reset_index(drop=True)
            if extra_chunk.shape[1] == 0:
                chunk_df = base_chunk
            else:
                chunk_df = pd.concat([base_chunk, extra_chunk], axis=1)
            pending_frames.append(chunk_df)
            pending_rows += len(chunk_df)
            if pending_rows >= write_chunk_size:
                flush_pending()
        flush_pending()
    finally:
        if writer is not None:
            writer.close()

    if generated_feature_count is None:
        generated_feature_count = 0

    label_counts = {
        str(k): int(v) for k, v in base_df["label"].value_counts().sort_index().items()
    }
    train_end, valid_end, split_days = _chronological_split_points(
        df["day_index"].to_numpy()
    )
    manifest = {
        "csv_path": str(csv_path),
        "output_path": str(output_path),
        "rows": int(len(df)),
        "max_rows": None if max_rows is None else int(max_rows),
        "vertices": None if vertex_count is None else int(vertex_count),
        "graph_ready": bool(graph_ready),
        "graph_id_columns": GRAPH_ID_COLUMNS if graph_ready else [],
        "batch_size": int(batch_size),
        "write_chunk_size": int(write_chunk_size),
        "feature_engineering": engineer.name,
        "feature_prefix": engineer.feature_prefix,
        "feature_engineering_params": engineer.manifest(),
        "base_columns": FEATURE_BASE_COLUMNS,
        "generated_feature_count": int(generated_feature_count),
        "gfp_feature_count": (
            int(generated_feature_count) if engineer.name == "gfp" else 0
        ),
        "label_counts": label_counts,
        "split_counts": {
            str(k): int(v) for k, v in base_df["split"].value_counts().items()
        },
        "split": {
            "train_rows": int(train_end),
            "valid_rows": int(valid_end - train_end),
            "test_rows": int(len(df) - valid_end),
            "train_days": split_days[0],
            "valid_days": split_days[1],
            "test_days": split_days[2],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return FeatureBuildResult(
        output_path=output_path,
        manifest_path=manifest_path,
        rows=len(df),
        feature_engineering=engineer.name,
        generated_feature_count=generated_feature_count,
        label_counts=label_counts,
    )


def build_gfp_features(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = PAPER_BATCH_SIZE,
    write_chunk_size: int = 65536,
    num_threads: int | None = None,
) -> FeatureBuildResult:
    return build_feature_table(
        csv_path,
        output_path,
        feature_engineering="gfp",
        batch_size=batch_size,
        write_chunk_size=write_chunk_size,
        num_threads=num_threads,
    )
