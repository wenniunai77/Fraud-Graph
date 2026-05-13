"""Modular graph experts for AML edge classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from .GNNs import build_edge_model


GRAPH_ID_COLUMNS = ("src_node_id", "dst_node_id")
GRAPH_EDGE_FEATURE_PREFIXES = ("gfp_", "gad_")


class GraphExpert(Protocol):
    def fit(
        self,
        *,
        edge_ids: np.ndarray,
        labels: np.ndarray,
        gold_mask: np.ndarray,
        teacher_prob: np.ndarray | None,
        teacher_mask: np.ndarray | None,
        pseudo_weight: float,
        epochs: int,
    ) -> dict[str, float]:
        ...

    def predict(self, edge_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def save_checkpoint(self, path: Path) -> None:
        ...


@dataclass(frozen=True)
class PnaConfig:
    hidden_dim: int = 64
    node_embedding_dim: int = 64
    edge_embedding_dim: int = 32
    num_layers: int = 2
    towers: int = 4
    batch_size: int = 4096
    num_neighbors: tuple[int, ...] = (15, 10)
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.1
    loader_num_workers: int = 0
    amp: bool = True
    grad_clip_norm: float = 1.0
    device: str = "cuda"


class PnaEdgeExpert:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        config: PnaConfig,
        seed: int,
        architecture: str = "multi-pna-eu",
    ) -> None:
        _require_torch_geometric_neighbor_sampling()
        self.config = config
        self.seed = int(seed)
        self.architecture = str(architecture)
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PNA graph expert requested CUDA but torch.cuda is unavailable")

        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.edge_feature_columns = _edge_feature_columns(frame)
        self.edge_index_label = _edge_label_index(frame)
        normalization_mask = (frame["split"] == "train").to_numpy(dtype=bool)
        edge_feature_values, edge_feature_center, edge_feature_scale = _prepare_edge_features(
            frame,
            self.edge_feature_columns,
            normalization_mask=normalization_mask,
            normalization="zscore",
        )
        self.edge_feature_center = edge_feature_center
        self.edge_feature_scale = edge_feature_scale
        self.edge_features = torch.as_tensor(
            edge_feature_values,
            dtype=torch.float32,
        )
        self.labels = torch.as_tensor(
            frame["label"].to_numpy(dtype=np.float32, copy=True),
            dtype=torch.float32,
        )
        self.num_edges = int(len(frame))
        self.num_nodes = int(
            max(frame["src_node_id"].max(), frame["dst_node_id"].max()) + 1
        )

        message_edge_index = torch.cat(
            [self.edge_index_label, self.edge_index_label.flip(0)],
            dim=1,
        )
        message_edge_attr = torch.cat([self.edge_features, self.edge_features], dim=0)
        self.data = Data(
            edge_index=message_edge_index,
            edge_attr=message_edge_attr,
            num_nodes=self.num_nodes,
        )
        self.model = build_edge_model(
            architecture=self.architecture,
            num_nodes=self.num_nodes,
            raw_edge_dim=len(self.edge_feature_columns),
            config=config,
            edge_index=message_edge_index,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        pos = float(self.labels.sum().item())
        neg = float(self.num_edges - pos)
        if pos <= 0:
            raise ValueError("PNA graph expert requires at least one positive edge")
        self.pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=self.device)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(config.amp and self.device.type == "cuda")
        )

    def fit(
        self,
        *,
        edge_ids: np.ndarray,
        labels: np.ndarray,
        gold_mask: np.ndarray,
        teacher_prob: np.ndarray | None,
        teacher_mask: np.ndarray | None,
        pseudo_weight: float,
        epochs: int,
    ) -> dict[str, float]:
        del labels
        if epochs <= 0:
            raise ValueError("epochs must be positive for PNA training")
        if not 0.0 <= pseudo_weight <= 1.0:
            raise ValueError("pseudo_weight must be in [0, 1]")
        gold_mask_t = torch.as_tensor(gold_mask.astype(np.bool_), dtype=torch.bool)
        if teacher_prob is None:
            teacher_prob_t = torch.zeros(self.num_edges, dtype=torch.float32)
        else:
            teacher_prob_t = torch.as_tensor(teacher_prob.astype(np.float32), dtype=torch.float32)
        if teacher_mask is None:
            teacher_mask_t = torch.zeros(self.num_edges, dtype=torch.bool)
        else:
            teacher_mask_t = torch.as_tensor(teacher_mask.astype(np.bool_), dtype=torch.bool)
        train_edge_ids = np.asarray(edge_ids, dtype=np.int64)
        if train_edge_ids.ndim != 1 or train_edge_ids.size == 0:
            raise ValueError("edge_ids must be a non-empty 1-D array")

        history: list[float] = []
        self.model.train()
        for _ in range(int(epochs)):
            epoch_loss = 0.0
            seen = 0
            for batch in self._loader(train_edge_ids, shuffle=True):
                row_ids_cpu = batch.edge_label.long()
                row_ids = row_ids_cpu.to(self.device, non_blocking=True)
                batch = batch.to(self.device, non_blocking=True)
                edge_label_attr = self.edge_features[row_ids_cpu].to(
                    self.device,
                    non_blocking=True,
                )
                labels_t = self.labels[row_ids_cpu].to(self.device, non_blocking=True)
                gold = gold_mask_t[row_ids_cpu].to(self.device, non_blocking=True)
                pseudo = teacher_mask_t[row_ids_cpu].to(self.device, non_blocking=True)
                teacher = teacher_prob_t[row_ids_cpu].to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(
                    enabled=bool(self.config.amp and self.device.type == "cuda")
                ):
                    logits, _ = self.model(batch, edge_label_attr)
                    loss = self._loss(
                        logits=logits,
                        labels=labels_t,
                        gold_mask=gold,
                        teacher_prob=teacher,
                        teacher_mask=pseudo,
                        pseudo_weight=pseudo_weight,
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.config.grad_clip_norm),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                batch_size = int(row_ids.numel())
                epoch_loss += float(loss.detach().cpu()) * batch_size
                seen += batch_size
            if seen == 0:
                raise RuntimeError("PNA loader produced no training edges")
            history.append(epoch_loss / seen)
        return {
            "train_loss": float(history[-1]),
            "train_loss_first": float(history[0]),
            "epochs": float(epochs),
        }

    @torch.no_grad()
    def predict(self, edge_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(edge_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("edge_ids must be a non-empty 1-D array")
        logits = np.empty(ids.shape[0], dtype=np.float32)
        embeddings = np.empty(
            (ids.shape[0], self.config.edge_embedding_dim),
            dtype=np.float32,
        )
        direct_positions = (
            ids.shape[0] == self.num_edges
            and ids[0] == 0
            and ids[-1] == self.num_edges - 1
            and np.all(ids == np.arange(self.num_edges, dtype=np.int64))
        )
        write_position = None if direct_positions else {
            int(edge_id): pos for pos, edge_id in enumerate(ids)
        }
        self.model.eval()
        for batch in self._loader(ids, shuffle=False):
            row_ids_cpu = batch.edge_label.long()
            batch = batch.to(self.device, non_blocking=True)
            edge_label_attr = self.edge_features[row_ids_cpu].to(
                self.device,
                non_blocking=True,
            )
            batch_logits, batch_embeddings = self.model(batch, edge_label_attr)
            batch_logits_np = batch_logits.detach().cpu().numpy().astype(np.float32)
            batch_embeddings_np = (
                batch_embeddings.detach().cpu().numpy().astype(np.float32)
            )
            if direct_positions:
                out_idx = row_ids_cpu.numpy()
                logits[out_idx] = batch_logits_np
                embeddings[out_idx] = batch_embeddings_np
            else:
                if write_position is None:
                    raise RuntimeError("internal prediction indexing error")
                for local_idx, edge_id in enumerate(row_ids_cpu.tolist()):
                    out_idx = write_position[int(edge_id)]
                    logits[out_idx] = batch_logits_np[local_idx]
                    embeddings[out_idx] = batch_embeddings_np[local_idx]
        return logits, embeddings

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "seed": self.seed,
                "architecture": self.architecture,
                "edge_feature_columns": self.edge_feature_columns,
                "edge_feature_center": self.edge_feature_center,
                "edge_feature_scale": self.edge_feature_scale,
                "num_nodes": self.num_nodes,
            },
            path,
        )

    def _loader(self, edge_ids: np.ndarray, *, shuffle: bool) -> LinkNeighborLoader:
        edge_ids_t = torch.as_tensor(edge_ids, dtype=torch.long)
        edge_label_index = self.edge_index_label[:, edge_ids_t]
        return LinkNeighborLoader(
            self.data,
            num_neighbors=list(self.config.num_neighbors),
            edge_label_index=edge_label_index,
            edge_label=edge_ids_t,
            batch_size=int(self.config.batch_size),
            shuffle=shuffle,
            num_workers=int(self.config.loader_num_workers),
            pin_memory=self.device.type == "cuda",
        )

    def _loss(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gold_mask: torch.Tensor,
        teacher_prob: torch.Tensor,
        teacher_mask: torch.Tensor,
        pseudo_weight: float,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        weights: list[float] = []
        if bool(gold_mask.any()):
            gold_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[gold_mask],
                labels[gold_mask],
                pos_weight=self.pos_weight,
            )
            losses.append(gold_loss)
            weights.append(1.0 - pseudo_weight if bool(teacher_mask.any()) else 1.0)
        if pseudo_weight > 0.0 and bool(teacher_mask.any()):
            pseudo_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[teacher_mask],
                teacher_prob[teacher_mask],
            )
            losses.append(pseudo_loss)
            weights.append(pseudo_weight if bool(gold_mask.any()) else 1.0)
        if not losses:
            raise RuntimeError("batch has neither gold labels nor pseudo-labels")
        total = torch.zeros((), dtype=logits.dtype, device=logits.device)
        for loss, weight in zip(losses, weights):
            total = total + float(weight) * loss
        return total


def _edge_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if str(column).startswith(GRAPH_EDGE_FEATURE_PREFIXES)
    ]
    if not columns:
        raise ValueError(
            "graph expert requires gfp_* or gad_* edge feature columns; "
            "regenerate features with --feature-engineering gfp"
        )
    return columns


def _edge_label_index(frame: pd.DataFrame) -> torch.Tensor:
    missing = [column for column in GRAPH_ID_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "graph-ready feature table is missing graph ID columns: "
            f"{missing}. Regenerate with `features --graph-ready`."
        )
    src = frame["src_node_id"].to_numpy(dtype=np.int64, copy=True)
    dst = frame["dst_node_id"].to_numpy(dtype=np.int64, copy=True)
    if src.min(initial=0) < 0 or dst.min(initial=0) < 0:
        raise ValueError("src_node_id and dst_node_id must be non-negative")
    return torch.as_tensor(np.stack([src, dst], axis=0), dtype=torch.long)


def _prepare_edge_features(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    normalization_mask: np.ndarray,
    normalization: str = "zscore",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if normalization == "none":
        values = frame[columns].to_numpy(dtype=np.float32, copy=True)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            values.astype(np.float32, copy=False),
            np.zeros(len(columns), dtype=np.float32),
            np.ones(len(columns), dtype=np.float32),
        )
    if normalization != "zscore":
        raise ValueError("edge feature normalization must be 'zscore' or 'none'")
    return _standardize_edge_features(
        frame,
        columns,
        normalization_mask=normalization_mask,
    )


def _standardize_edge_features(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    normalization_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values64 = frame[columns].to_numpy(dtype=np.float64, copy=True)
    values64 = np.nan_to_num(values64, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.asarray(normalization_mask, dtype=bool)
    if mask.shape[0] != values64.shape[0]:
        raise ValueError("normalization_mask length must match frame rows")
    if not bool(mask.any()):
        raise ValueError("edge feature normalization requires at least one training row")
    train_values = values64[mask]
    center = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    standardized = (values64 - center) / scale
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=10.0, neginf=-10.0)
    standardized = np.clip(standardized, -10.0, 10.0)
    return (
        standardized.astype(np.float32, copy=False),
        center.astype(np.float32, copy=False),
        scale.astype(np.float32, copy=False),
    )


def _require_torch_geometric_neighbor_sampling() -> None:
    try:
        import pyg_lib  # noqa: F401
        import torch_sparse  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNA Co-EM requires PyG neighbor sampling dependencies "
            "`pyg_lib` and `torch_sparse`; install them for the active Torch/CUDA build."
        ) from exc
