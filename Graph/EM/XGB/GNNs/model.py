"""Multi-PNA-EU edge model adapted from the local Multi-GNN reference."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, PNAConv
from torch_geometric.utils import degree


class MultiPnaEuEdgeModel(nn.Module):
    """PNA edge classifier with per-layer edge-update MLPs.

    The hidden size follows the Multi-GNN PNA convention: it is rounded down to a
    multiple of the tower count, so the historical `hidden_dim=32, towers=5`
    configuration runs with an effective hidden width of 30.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        raw_edge_dim: int,
        config,
        deg_hist: torch.Tensor,
    ) -> None:
        super().__init__()
        towers = int(config.towers)
        hidden_dim = int(config.hidden_dim // towers * towers)
        if hidden_dim <= 0:
            raise ValueError("effective Multi-PNA-EU hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.edge_embedding_dim = int(config.edge_embedding_dim)
        self.node_embedding = nn.Embedding(int(num_nodes), int(config.node_embedding_dim))
        self.node_projection = nn.Linear(int(config.node_embedding_dim), hidden_dim)
        self.edge_encoder = nn.Linear(int(raw_edge_dim), hidden_dim)
        self.convs = nn.ModuleList()
        self.edge_update_mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        for _ in range(int(config.num_layers)):
            self.convs.append(
                PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg_hist,
                    edge_dim=hidden_dim,
                    towers=towers,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )
            )
            self.edge_update_mlps.append(
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.batch_norms.append(BatchNorm(hidden_dim))
        self.dropout = nn.Dropout(float(config.dropout))
        self.edge_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, 50),
            nn.ReLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(25, self.edge_embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(self.edge_embedding_dim, 1)

    def forward(
        self,
        batch: Data,
        edge_label_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(batch, "n_id"):
            raise RuntimeError("Multi-PNA-EU batches must include global n_id")
        x = self.node_projection(self.node_embedding(batch.n_id))
        edge_attr = self.edge_encoder(batch.edge_attr)
        src, dst = batch.edge_index
        for conv, edge_update_mlp, batch_norm in zip(
            self.convs, self.edge_update_mlps, self.batch_norms
        ):
            node_delta = conv(x, batch.edge_index, edge_attr)
            x = (x + F.relu(batch_norm(node_delta))) / 2.0
            x = self.dropout(x)
            edge_delta = edge_update_mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
            edge_attr = (edge_attr + edge_delta) / 2.0

        seed_edge_attr = self.edge_encoder(edge_label_attr)
        seed_edge_attr = self._prefer_sampled_seed_edges(batch, seed_edge_attr, edge_attr)
        edge_src = x[batch.edge_label_index[0]]
        edge_dst = x[batch.edge_label_index[1]]
        edge_embedding = self.edge_head(torch.cat([edge_src, edge_dst, seed_edge_attr], dim=-1))
        logits = self.classifier(edge_embedding).squeeze(-1)
        return logits, edge_embedding

    def _prefer_sampled_seed_edges(
        self,
        batch: Data,
        encoded_seed_attr: torch.Tensor,
        updated_message_attr: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(batch, "e_id") or not hasattr(batch, "edge_label"):
            return encoded_seed_attr
        if batch.e_id.numel() == 0 or batch.edge_label.numel() == 0:
            return encoded_seed_attr
        message_ids = batch.e_id.to(encoded_seed_attr.device)
        seed_ids = batch.edge_label.to(encoded_seed_attr.device)
        order = torch.argsort(message_ids)
        sorted_ids = message_ids[order]
        positions = torch.searchsorted(sorted_ids, seed_ids)
        safe_positions = positions.clamp(max=max(0, sorted_ids.numel() - 1))
        in_range = positions < sorted_ids.numel()
        matches = in_range & sorted_ids[safe_positions].eq(seed_ids)
        if not bool(matches.any()):
            return encoded_seed_attr
        seed_attr = encoded_seed_attr.clone()
        seed_attr[matches] = updated_message_attr[order[safe_positions[matches]]]
        return seed_attr


def build_edge_model(
    *,
    architecture: str,
    num_nodes: int,
    raw_edge_dim: int,
    config,
    edge_index: torch.Tensor,
) -> nn.Module:
    if architecture != "multi-pna-eu":
        raise ValueError("only the 'multi-pna-eu' graph expert is restored")
    deg_hist = _degree_histogram(edge_index, int(num_nodes))
    return MultiPnaEuEdgeModel(
        num_nodes=int(num_nodes),
        raw_edge_dim=int(raw_edge_dim),
        config=config,
        deg_hist=deg_hist,
    )


def _degree_histogram(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    node_degree = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    max_degree = int(node_degree.max().item()) if node_degree.numel() else 0
    return torch.bincount(node_degree, minlength=max_degree + 1)
