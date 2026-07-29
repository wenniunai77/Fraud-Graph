#!/usr/bin/env python3
"""R4-Full default implementation for internal graph embedding."""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv


SEED = 42
RP_DIM = 256
RP_SEED = 260715
HIDDEN_DIM = 64
EMBEDDING_DIM = 64
CLASSIFIER_HIDDEN_DIM = 32
DROPOUT = 0.5
FANOUTS = [25, 10]
FULL_GRAPH_BATCH_SIZE = 50000
SEEN_NORMAL_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 201
ORIGIN_WEIGHT = 2.0
ETA_INIT = -2.0
SEEN_CLASS_ID = 1
ERF_SKETCH_ID = 0


def seed_all():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_graph(path):
    payload = np.load(path)
    return (
        payload["x"].astype(np.float32),
        torch.from_numpy(payload["edge_index"].astype(np.int64)),
        payload["normal_idx"].astype(np.int64),
        payload["seen_idx"].astype(np.int64),
        payload["deployment_idx"].astype(np.int64),
    )


def build_rp_context(x):
    x = x.copy()
    if sp.issparse(x):
        x.data = np.sign(x.data) * np.log1p(np.abs(x.data))
    else:
        x = np.sign(x) * np.log1p(np.abs(x))
    x = normalize(x, norm="l2", axis=1)
    projector = SparseRandomProjection(
        n_components=RP_DIM,
        dense_output=True,
        random_state=RP_SEED,
    )
    return normalize(
        projector.fit_transform(x), norm="l2", axis=1
    ).astype(np.float32)


def canonicalize_basis(basis):
    pivots = np.abs(basis).argmax(axis=0)
    signs = np.sign(basis[pivots, np.arange(basis.shape[1])])
    signs[signs == 0] = 1
    return basis * signs


def erf_seed(dataset_name):
    dataset_code = int.from_bytes(
        hashlib.sha256(dataset_name.encode()).digest()[:4], "little"
    )
    sequence = np.random.SeedSequence(
        [RP_SEED, dataset_code, SEEN_CLASS_ID, SEED, ERF_SKETCH_ID]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def build_erf_root(x, normal_idx, dataset_name):
    reference = (
        x[normal_idx].toarray().astype(np.float64)
        if sp.issparse(x)
        else x[normal_idx].astype(np.float64)
    )
    _, singular, vh = scipy.linalg.svd(
        reference,
        full_matrices=False,
        lapack_driver="gesdd",
    )
    tolerance = (
        max(reference.shape)
        * np.finfo(np.float32).eps
        * singular[0]
    )
    rank = int((singular > tolerance).sum())
    basis = canonicalize_basis(vh[:rank].T)

    rng = np.random.default_rng(erf_seed(dataset_name))
    dimension = x.shape[1]
    permutation = rng.permutation(dimension)
    signs = rng.choice(np.array([-1.0, 1.0]), size=dimension)
    gv = signs[:, None] * basis[permutation]
    residual = gv - basis @ (basis.T @ gv)
    transform = basis + np.sqrt(dimension / rank) * residual

    root = np.asarray(x @ transform, dtype=np.float32)
    scale = np.sqrt(
        np.mean(np.sum(np.square(root[normal_idx]), axis=1))
    )
    return root / scale


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class R4Full(nn.Module):
    def __init__(self, root_dim):
        super().__init__()
        self.conv1 = SAGEConv((RP_DIM, root_dim), HIDDEN_DIM)
        self.conv2 = SAGEConv(HIDDEN_DIM, HIDDEN_DIM)
        self.projector = MLP(
            HIDDEN_DIM, EMBEDDING_DIM, EMBEDDING_DIM
        )
        self.classifier = MLP(
            EMBEDDING_DIM, CLASSIFIER_HIDDEN_DIM, 1
        )
        self.eta = nn.Parameter(torch.tensor(ETA_INIT))

    def forward_sampled(self, context, root, adjs):
        edge_index, _, size = adjs[0]
        hidden = self.conv1(
            (context, root[:size[1]]),
            edge_index,
        )
        hidden = F.dropout(
            F.relu(hidden), p=DROPOUT, training=self.training
        )

        edge_index, _, size = adjs[1]
        hidden = self.conv2(
            (hidden, hidden[:size[1]]),
            edge_index,
        )
        embedding = self.projector(hidden)
        logit = self.classifier(embedding).squeeze(-1)
        return embedding, logit


def build_loader(edge_index):
    return NeighborSampler(
        edge_index,
        node_idx=None,
        sizes=FANOUTS,
        batch_size=FULL_GRAPH_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def forward_all(model, loader, context_all, root_all, device):
    embeddings = []
    logits = []

    for _, node_ids, adjs in loader:
        adjs = [adj.to(device) for adj in adjs]
        context = context_all[node_ids].to(device)
        root = root_all[node_ids].to(device)
        embedding, logit = model.forward_sampled(
            context, root, adjs
        )
        embeddings.append(embedding)
        logits.append(logit)

    return torch.cat(embeddings), torch.cat(logits)


def r4_loss(model, logit, normal_idx, seen_idx, deployment_idx):
    normal_batch = normal_idx[
        torch.randperm(
            len(normal_idx), device=normal_idx.device
        )[:SEEN_NORMAL_BATCH_SIZE]
    ]
    supervised_idx = torch.cat([normal_batch, seen_idx])
    supervised_target = torch.cat([
        torch.zeros(len(normal_batch), device=logit.device),
        torch.ones(len(seen_idx), device=logit.device),
    ])
    seen_loss = F.binary_cross_entropy_with_logits(
        logit[supervised_idx],
        supervised_target,
    )

    anchor = (
        torch.logsumexp(logit[normal_idx], dim=0)
        - math.log(len(normal_idx))
    )
    relative_logit = logit - anchor
    origin_logit = torch.logaddexp(
        F.logsigmoid(-model.eta),
        F.logsigmoid(model.eta) + relative_logit,
    )
    origin_loss = (
        0.5
        * F.binary_cross_entropy_with_logits(
            origin_logit[normal_idx],
            torch.zeros(len(normal_idx), device=logit.device),
        )
        + 0.5
        * F.binary_cross_entropy_with_logits(
            origin_logit[deployment_idx],
            torch.ones(len(deployment_idx), device=logit.device),
        )
    )
    return seen_loss + ORIGIN_WEIGHT * origin_loss


def train_r4(x, edge_index, normal, seen, deployment, dataset_name, device):
    context = torch.from_numpy(build_rp_context(x)).float()
    root = torch.from_numpy(
        build_erf_root(x, normal, dataset_name)
    ).float()
    normal_idx = torch.from_numpy(normal).to(device)
    seen_idx = torch.from_numpy(seen).to(device)
    deployment_idx = torch.from_numpy(deployment).to(device)

    loader = build_loader(edge_index)
    model = R4Full(root.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        _, logit = forward_all(
            model, loader, context, root, device
        )
        loss = r4_loss(
            model,
            logit,
            normal_idx,
            seen_idx,
            deployment_idx,
        )
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"epoch={epoch:03d} loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        embedding, logit = forward_all(
            model, loader, context, root, device
        )
        score = torch.sigmoid(logit)
    return model, embedding, logit, score


def save_outputs(output_dir, model, embedding, logit, score):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "r4_embedding.npz",
        embedding=embedding.cpu().numpy(),
        logit=logit.cpu().numpy(),
        score=score.cpu().numpy(),
        node_id=np.arange(len(score), dtype=np.int64),
    )
    torch.save(model.state_dict(), output_dir / "r4_model.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all()
    x, edge_index, normal, seen, deployment = load_graph(
        args.input
    )
    model, embedding, logit, score = train_r4(
        x,
        edge_index,
        normal,
        seen,
        deployment,
        args.dataset_name,
        torch.device(args.device),
    )
    save_outputs(
        args.output_dir,
        model,
        embedding,
        logit,
        score,
    )


if __name__ == "__main__":
    main()
